import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor

import plain as base


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_raw_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics_batch_size", type=int, default=256)
    parser.add_argument("--kid_num_samples", type=int, default=0)
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    parser.add_argument("--kid_num_subsets", type=int, default=50)
    parser.add_argument("--data_root", type=str, default=str(base.DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--classifier_checkpoint",
        type=str,
        default=str(base.DEFAULT_CLASSIFIER_CHECKPOINT),
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def _polynomial_kernel(x: Tensor, y: Tensor) -> Tensor:
    dim = float(x.size(1))
    return ((x @ y.T) / dim + 1.0) ** 3


def _kid_mmd2_unbiased(x: Tensor, y: Tensor) -> float:
    m = x.size(0)
    n = y.size(0)
    if m < 2 or n < 2:
        raise ValueError(f"KID needs at least 2 samples per set, got m={m}, n={n}")

    k_xx = _polynomial_kernel(x, x)
    k_yy = _polynomial_kernel(y, y)
    k_xy = _polynomial_kernel(x, y)

    sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (m * (m - 1))
    sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (n * (n - 1))
    sum_xy = k_xy.mean()
    return float((sum_xx + sum_yy - 2.0 * sum_xy).item())


def compute_kid_from_features(
    real_features: Tensor,
    fake_features: Tensor,
    subset_size: int,
    num_subsets: int,
    seed: int,
) -> Tuple[float, float]:
    real_features = real_features.float()
    fake_features = fake_features.float()

    max_subset = min(real_features.size(0), fake_features.size(0))
    if max_subset < 2:
        raise ValueError(f"KID needs at least 2 features per set, got {max_subset}")

    subset_size = max(2, min(int(subset_size), max_subset))
    num_subsets = max(1, int(num_subsets))
    rng = np.random.default_rng(seed)
    scores = []

    for _ in range(num_subsets):
        real_idx = rng.choice(real_features.size(0), size=subset_size, replace=False)
        fake_idx = rng.choice(fake_features.size(0), size=subset_size, replace=False)
        scores.append(
            _kid_mmd2_unbiased(
                real_features[torch.from_numpy(real_idx).long()],
                fake_features[torch.from_numpy(fake_idx).long()],
            )
        )

    scores_np = np.asarray(scores, dtype=np.float64)
    return float(scores_np.mean()), float(scores_np.std(ddof=0))


def load_generated_images_from_raw(raw_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"generated raw dir not found: {raw_dir}")

    images = []
    labels = []
    label_dirs = sorted([path for path in raw_dir.iterdir() if path.is_dir()], key=lambda p: int(p.name))
    for label_dir in label_dirs:
        label = int(label_dir.name)
        for image_path in sorted(label_dir.glob("*.png")):
            image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
            images.append(image[None, :, :])
            labels.append(label)

    if not images:
        raise ValueError(f"no png images found under: {raw_dir}")

    return np.stack(images, axis=0), np.asarray(labels, dtype=np.int32)


def calculate_kid_score(
    real_images_01: np.ndarray,
    fake_images_01: np.ndarray,
    classifier,
    device: torch.device,
    batch_size: int,
    subset_size: int,
    num_subsets: int,
    seed: int,
):
    real_tensor = torch.from_numpy(real_images_01).float()
    fake_tensor = torch.from_numpy(fake_images_01).float()

    try:
        feature_extractor = base.InceptionV3Feature().to(device)
        feature_name = "inception_v3"
    except Exception as exc:
        print(f"[WARN] failed to load InceptionV3 weights, fallback to classifier features: {exc}")
        feature_extractor = base.ClassifierFeatureExtractor(classifier).to(device)
        feature_name = "mnist_classifier"

    real_features = base.extract_features(feature_extractor, real_tensor, device, batch_size, "[kid real]")
    fake_features = base.extract_features(feature_extractor, fake_tensor, device, batch_size, "[kid fake]")
    kid_mean, kid_std = compute_kid_from_features(
        real_features,
        fake_features,
        subset_size=subset_size,
        num_subsets=num_subsets,
        seed=seed,
    )
    return kid_mean, kid_std, feature_name


def print_extra_config(args):
    print(f"===> source_root : {base.SOURCE_ROOT}")
    print(f"===> jax.default_backend() : {base.jax.default_backend()}")
    print(f"===> jax.devices() : {base.jax.devices()}")
    print(f"===> model_path :")
    print(f"===> save_dir : {args.save_dir}")
    print(f"===> num_samples : {args.num_samples}")
    print(f"===> metrics_batch_size : {args.metrics_batch_size}")
    print(f"===> data_root : {args.data_root}")
    print(f"===> classifier_checkpoint : {args.classifier_checkpoint}")
    print(f"===> device : {args.device}")
    print(f"===> generated_raw_dir : {args.generated_raw_dir}")
    print(f"===> kid_num_samples : {args.kid_num_samples}")
    print(f"===> kid_subset_size : {args.kid_subset_size}")
    print(f"===> kid_num_subsets : {args.kid_num_subsets}")


def main(args):
    args = base.apply_mnist10_small_defaults(args)
    args.model_path = ""
    if not args.save_dir:
        raw_dir = Path(args.generated_raw_dir)
        args.save_dir = str(raw_dir.parent.parent)

    print_extra_config(args)

    save_dir = Path(args.save_dir)
    base.file_utils.mkdir(str(save_dir))
    base.set_seed(args.seed)
    start_time = time.time()

    generated_images_01, labels = load_generated_images_from_raw(Path(args.generated_raw_dir))
    if args.num_samples > 0:
        sample_count = min(int(args.num_samples), generated_images_01.shape[0])
        generated_images_01 = generated_images_01[:sample_count]
        labels = labels[:sample_count]
    print(f"[INFO] loaded {generated_images_01.shape[0]} generated images from {args.generated_raw_dir}")

    real_images_m11, _ = base.collect_testset(generated_images_01.shape[0], Path(args.data_root))
    real_images_01 = base.to_zero_one(real_images_m11)

    kid_num_samples = int(args.kid_num_samples)
    if kid_num_samples > 0:
        kid_num_samples = min(kid_num_samples, real_images_01.shape[0], generated_images_01.shape[0])
        real_images_for_kid = real_images_01[:kid_num_samples]
        generated_images_for_kid = generated_images_01[:kid_num_samples]
    else:
        kid_num_samples = int(min(real_images_01.shape[0], generated_images_01.shape[0]))
        real_images_for_kid = real_images_01[:kid_num_samples]
        generated_images_for_kid = generated_images_01[:kid_num_samples]

    device = torch.device(base.normalize_torch_device(args.device))
    classifier_checkpoint = Path(args.classifier_checkpoint)
    classifier = base.load_classifier_checkpoint(device, classifier_checkpoint)
    classifier_info = base.load_classifier_info(classifier_checkpoint)

    kid_mean, kid_std, feature_name = calculate_kid_score(
        real_images_for_kid,
        generated_images_for_kid,
        classifier,
        device,
        args.metrics_batch_size,
        subset_size=args.kid_subset_size,
        num_subsets=args.kid_num_subsets,
        seed=args.seed,
    )

    metrics = {
        "variant": "plain",
        "kid_mean": float(kid_mean),
        "kid_std": float(kid_std),
        "kid_num_samples": int(kid_num_samples),
        "kid_subset_size": int(min(args.kid_subset_size, kid_num_samples)),
        "kid_num_subsets": int(args.kid_num_subsets),
        "num_samples": int(labels.shape[0]),
        "generated_raw_dir": str(Path(args.generated_raw_dir)),
        "save_dir": str(save_dir),
        "metrics_batch_size": int(args.metrics_batch_size),
        "kid_feature_extractor": feature_name,
        "classifier_checkpoint": str(classifier_checkpoint),
        "classifier_info": classifier_info,
        "elapsed_seconds": float(time.time() - start_time),
    }

    (save_dir / "kid_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[DONE] variant=plain samples={labels.shape[0]} "
        f"kid_mean={kid_mean:.6f} kid_std={kid_std:.6f}"
    )


if __name__ == "__main__":
    main(get_parser().parse_args())
