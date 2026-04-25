import argparse
import inspect
import json
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.nn as nn
import torch.nn.functional as F
from flax.training import train_state
from PIL import Image
from torch import Tensor
from torchvision import utils
from torchvision.models import inception_v3
from tqdm import tqdm

from flax_train_tiny_mnist import SOURCE_ROOT, apply_mnist10_small_defaults
from flax_ddpm.script_utils import get_args
from flax_ddpm.script_utils import get_diffusion_from_args
from datasets.flax_tiny_mnist import MnistDataset
from tools import file_utils


DEFAULT_DATA_ROOT = Path(SOURCE_ROOT) / "datasets" / "mnist"
DEFAULT_CLASSIFIER_CHECKPOINT = Path(SOURCE_ROOT) / "classifier_out" / "best_classifier.pt"


def get_sample_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./model_eval")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--sample_batch_size", type=int, default=100)
    parser.add_argument("--sampling_method", type=str, default="ddim", choices=("ddim", "ddpm"))
    parser.add_argument("--ddim_timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_image_scale", type=int, default=10)
    parser.add_argument("--grid_rows", type=int, default=10)
    parser.add_argument("--metrics_batch_size", type=int, default=256)
    parser.add_argument("--fid_num_samples", type=int, default=0)
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--classifier_checkpoint",
        type=str,
        default=str(DEFAULT_CLASSIFIER_CHECKPOINT),
    )
    return parser


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_effective_config(args):
    keys = [
        "model_path",
        "save_dir",
        "num_samples",
        "sample_batch_size",
        "sampling_method",
        "num_timesteps",
        "ddim_timesteps",
        "fid_num_samples",
        "save_image_scale",
        "device",
        "data_root",
        "classifier_checkpoint",
        "img_size",
        "img_channels",
        "num_classes",
        "base_channels",
        "channel_mults",
        "num_res_blocks",
        "time_emb_dim",
    ]
    print(f"===> source_root : {SOURCE_ROOT}")
    print(f"===> jax.default_backend() : {jax.default_backend()}")
    print(f"===> jax.devices() : {jax.devices()}")
    for key in keys:
        print(f"===> {key} : {getattr(args, key)}")


def normalize_torch_device(device_name: str) -> str:
    device_name = str(device_name).strip().lower()
    if device_name == "gpu":
        return "cuda"
    return device_name


def load_torch_checkpoint(path: Path, map_location: Any = "cpu"):
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class MNISTCnnClassifier(nn.Module):
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc1 = nn.Linear(64 * 4 * 4, feat_dim)
        self.fc2 = nn.Linear(feat_dim, 10)

    def features(self, x: Tensor) -> Tensor:
        h = self.conv(x).flatten(1)
        return F.relu(self.fc1(h), inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.features(x))


class InceptionV3Feature(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import Inception_V3_Weights

            inception = inception_v3(
                weights=Inception_V3_Weights.DEFAULT,
                transform_input=False,
            )
        except Exception:
            try:
                inception = inception_v3(weights="DEFAULT", transform_input=False)
            except Exception:
                inception = inception_v3(pretrained=True, transform_input=False)
        self.inception = inception.eval()
        for param in self.inception.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)


class ClassifierFeatureExtractor(nn.Module):
    def __init__(self, classifier: MNISTCnnClassifier):
        super().__init__()
        self.classifier = classifier

    @torch.inference_mode()
    def forward(self, x: Tensor) -> Tensor:
        return self.classifier.features(x)


def load_classifier_checkpoint(device: torch.device, checkpoint_path: Path) -> MNISTCnnClassifier:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"classifier checkpoint not found: {checkpoint_path}")

    model = MNISTCnnClassifier().to(device=device, dtype=torch.float32)
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_classifier_info(checkpoint_path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"loaded_from": str(checkpoint_path)}
    metrics_path = checkpoint_path.parent / "metrics.json"
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return info

        if "best_test_acc" in payload:
            info["best_test_acc"] = payload["best_test_acc"]
        final_test = payload.get("final_test")
        if isinstance(final_test, dict) and "acc" in final_test:
            info["final_test_acc"] = final_test["acc"]
    return info


def create_state_and_diffusion(args):
    diffusion = get_diffusion_from_args(args)
    rng = jax.random.PRNGKey(args.seed)
    rng, x_rng = jax.random.split(rng)
    x = jax.random.normal(x_rng, (1, *args.img_size, args.img_channels))
    y = jnp.zeros((1,), dtype=jnp.int32)
    rng, state_rng = jax.random.split(rng)
    variables = diffusion.init(rngs=state_rng, rng=rng, x=x, y=y)
    optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=1e-4)
    state = train_state.TrainState.create(
        apply_fn=diffusion.apply,
        params=variables["params"],
        tx=optimizer,
    )
    return diffusion, state


def load_params(state, model_path: Path):
    with open(model_path, "rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        if "ema_params" in payload:
            return flax.serialization.from_state_dict(target=state.params, state=payload["ema_params"])
        if "params" in payload:
            return flax.serialization.from_state_dict(target=state.params, state=payload["params"])

    try:
        restored_state = flax.serialization.from_state_dict(target=state, state=payload)
        if hasattr(restored_state, "ema_params"):
            return restored_state.ema_params
        return restored_state.params
    except Exception:
        if isinstance(payload, dict) and "params" in payload:
            return flax.serialization.from_state_dict(target=state.params, state=payload["params"])
        raise


def create_jitted_sampler(diffusion, args):
    if args.sampling_method == "ddpm":
        @jax.jit
        def sampler(params, labels, rng):
            return diffusion.apply(
                {"params": params},
                batch_size=labels.shape[0],
                y=labels,
                use_ema=False,
                rng_key=rng,
                method=diffusion.sample,
            )
    else:
        ddim_timesteps = int(args.ddim_timesteps)

        @jax.jit
        def sampler(params, labels, rng):
            return diffusion.apply(
                {"params": params},
                batch_size=labels.shape[0],
                y=labels,
                use_ema=False,
                rng_key=rng,
                ddim_timesteps=ddim_timesteps,
                method=diffusion.sample_ddim,
            )

    return sampler


def sample_batch(sampler, params, labels: jnp.ndarray, rng: jax.Array):
    samples = sampler(params, labels, rng)
    samples = jax.device_get(samples)
    return np.asarray(samples, dtype=np.float32).transpose(0, 3, 1, 2)


def to_zero_one(x: np.ndarray) -> np.ndarray:
    return np.clip((x + 1.0) * 0.5, 0.0, 1.0)


def save_png(image: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(output_path, format="PNG")


def save_generated_images(
    images_01: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    image_scale: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = output_dir / "raw"
    view_root = output_dir / "view"
    image_scale = max(1, int(image_scale))

    for index in tqdm(range(images_01.shape[0]), desc="[save images]"):
        image = (images_01[index, 0] * 255.0).round().clip(0, 255).astype(np.uint8)
        label = int(labels[index])
        raw_path = raw_root / str(label) / f"{index:05d}.png"
        save_png(image, raw_path)

        if image_scale > 1:
            view_image = np.repeat(np.repeat(image, image_scale, axis=0), image_scale, axis=1)
        else:
            view_image = image
        view_path = view_root / str(label) / f"{index:05d}.png"
        save_png(view_image, view_path)


def save_grid(images_01: np.ndarray, output_path: Path, grid_rows: int):
    num_images = min(images_01.shape[0], grid_rows * grid_rows)
    if num_images == 0:
        return
    tensor = torch.from_numpy(images_01[:num_images]).float()
    grid = utils.make_grid(tensor, nrow=grid_rows, normalize=False, value_range=(0, 1))
    utils.save_image(grid, output_path)


@torch.no_grad()
def classifier_accuracy(model, images_m11: np.ndarray, labels: np.ndarray, device: torch.device, batch_size: int):
    model.eval()
    total_correct = 0
    total = 0
    predictions = []
    labels_tensor = torch.from_numpy(labels).long()

    for start in tqdm(range(0, images_m11.shape[0], batch_size), desc="[accuracy]"):
        end = min(start + batch_size, images_m11.shape[0])
        batch = torch.from_numpy(images_m11[start:end]).float().to(device=device, dtype=torch.float32)
        logits = model(batch)
        pred = logits.argmax(dim=1).cpu()
        predictions.append(pred)
        total_correct += (pred == labels_tensor[start:end]).sum().item()
        total += end - start

    predictions = torch.cat(predictions, dim=0).numpy()
    return total_correct / max(1, total), predictions


def mse_against_testset(fake_01: np.ndarray, real_01: np.ndarray):
    return float(((fake_01 - real_01) ** 2).mean(axis=(1, 2, 3)).mean())


def compute_activation_statistics(features: Tensor):
    mu = features.mean(dim=0)
    if features.size(0) <= 1:
        sigma = torch.eye(features.size(1), device=features.device, dtype=features.dtype)
    else:
        sigma = torch.cov(features.T)
    return mu, sigma


def _sqrt_matrix_product(a: Tensor, b: Tensor) -> Tensor:
    eigvals_a, eigvecs_a = torch.linalg.eigh(a)
    eigvals_b, eigvecs_b = torch.linalg.eigh(b)

    eigvals_a = torch.clamp(eigvals_a, min=1e-10)
    eigvals_b = torch.clamp(eigvals_b, min=1e-10)

    sqrt_a = eigvecs_a @ torch.diag(torch.sqrt(eigvals_a)) @ eigvecs_a.T
    sqrt_b = eigvecs_b @ torch.diag(torch.sqrt(eigvals_b)) @ eigvecs_b.T
    return sqrt_a @ sqrt_b


def compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps: float = 1e-10) -> float:
    diff = mu1 - mu2
    eye1 = torch.eye(sigma1.size(0), device=sigma1.device, dtype=sigma1.dtype)
    eye2 = torch.eye(sigma2.size(0), device=sigma2.device, dtype=sigma2.dtype)

    sigma1 = sigma1 + eps * eye1
    sigma2 = sigma2 + eps * eye2
    covmean = _sqrt_matrix_product(sigma1, sigma2)

    fid = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid.item())


@torch.inference_mode()
def extract_features(
    feature_extractor: nn.Module,
    images_01: Tensor,
    device: torch.device,
    batch_size: int,
    desc: str,
) -> Tensor:
    feature_extractor.eval()
    outputs: List[Tensor] = []
    for start in tqdm(range(0, images_01.size(0), batch_size), desc=desc):
        end = min(start + batch_size, images_01.size(0))
        batch = images_01[start:end].to(device=device, dtype=torch.float32)
        outputs.append(feature_extractor(batch).float().cpu())
    return torch.cat(outputs, dim=0)


def calculate_fid_score(
    real_images_01: np.ndarray,
    fake_images_01: np.ndarray,
    classifier: MNISTCnnClassifier,
    device: torch.device,
    batch_size: int,
):
    real_tensor = torch.from_numpy(real_images_01).float()
    fake_tensor = torch.from_numpy(fake_images_01).float()

    try:
        feature_extractor = InceptionV3Feature().to(device)
        feature_name = "inception_v3"
    except Exception as exc:
        print(f"[WARN] failed to load InceptionV3 weights, fallback to classifier features: {exc}")
        feature_extractor = ClassifierFeatureExtractor(classifier).to(device)
        feature_name = "mnist_classifier"

    real_features = extract_features(feature_extractor, real_tensor, device, batch_size, "[fid real]")
    fake_features = extract_features(feature_extractor, fake_tensor, device, batch_size, "[fid fake]")

    mu_real, sigma_real = compute_activation_statistics(real_features)
    mu_fake, sigma_fake = compute_activation_statistics(fake_features)
    return compute_fid(mu_real, sigma_real, mu_fake, sigma_fake), feature_name


def build_test_dataset(data_root: Path):
    kwargs = {
        "is_train": False,
        "target_labels": list(range(10)),
    }
    signature = inspect.signature(MnistDataset)
    if "root" in signature.parameters:
        kwargs["root"] = str(data_root)
    if "data_root" in signature.parameters:
        kwargs["data_root"] = str(data_root)
    return MnistDataset(**kwargs)


def collect_testset(num_samples: int, data_root: Path):
    test_dataset = build_test_dataset(data_root)
    sample_count = min(num_samples, len(test_dataset))
    real_images = []
    labels = []
    for index in range(sample_count):
        image, label = test_dataset[index]
        real_images.append(np.asarray(image, dtype=np.float32))
        labels.append(int(label))
    real_images = np.stack(real_images, axis=0)
    labels = np.asarray(labels, dtype=np.int32)
    return real_images, labels


def sample_all_images(sampler, params, labels: np.ndarray, args):
    generated_batches = []
    rng = jax.random.PRNGKey(args.seed)
    sample_start_time = time.time()

    for start in tqdm(range(0, labels.shape[0], args.sample_batch_size), desc="[sample]"):
        end = min(start + args.sample_batch_size, labels.shape[0])
        batch_labels = jnp.asarray(labels[start:end], dtype=jnp.int32)
        rng, batch_rng = jax.random.split(rng)
        batch_start_time = time.time()
        generated_batches.append(sample_batch(sampler, params, batch_labels, batch_rng))
        batch_elapsed = time.time() - batch_start_time
        if start == 0:
            print(
                f"[INFO] first sample batch took {batch_elapsed:.2f}s "
                f"(includes JAX compile/warmup)"
            )

    total_elapsed = time.time() - sample_start_time
    print(f"[INFO] total sample time: {total_elapsed:.2f}s")
    return np.concatenate(generated_batches, axis=0)


def main(args):
    args = apply_mnist10_small_defaults(args)
    print_effective_config(args)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model file not exist: {model_path}")

    save_dir = Path(args.save_dir)
    generated_dir = save_dir / "generated"
    file_utils.mkdir(str(save_dir))
    set_seed(args.seed)

    start_time = time.time()
    diffusion, state = create_state_and_diffusion(args)
    params = load_params(state, model_path)
    sampler = create_jitted_sampler(diffusion, args)

    real_images_m11, labels = collect_testset(args.num_samples, Path(args.data_root))
    generated_images_m11 = sample_all_images(sampler, params, labels, args)

    real_images_01 = to_zero_one(real_images_m11)
    generated_images_01 = to_zero_one(generated_images_m11)

    if args.save_images:
        save_generated_images(
            generated_images_01,
            labels,
            generated_dir,
            args.save_image_scale,
        )

    save_grid(generated_images_01, save_dir / "grid.png", args.grid_rows)
    np.savez_compressed(
        save_dir / "generated_samples.npz",
        images=generated_images_m11,
        labels=labels,
    )

    device = torch.device(normalize_torch_device(args.device))
    classifier_checkpoint = Path(args.classifier_checkpoint)
    classifier = load_classifier_checkpoint(device, classifier_checkpoint)
    classifier_info = load_classifier_info(classifier_checkpoint)
    accuracy, predictions = classifier_accuracy(
        classifier,
        generated_images_m11,
        labels,
        device,
        args.metrics_batch_size,
    )
    mse_value = mse_against_testset(generated_images_01, real_images_01)
    fid_num_samples = int(args.fid_num_samples)
    if fid_num_samples > 0:
        fid_num_samples = min(fid_num_samples, real_images_01.shape[0], generated_images_01.shape[0])
        real_images_for_fid = real_images_01[:fid_num_samples]
        generated_images_for_fid = generated_images_01[:fid_num_samples]
    else:
        fid_num_samples = int(real_images_01.shape[0])
        real_images_for_fid = real_images_01
        generated_images_for_fid = generated_images_01

    fid_value, fid_feature_extractor = calculate_fid_score(
        real_images_for_fid,
        generated_images_for_fid,
        classifier,
        device,
        args.metrics_batch_size,
    )

    metrics = {
        "accuracy": float(accuracy),
        "fid": float(fid_value),
        "mse": float(mse_value),
        "num_samples": int(labels.shape[0]),
        "sampling_method": args.sampling_method,
        "ddim_timesteps": int(args.ddim_timesteps),
        "sample_batch_size": int(args.sample_batch_size),
        "metrics_batch_size": int(args.metrics_batch_size),
        "fid_num_samples": int(fid_num_samples),
        "model_path": str(model_path),
        "save_dir": str(save_dir),
        "fid_feature_extractor": fid_feature_extractor,
        "classifier_checkpoint": str(classifier_checkpoint),
        "classifier_info": classifier_info,
        "elapsed_seconds": float(time.time() - start_time),
    }

    (save_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (save_dir / "predictions.json").write_text(
        json.dumps(
            {
                "labels": labels.tolist(),
                "predictions": predictions.tolist(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"[DONE] samples={labels.shape[0]} "
        f"accuracy={accuracy:.4f} fid={fid_value:.4f} mse={mse_value:.6f}"
    )


if __name__ == "__main__":
    parser = get_sample_arg_parser()
    main(get_args(parser))
