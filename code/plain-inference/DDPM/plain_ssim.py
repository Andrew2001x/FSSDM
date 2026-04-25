import argparse
import json
import time
from pathlib import Path

import torch

import ssim_common as common


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_raw_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics_batch_size", type=int, default=256)
    parser.add_argument("--ssim_num_samples", type=int, default=0)
    parser.add_argument("--data_root", type=str, default=str(common.DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--classifier_checkpoint",
        type=str,
        default=str(common.DEFAULT_CLASSIFIER_CHECKPOINT),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ssim_window_size", type=int, default=11)
    parser.add_argument("--ssim_sigma", type=float, default=1.5)
    return parser


def print_extra_config(args):
    print(f"===> source_root : {common.SOURCE_ROOT}")
    print(f"===> model_path :")
    print(f"===> save_dir : {args.save_dir}")
    print(f"===> num_samples : {args.num_samples}")
    print(f"===> metrics_batch_size : {args.metrics_batch_size}")
    print(f"===> data_root : {args.data_root}")
    print(f"===> classifier_checkpoint : {args.classifier_checkpoint}")
    print(f"===> device : {args.device}")
    print(f"===> generated_raw_dir : {args.generated_raw_dir}")
    print(f"===> ssim_num_samples : {args.ssim_num_samples}")
    print(f"===> ssim_window_size : {args.ssim_window_size}")
    print(f"===> ssim_sigma : {args.ssim_sigma}")


def main(args):
    args = common.apply_mnist10_small_defaults(args)
    args.model_path = ""
    if not args.save_dir:
        raw_dir = Path(args.generated_raw_dir)
        args.save_dir = str(raw_dir.parent.parent)

    print_extra_config(args)

    save_dir = Path(args.save_dir)
    common.mkdir(str(save_dir))
    common.set_seed(args.seed)
    start_time = time.time()

    generated_images_01, labels = common.load_generated_images_from_raw(Path(args.generated_raw_dir))
    if args.num_samples > 0:
        sample_count = min(int(args.num_samples), generated_images_01.shape[0])
        generated_images_01 = generated_images_01[:sample_count]
        labels = labels[:sample_count]
    print(f"[INFO] loaded {generated_images_01.shape[0]} generated images from {args.generated_raw_dir}")

    real_images_01 = common.collect_testset_by_labels(labels, Path(args.data_root))

    ssim_num_samples = int(args.ssim_num_samples)
    if ssim_num_samples > 0:
        ssim_num_samples = min(ssim_num_samples, real_images_01.shape[0], generated_images_01.shape[0])
        real_images_for_ssim = real_images_01[:ssim_num_samples]
        generated_images_for_ssim = generated_images_01[:ssim_num_samples]
    else:
        ssim_num_samples = int(min(real_images_01.shape[0], generated_images_01.shape[0]))
        real_images_for_ssim = real_images_01[:ssim_num_samples]
        generated_images_for_ssim = generated_images_01[:ssim_num_samples]

    device = torch.device(common.normalize_torch_device(args.device))
    ssim_mean, ssim_std = common.calculate_ssim_score(
        real_images_for_ssim,
        generated_images_for_ssim,
        device=device,
        batch_size=args.metrics_batch_size,
        window_size=args.ssim_window_size,
        sigma=args.ssim_sigma,
    )

    metrics = {
        "variant": "plain",
        "ssim_mean": float(ssim_mean),
        "ssim_std": float(ssim_std),
        "ssim_num_samples": int(ssim_num_samples),
        "num_samples": int(labels.shape[0]),
        "generated_raw_dir": str(Path(args.generated_raw_dir)),
        "save_dir": str(save_dir),
        "metrics_batch_size": int(args.metrics_batch_size),
        "data_root": str(Path(args.data_root)),
        "pairing": "labelwise_ordered_testset",
        "ssim_window_size": int(max(3, args.ssim_window_size + (1 - args.ssim_window_size % 2))),
        "ssim_sigma": float(args.ssim_sigma),
        "elapsed_seconds": float(time.time() - start_time),
    }

    (save_dir / "ssim_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[DONE] variant=plain samples={labels.shape[0]} "
        f"ssim_mean={ssim_mean:.6f} ssim_std={ssim_std:.6f}"
    )


if __name__ == "__main__":
    main(get_parser().parse_args())
