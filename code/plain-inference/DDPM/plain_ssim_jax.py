import argparse
import json
import time
from pathlib import Path

import torch

import ssim_common as common


VARIANT_CHOICES = ("plain1jax", "plain3jax")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--sample_batch_size", type=int, default=100)
    parser.add_argument("--sampling_method", type=str, default="ddim", choices=("ddim", "ddpm"))
    parser.add_argument("--ddim_timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_image_scale", type=int, default=10)
    parser.add_argument("--grid_rows", type=int, default=10)
    parser.add_argument("--metrics_batch_size", type=int, default=256)
    parser.add_argument("--data_root", type=str, default=str(common.DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--classifier_checkpoint",
        type=str,
        default=str(common.DEFAULT_CLASSIFIER_CHECKPOINT),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--variant", type=str, required=True, choices=VARIANT_CHOICES)
    parser.add_argument("--generated_raw_dir", type=str, default="")
    parser.add_argument("--ssim_num_samples", type=int, default=0)
    parser.add_argument("--ssim_window_size", type=int, default=11)
    parser.add_argument("--ssim_sigma", type=float, default=1.5)
    return parser


def print_extra_config(args):
    print(f"===> source_root : {common.SOURCE_ROOT}")
    print(f"===> variant : {args.variant}")
    print(f"===> model_path : {args.model_path}")
    print(f"===> save_dir : {args.save_dir}")
    print(f"===> num_samples : {args.num_samples}")
    print(f"===> sample_batch_size : {args.sample_batch_size}")
    print(f"===> sampling_method : {args.sampling_method}")
    print(f"===> ddim_timesteps : {args.ddim_timesteps}")
    print(f"===> metrics_batch_size : {args.metrics_batch_size}")
    print(f"===> data_root : {args.data_root}")
    print(f"===> device : {args.device}")
    print(f"===> generated_raw_dir : {args.generated_raw_dir}")
    print(f"===> ssim_num_samples : {args.ssim_num_samples}")
    print(f"===> ssim_window_size : {args.ssim_window_size}")
    print(f"===> ssim_sigma : {args.ssim_sigma}")


def main(args):
    args = common.apply_mnist10_small_defaults(args)
    print_extra_config(args)

    generated_raw_dir = Path(args.generated_raw_dir) if args.generated_raw_dir else None
    model_path = Path(args.model_path) if args.model_path else None
    if generated_raw_dir is None:
        raise ValueError(
            "This lightweight SSIM script only supports --generated_raw_dir. "
            "It intentionally avoids importing plain.py to bypass the "
            "torchvision/transformers environment conflict."
        )

    save_dir = Path(args.save_dir)
    if not args.save_dir:
        save_dir = generated_raw_dir.parent.parent
    common.mkdir(str(save_dir))
    common.set_seed(args.seed)

    start_time = time.time()
    generated_images_01, labels = common.load_generated_images_from_raw(generated_raw_dir)
    if args.num_samples > 0:
        sample_count = min(int(args.num_samples), generated_images_01.shape[0])
        generated_images_01 = generated_images_01[:sample_count]
        labels = labels[:sample_count]
    print(f"[INFO] loaded {generated_images_01.shape[0]} generated images from {generated_raw_dir}")
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
        "variant": args.variant,
        "ssim_mean": float(ssim_mean),
        "ssim_std": float(ssim_std),
        "ssim_num_samples": int(ssim_num_samples),
        "num_samples": int(labels.shape[0]),
        "sampling_method": args.sampling_method,
        "ddim_timesteps": int(args.ddim_timesteps),
        "sample_batch_size": int(args.sample_batch_size),
        "metrics_batch_size": int(args.metrics_batch_size),
        "model_path": "" if model_path is None else str(model_path),
        "generated_raw_dir": str(generated_raw_dir),
        "save_dir": str(save_dir),
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
        f"[DONE] variant={args.variant} samples={labels.shape[0]} "
        f"ssim_mean={ssim_mean:.6f} ssim_std={ssim_std:.6f}"
    )


if __name__ == "__main__":
    main(get_parser().parse_args())
