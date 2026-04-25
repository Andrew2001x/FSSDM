import os
import random
import struct
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def _is_valid_source_root(path):
    required_dirs = ("flax_ddpm", "datasets", "tools")
    return bool(path) and all(os.path.isdir(os.path.join(path, name)) for name in required_dirs)


def _iter_search_roots():
    yielded = set()

    env_candidates = [
        os.environ.get("DDPMSIMFLAX_SOURCE_ROOT"),
        os.environ.get("FLAX_DDPM_SOURCE_ROOT"),
        os.environ.get("PYTHONPATH"),
    ]
    for value in env_candidates:
        if not value:
            continue
        for raw_path in str(value).split(os.pathsep):
            path = os.path.abspath(raw_path.strip())
            if path and path not in yielded:
                yielded.add(path)
                yield path

    root = CURRENT_DIR
    for _ in range(5):
        if root not in yielded:
            yielded.add(root)
            yield root

        try:
            entries = sorted(os.listdir(root))
        except OSError:
            entries = []

        for entry in entries:
            child = os.path.join(root, entry)
            if os.path.isdir(child) and child not in yielded:
                yielded.add(child)
                yield child

        parent = os.path.abspath(os.path.join(root, ".."))
        if parent == root:
            break
        root = parent


def _find_source_root():
    checked = []
    for candidate in _iter_search_roots():
        checked.append(candidate)
        if _is_valid_source_root(candidate):
            return candidate

    checked_preview = "\n".join(f"  - {path}" for path in checked[:20])
    raise ModuleNotFoundError(
        "Could not locate a source tree containing flax_ddpm, datasets and tools.\n"
        "Set DDPMSIMFLAX_SOURCE_ROOT to the project root that contains those three folders.\n"
        f"Checked paths:\n{checked_preview}"
    )


SOURCE_ROOT = _find_source_root()


DEFAULT_DATA_ROOT = Path(SOURCE_ROOT) / "datasets" / "mnist"
DEFAULT_CLASSIFIER_CHECKPOINT = Path(SOURCE_ROOT) / "classifier_out" / "best_classifier.pt"


def apply_mnist10_small_defaults(args):
    args.img_channels = 1
    args.img_size = (28, 28)
    args.num_classes = 10
    args.num_timesteps = getattr(args, "num_timesteps", 1000)
    return args


def normalize_torch_device(device_name: str) -> str:
    device_name = str(device_name).strip().lower()
    if device_name == "gpu":
        return "cuda"
    return device_name


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_zero_one(x: np.ndarray) -> np.ndarray:
    return np.clip((x + 1.0) * 0.5, 0.0, 1.0)


def mkdir(dir_: str):
    Path(dir_).mkdir(parents=True, exist_ok=True)


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


def _find_mnist_raw_dir(data_root: Path) -> Path:
    data_root = Path(data_root)
    candidates = [
        data_root,
        data_root / "raw",
        data_root / "MNIST" / "raw",
        data_root / "MNIST",
    ]
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        image_file = candidate / "t10k-images-idx3-ubyte"
        label_file = candidate / "t10k-labels-idx1-ubyte"
        if image_file.exists() and label_file.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate MNIST raw files under "
        f"{data_root}. Expected t10k-images-idx3-ubyte and "
        "t10k-labels-idx1-ubyte."
    )


def _read_idx_images(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 16:
        raise ValueError(f"invalid IDX image file: {path}")
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"unexpected image magic {magic} in {path}")
    expected = 16 + count * rows * cols
    if len(data) != expected:
        raise ValueError(f"corrupted IDX image file {path}: expected {expected} bytes, got {len(data)}")
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    return images.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"invalid IDX label file: {path}")
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"unexpected label magic {magic} in {path}")
    expected = 8 + count
    if len(data) != expected:
        raise ValueError(f"corrupted IDX label file {path}: expected {expected} bytes, got {len(data)}")
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return labels.astype(np.int32)


def load_mnist_testset_zero_one(data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    raw_dir = _find_mnist_raw_dir(Path(data_root))
    images = _read_idx_images(raw_dir / "t10k-images-idx3-ubyte").astype(np.float32) / 255.0
    labels = _read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte")
    return images[:, None, :, :], labels


def collect_testset_by_labels(labels: np.ndarray, data_root: Path) -> np.ndarray:
    test_images_01, test_labels = load_mnist_testset_zero_one(data_root)
    label_buckets = {}
    for index, raw_label in enumerate(test_labels.tolist()):
        label = int(raw_label)
        label_buckets.setdefault(label, []).append(test_images_01[index])

    offsets = {label: 0 for label in label_buckets}
    real_images = []
    for raw_label in labels.tolist():
        label = int(raw_label)
        if label not in label_buckets:
            raise ValueError(f"label {label} not found in test dataset")
        offset = offsets[label]
        if offset >= len(label_buckets[label]):
            raise ValueError(
                f"not enough real test images for label {label}: "
                f"need more than {len(label_buckets[label])}"
            )
        real_images.append(label_buckets[label][offset])
        offsets[label] = offset + 1

    return np.stack(real_images, axis=0)


def build_gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    window_size = max(3, int(window_size))
    if window_size % 2 == 0:
        window_size += 1

    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size // 2)
    gauss = torch.exp(-(coords ** 2) / (2.0 * float(sigma) ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = torch.outer(gauss, gauss)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def filter_with_window(images: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    pad = window.size(-1) // 2
    padded = F.pad(images, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, window, groups=images.size(1))


def ssim_per_image(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    window: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_real = filter_with_window(real_images, window)
    mu_fake = filter_with_window(fake_images, window)

    mu_real_sq = mu_real.square()
    mu_fake_sq = mu_fake.square()
    mu_real_fake = mu_real * mu_fake

    sigma_real_sq = torch.clamp(filter_with_window(real_images.square(), window) - mu_real_sq, min=0.0)
    sigma_fake_sq = torch.clamp(filter_with_window(fake_images.square(), window) - mu_fake_sq, min=0.0)
    sigma_real_fake = filter_with_window(real_images * fake_images, window) - mu_real_fake

    numerator = (2.0 * mu_real_fake + c1) * (2.0 * sigma_real_fake + c2)
    denominator = (mu_real_sq + mu_fake_sq + c1) * (sigma_real_sq + sigma_fake_sq + c2)
    ssim_map = numerator / torch.clamp(denominator, min=1e-12)
    return ssim_map.mean(dim=(1, 2, 3))


@torch.inference_mode()
def calculate_ssim_score(
    real_images_01: np.ndarray,
    fake_images_01: np.ndarray,
    device: torch.device,
    batch_size: int,
    window_size: int,
    sigma: float,
) -> Tuple[float, float]:
    if real_images_01.shape != fake_images_01.shape:
        raise ValueError(
            f"real/fake image shapes must match for SSIM, got "
            f"{real_images_01.shape} vs {fake_images_01.shape}"
        )

    real_tensor = torch.from_numpy(real_images_01).float()
    fake_tensor = torch.from_numpy(fake_images_01).float()

    window = build_gaussian_window(
        window_size=window_size,
        sigma=sigma,
        channels=real_tensor.size(1),
        device=device,
        dtype=torch.float32,
    )

    outputs = []
    for start in range(0, real_tensor.size(0), batch_size):
        end = min(start + batch_size, real_tensor.size(0))
        real_batch = real_tensor[start:end].to(device=device, dtype=torch.float32)
        fake_batch = fake_tensor[start:end].to(device=device, dtype=torch.float32)
        outputs.append(ssim_per_image(real_batch, fake_batch, window).cpu())

    scores = torch.cat(outputs, dim=0).numpy().astype(np.float64)
    return float(scores.mean()), float(scores.std(ddof=0))
