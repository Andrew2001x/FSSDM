import os
import sys
import pickle
import logging
import time
import multiprocessing as mp


DEFAULT_VISIBLE_GPU = os.environ.get("DDPMSIMFLAX_GPU", "4")
os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_VISIBLE_GPU

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax.core import FrozenDict
from torch.utils.data import DataLoader, Dataset


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
logger = logging.getLogger(__name__)


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

        for entry in entries:
            child = os.path.join(root, entry)
            if not os.path.isdir(child):
                continue
            try:
                sub_entries = sorted(os.listdir(child))
            except OSError:
                sub_entries = []
            for sub_entry in sub_entries:
                grandchild = os.path.join(child, sub_entry)
                if os.path.isdir(grandchild) and grandchild not in yielded:
                    yielded.add(grandchild)
                    yield grandchild

        parent = os.path.abspath(os.path.join(root, ".."))
        if parent == root:
            break
        root = parent


def _find_source_root():
    checked = []
    for candidate in _iter_search_roots():
        checked.append(candidate)
        if _is_valid_source_root(candidate):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate

    checked_preview = "\n".join(f"  - {path}" for path in checked[:20])
    raise ModuleNotFoundError(
        "Could not locate a source tree containing flax_ddpm, datasets and tools.\n"
        "Set DDPMSIMFLAX_SOURCE_ROOT to the project root that contains those three folders.\n"
        f"Checked paths:\n{checked_preview}"
    )


SOURCE_ROOT = _find_source_root()

from flax_ddpm import script_utils
from flax_ddpm.script_utils import get_args
from flax_ddpm.script_utils import get_diffusion_from_args
from datasets.flax_tiny_mnist import MnistDataset
from tools import file_utils


def _normalize_int_tuple(value):
    if isinstance(value, tuple):
        if all(isinstance(v, int) for v in value):
            return tuple(value)
        value = "".join(str(v) for v in value)
    elif isinstance(value, list):
        if all(isinstance(v, int) for v in value):
            return tuple(value)
        value = ",".join(str(v) for v in value)

    if isinstance(value, str):
        cleaned = value.strip().strip("()[]")
        if not cleaned:
            return tuple()
        return tuple(int(part.strip()) for part in cleaned.split(",") if part.strip())

    return (int(value),)


def apply_mnist10_small_defaults(args):
    args.img_channels = 1
    args.img_size = (28, 28)
    args.num_classes = 10
    args.num_timesteps = 1000

    args.channel_mults = _normalize_int_tuple(getattr(args, "channel_mults", (1, 2)))
    if args.channel_mults == (1, 2):
        args.channel_mults = (1, 2, 4)

    args.attention_resolutions = _normalize_int_tuple(
        getattr(args, "attention_resolutions", (1,))
    )
    if not args.attention_resolutions:
        args.attention_resolutions = (1,)

    args.base_channels = max(int(getattr(args, "base_channels", 4)), 32)
    args.num_res_blocks = max(int(getattr(args, "num_res_blocks", 1)), 3)
    args.time_emb_dim = max(int(getattr(args, "time_emb_dim", 8)), 128)
    args.num_groups = min(8, args.base_channels)
    while args.base_channels % args.num_groups != 0:
        args.num_groups -= 1

    raw_dropout = float(getattr(args, "dropout", 0.1))
    if abs(raw_dropout - 0.1) < 1e-8:
        raw_dropout = 0.0
    args.dropout = min(max(raw_dropout, 0.0), 0.05)
    args.learning_rate = min(float(getattr(args, "learning_rate", 2e-4)), 1e-4)
    if getattr(args, "schedule", "linear") == "linear":
        args.schedule = "cosine"

    if getattr(args, "run_name", "").startswith("tiny_mnist_"):
        args.run_name = args.run_name.replace("tiny_mnist_", "mnist10_small_")

    return args


def print_effective_config(args):
    keys = [
        "img_size",
        "img_channels",
        "num_classes",
        "batch_size",
        "num_workers",
        "print_rate",
        "learning_rate",
        "iterations",
        "num_timesteps",
        "base_channels",
        "channel_mults",
        "num_res_blocks",
        "time_emb_dim",
        "num_groups",
        "dropout",
        "attention_resolutions",
        "log_dir",
        "run_name",
    ]
    print(f"===> source_root : {SOURCE_ROOT}")
    print(f"===> CUDA_VISIBLE_DEVICES : {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    for key in keys:
        print(f"===> {key} : {getattr(args, key)}")
    print(f"===> jax.default_backend() : {jax.default_backend()}")
    print(f"===> jax.devices() : {jax.devices()}")


def numpy_collate(batch):
    images, labels = zip(*batch)
    return (
        np.stack([np.asarray(image, dtype=np.float32) for image in images]),
        np.asarray(labels, dtype=np.int32),
    )


class NumpyCachedDataset(Dataset):
    def __init__(self, dataset, name="dataset"):
        self.name = name
        images = []
        labels = []
        for index in range(len(dataset)):
            image, label = dataset[index]
            images.append(np.asarray(image, dtype=np.float32))
            labels.append(int(label))
        self.images = np.stack(images, axis=0)
        self.labels = np.asarray(labels, dtype=np.int32)
        print(f"[INFO] cached {len(self.labels)} samples for {self.name}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def prepare_batch(batch):
    x, y = batch
    x = jnp.asarray(x, dtype=jnp.float32).transpose(0, 2, 3, 1)
    y = jnp.asarray(y, dtype=jnp.int32)
    return x, y


class DiffusionTrainState(train_state.TrainState):
    ema_params: FrozenDict


@jax.jit
def train_step(state, batch, rng, ema_decay):
    x, y = prepare_batch(batch)

    def loss_fn(params):
        return state.apply_fn({"params": params}, rng, x, y)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    ema_params = jax.tree_util.tree_map(
        lambda ema, param: ema * ema_decay + param * (1.0 - ema_decay),
        state.ema_params,
        state.params,
    )
    state = state.replace(ema_params=ema_params)
    return loss, state


@jax.jit
def eval_step(state, batch, rng):
    x, y = prepare_batch(batch)
    return state.apply_fn({"params": state.params}, rng, x, y)


def save_checkpoint(state, args, iteration, suffix="model"):
    filename = os.path.join(
        args.log_dir,
        f"{args.project_name}-{args.run_name}-iteration-{iteration}-{suffix}.msgpack",
    )
    state_dict = flax.serialization.to_state_dict(state)
    with open(filename, "wb") as handle:
        pickle.dump(state_dict, handle)


def main(args):
    args = apply_mnist10_small_defaults(args)
    print_effective_config(args)
    file_utils.mkdir(args.log_dir)

    try:
        batch_size = args.batch_size

        target_labels = list(range(10))
        raw_train_dataset = MnistDataset(is_train=True, target_labels=target_labels)
        raw_test_dataset = MnistDataset(is_train=False, target_labels=target_labels)
        train_dataset = NumpyCachedDataset(raw_train_dataset, name="train")
        test_dataset = NumpyCachedDataset(raw_test_dataset, name="test")

        train_dataloader_kwargs = dict(
            batch_size=batch_size,
            collate_fn=numpy_collate,
            num_workers=max(0, int(getattr(args, "num_workers", 0))),
        )
        if train_dataloader_kwargs["num_workers"] > 0:
            train_dataloader_kwargs["persistent_workers"] = True
            train_dataloader_kwargs["prefetch_factor"] = 2
            train_dataloader_kwargs["multiprocessing_context"] = mp.get_context("spawn")

        train_loader = script_utils.cycle(
            DataLoader(
                train_dataset,
                shuffle=True,
                drop_last=True,
                **train_dataloader_kwargs,
            )
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=numpy_collate,
            shuffle=False,
            num_workers=0,
        )
        pending_batch = next(train_loader)
        print("[INFO] train DataLoader workers are ready")

        rng = jax.random.PRNGKey(0)
        rng, x_rng = jax.random.split(rng)

        diffusion = get_diffusion_from_args(args)

        x = jax.random.normal(x_rng, (batch_size, *args.img_size, args.img_channels))
        y = jnp.arange(batch_size, dtype=jnp.int32) % args.num_classes
        rng, state_rng = jax.random.split(rng)
        variables = diffusion.init(rngs=state_rng, rng=rng, x=x, y=y)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=args.learning_rate, weight_decay=1e-4),
        )

        state = DiffusionTrainState.create(
            apply_fn=diffusion.apply,
            params=variables["params"],
            ema_params=variables["params"],
            tx=optimizer,
        )

        print("Run start.")
        best_test_loss = float("inf")
        print(
            "[INFO] The first few iterations can be slow because JAX/XLA is compiling "
            "and autotuning kernels. Steady-state speed is usually reached after warmup."
        )

        for iteration in range(1, args.iterations + 1):
            iter_start = time.time()
            if iteration == 1:
                batch = pending_batch
            else:
                batch = next(train_loader)
            rng, batch_rng = jax.random.split(rng)
            loss, state = train_step(state, batch, batch_rng, args.ema_decay)
            iter_time = time.time() - iter_start
            if iteration <= 5 or iteration % max(1, int(getattr(args, "print_rate", 50))) == 0:
                print(
                    f"=====> iter: {iteration}, loss: {float(loss):.6f}, "
                    f"iter_time: {iter_time:.3f}s"
                )

            if iteration % args.log_rate == 0:
                total_test_loss = 0.0
                for batch in test_loader:
                    rng, eval_rng = jax.random.split(rng)
                    total_test_loss += float(eval_step(state, batch, eval_rng))

                test_loss = total_test_loss / len(test_loader)
                print(f"---------> test loss: {test_loss:.6f}")

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    save_checkpoint(state, args, iteration, suffix="best")

            if iteration % args.checkpoint_rate == 0:
                save_checkpoint(state, args, iteration, suffix="model")

        print(f"Run finished. best test loss: {best_test_loss:.6f}")

    except KeyboardInterrupt:
        print("Keyboard interrupt, run finished early")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(get_args())
