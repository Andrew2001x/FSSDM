#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run StableUnCLIPImg2ImgPipeline ONCE, then generate variations for:
- CC3M (WebDataset tars)
- Flickr30k
- COCO 2017

Design goals:
- Load diffusers pipeline a single time, reuse across datasets.
- Generate in logical chunks/windows (e.g. 10 samples at a time) while keeping the model loaded.
- Keep the SAME output layout as your per-dataset scripts:
    <outdir>/
      images/
      inputs/
      records.jsonl
- Keep per-dataset record schemas compatible with your existing eval_* scripts.

Notes:
- Default dtype is fp32.
- xFormers is NOT enabled here.
- Safety checker can be disabled.
"""

import argparse
import ast
import csv
import gc
import inspect
import io
import json
import queue
import random
import tarfile
import threading
import time
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from diffusers import StableUnCLIPImg2ImgPipeline


# -----------------------------
# Common utils - DETERMINISTIC VERSION
# -----------------------------
def set_seed(seed: int):
    import numpy as np
    import os
    
    # Set Python random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Force deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable fully deterministic mode (PyTorch 1.8+)
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    
    # Disable TF32 for more deterministic results
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    # Disable parallelism to ensure reproducibility
    torch.set_num_threads(1)
    
    # Set environment variable for CUDA
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Empty CUDA cache to ensure deterministic behavior
    torch.cuda.empty_cache()


def pil_to_rgb512(pil_img: Image.Image, size: int = 512) -> Image.Image:
    return pil_img.convert("RGB").resize((size, size), resample=Image.BICUBIC)


def adapt_and_call_pipe(pipe, **kwargs):
    """
    diffusers signature compatibility:
    - Pass only supported kwargs.
    - Handle `image` parameter name differences.
    """
    sig = inspect.signature(pipe.__call__)
    allowed = set(sig.parameters.keys())

    if "image" in kwargs and "image" not in allowed:
        for alt in ("init_image", "input_image", "images"):
            if alt in allowed:
                kwargs[alt] = kwargs.pop("image")
                break

    filtered = {k: v for k, v in kwargs.items() if (k in allowed and v is not None)}
    return pipe(**filtered)


def maybe_fresh_dir(outdir: Path, fresh: bool):
    if not fresh:
        return
    for sub in ("images", "inputs"):
        p = outdir / sub
        if p.exists():
            for f in p.glob("*"):
                if f.is_file():
                    f.unlink()
    rec = outdir / "records.jsonl"
    if rec.exists():
        rec.unlink()


def derive_noise_level(noise_level: int, variation_strength: float, strength_alias, noise_scale: int) -> int:
    if noise_level is not None and noise_level >= 0:
        return int(noise_level)
    base = variation_strength
    if strength_alias is not None:
        base = float(strength_alias)
    base = max(0.0, min(1.0, float(base)))
    return int(round(base * noise_scale))


# Global flag for deterministic mode
DETERMINISTIC_MODE = True


def prefetch_iter(it: Iterator, prefetch: int, deterministic: bool = False) -> Iterator:
    """
    Background prefetch to overlap CPU IO/decode with GPU compute.
    If deterministic=True, disable threading to ensure reproducibility.
    """
    # Disable prefetch in deterministic mode
    if deterministic or prefetch <= 0:
        yield from it
        return

    q: "queue.Queue[object]" = queue.Queue(maxsize=prefetch)
    sentinel = object()

    def _worker():
        try:
            for x in it:
                q.put(x)
        finally:
            q.put(sentinel)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while True:
        x = q.get()
        if x is sentinel:
            break
        yield x


def save_img(img: Image.Image, path: Path, save_format: str, png_compress_level: int, jpg_quality: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    if save_format.lower() in ("jpg", "jpeg"):
        img.convert("RGB").save(path, quality=int(jpg_quality), optimize=False)
    else:
        img.save(path, compress_level=int(png_compress_level))


def ensure_outdir(outdir: Path, fresh: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "images").mkdir(parents=True, exist_ok=True)
    (outdir / "inputs").mkdir(parents=True, exist_ok=True)
    maybe_fresh_dir(outdir, fresh)


def maybe_between_dataset_cleanup(device: str, do_cleanup: bool):
    """
    Keep model weights, but clear temporary GPU memory + python garbage.
    """
    if not do_cleanup:
        return
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def resolve_chunk_size(global_chunk_size: int, specific_chunk_size: int) -> int:
    size = int(specific_chunk_size) if int(specific_chunk_size) > 0 else int(global_chunk_size)
    if size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {size}")
    return size


# -----------------------------
# CC3M reader (tars)
# -----------------------------
def get_caption_from_json(json_bytes: bytes) -> str:
    try:
        data = json.loads(json_bytes.decode("utf-8"))
        return str(data.get("caption", data.get("text", ""))).strip()
    except Exception:
        return ""


def yield_cc3m_samples_from_tars(tar_paths: List[Path]) -> Generator[Tuple[str, Image.Image, str], None, None]:
    """
    Yield (key, PIL_Image, caption).
    Expect <key>.jpg + <key>.json (or .txt) inside tar.
    """
    for tp in tar_paths:
        if not tp.exists():
            print(f"[WARN] Tar file not found: {tp}")
            continue

        print(f"[INFO] Reading tar: {tp.name}")
        try:
            with tarfile.open(tp, "r") as tar:
                members = tar.getmembers()
                image_members = [m for m in members if m.name.lower().endswith((".jpg", ".jpeg", ".png"))]
                image_members.sort(key=lambda x: x.name)

                for img_m in image_members:
                    base_name = img_m.name.rsplit(".", 1)[0]
                    json_name = base_name + ".json"
                    txt_name = base_name + ".txt"

                    try:
                        f_img = tar.extractfile(img_m)
                        if f_img is None:
                            continue
                        img_bytes = f_img.read()
                        with Image.open(io.BytesIO(img_bytes)) as im:
                            pil_img = im.convert("RGB").copy()

                        caption = ""
                        try:
                            f_json = tar.extractfile(json_name)
                            if f_json:
                                caption = get_caption_from_json(f_json.read())
                        except KeyError:
                            pass

                        if not caption:
                            try:
                                f_txt = tar.extractfile(txt_name)
                                if f_txt:
                                    caption = f_txt.read().decode("utf-8", errors="ignore").strip()
                            except Exception:
                                pass

                        yield base_name, pil_img, caption

                    except Exception as e:
                        print(f"[WARN] Error reading {img_m.name}: {e}")
                        continue

        except Exception as e:
            print(f"[ERROR] Failed to open tar {tp}: {e}")


# -----------------------------
# Flickr loaders
# -----------------------------
def _parse_raw_captions(raw_val) -> List[str]:
    if raw_val is None:
        return []
    s = str(raw_val).strip()
    if not s:
        return []

    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass

    if "\n" in s:
        return [t.strip() for t in s.splitlines() if t.strip()]

    if "|" in s:
        return [t.strip() for t in s.split("|") if t.strip()]

    return [s]


def load_pairs_from_token(token_path: Path) -> Dict[str, List[str]]:
    img2caps: Dict[str, List[str]] = {}
    with token_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            left, cap = line.split("\t", 1)
            fn = left.split("#")[0].strip()
            cap = cap.strip()
            if fn and cap:
                img2caps.setdefault(fn, []).append(cap)
    for k in list(img2caps.keys()):
        uniq, seen = [], set()
        for c in img2caps[k]:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        img2caps[k] = uniq
    return img2caps


def load_pairs_from_csv(csv_path: Path, split: Optional[str] = None) -> Dict[str, List[str]]:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []

        img_col = None
        cap_col = None
        split_col = "split" if "split" in cols else None

        for c in ["image_name", "filename", "file_name", "image", "img"]:
            if c in cols:
                img_col = c
                break
        for c in ["caption", "comment", "raw", "text", "sentence", "sentences"]:
            if c in cols:
                cap_col = c
                break
        if img_col is None or cap_col is None:
            raise RuntimeError("Runtime error (message translated to English).")

        img2caps: Dict[str, List[str]] = {}
        for row in reader:
            if split is not None and split_col is not None:
                if str(row.get(split_col, "")).strip() != split:
                    continue
            fn = str(row.get(img_col, "")).strip()
            caps = _parse_raw_captions(row.get(cap_col, ""))
            if fn and caps:
                img2caps.setdefault(fn, []).extend(caps)

        for k in list(img2caps.keys()):
            uniq, seen = [], set()
            for c in img2caps[k]:
                c = c.strip()
                if c and c not in seen:
                    seen.add(c)
                    uniq.append(c)
            img2caps[k] = uniq
        return img2caps


def pick_caption_deterministic(caps: List[str], seed: int, idx: int) -> str:
    if not caps:
        return ""
    import numpy as np
    r = np.random.RandomState(seed + idx * 99991)
    return caps[int(r.randint(0, len(caps)))]


def resolve_images_root(flickr_root: Path) -> Path:
    cand = [
        flickr_root / "images" / "flickr30k-images",
        flickr_root / "images",
        flickr_root / "flickr30k-images",
        flickr_root / "data" / "flickr30k" / "flickr30k-images",
        flickr_root / "flickr30k" / "images",
    ]
    for c in cand:
        if c.exists() and c.is_dir():
            return c
    return flickr_root / "images" / "flickr30k-images"


def resolve_image_path(images_root: Path, filename: str) -> Path:
    p = images_root / filename
    if p.exists():
        return p
    stem = Path(filename).stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        pp = images_root / (stem + ext)
        if pp.exists():
            return pp
    return p


# -----------------------------
# Per-dataset generation
# -----------------------------
@torch.inference_mode()
def generate_cc3m(
    pipe,
    outdir: Path,
    data_root: Path,
    limit: int,
    start_index: int,
    k: int,
    batch_inputs: int,
    prefetch: int,
    steps: int,
    guidance_scale: float,
    eff_noise_level: int,
    variation_strength_value: float,
    noise_scale_value: int,
    seed: int,
    prompt_template: str,
    negative_prompt: str,
    save_format: str,
    png_compress_level: int,
    jpg_quality: int,
    chunk_size: int,
    cleanup_device: str,
    between_chunk_cleanup: bool,
):
    ensure_outdir(outdir, fresh=False)

    tar_files = sorted(list(data_root.glob("*.tar")))
    if not tar_files:
        raise FileNotFoundError(f"No .tar files found in {data_root}")
    print(f"[CC3M] Found {len(tar_files)} tar files.")

    records_path = outdir / "records.jsonl"
    print(f"[CC3M] Writing records to: {records_path}")

    data_gen = yield_cc3m_samples_from_tars(tar_files)

    if start_index > 0:
        print(f"[CC3M] Skipping first {start_index} samples...")
        for _ in range(start_index):
            try:
                next(data_gen)
            except StopIteration:
                print("[CC3M] WARN: start_index exceeded dataset size.")
                break

    # Use deterministic mode to ensure reproducibility
    it = prefetch_iter(data_gen, prefetch, deterministic=True)

    processed = 0
    chunk_id = 0

    with records_path.open("a", encoding="utf-8") as f:
        pbar = tqdm(total=limit, desc="CC3M Generating")

        while processed < limit:
            current_chunk = min(chunk_size, limit - processed)
            chunk_begin_abs = start_index + processed
            chunk_end_abs = chunk_begin_abs + current_chunk
            chunk_id += 1

            print(
                f"[CC3M] Chunk {chunk_id}: generating {current_chunk} samples "
                f"(absolute range [{chunk_begin_abs}, {chunk_end_abs}))"
            )

            chunk_processed = 0
            while chunk_processed < current_chunk:
                batch: List[Tuple[int, str, Image.Image, str]] = []

                while len(batch) < max(1, int(batch_inputs)) and (chunk_processed + len(batch) < current_chunk):
                    try:
                        key, pil_img, caption = next(it)
                    except StopIteration:
                        break

                    caption = (caption or "").strip()
                    if not caption:
                        continue

                    idx = start_index + processed + len(batch)
                    batch.append((idx, key, pil_img, caption))

                if not batch:
                    break

                prompts, init_images, input_paths, keys, indices, captions = [], [], [], [], [], []
                for idx, key, pil_img, caption in batch:
                    input_filename = f"{idx:06d}_{Path(key).name}.jpg"
                    input_path = outdir / "inputs" / input_filename
                    if not input_path.exists():
                        pil_img.convert("RGB").save(input_path, quality=95)

                    init_image = pil_to_rgb512(pil_img, size=512)
                    prompt = prompt_template.format(caption=caption, key=Path(key).name)

                    indices.append(int(idx))
                    keys.append(str(key))
                    captions.append(caption)
                    prompts.append(prompt)
                    init_images.append(init_image)
                    input_paths.append(str(input_path))

                gens: List[torch.Generator] = []
                for idx in indices:
                    for kk in range(int(k)):
                        g = torch.Generator(device=pipe._gen_device)
                        g.manual_seed(seed + idx * 100000 + kk)
                        gens.append(g)

                t0 = time.time()
                out = adapt_and_call_pipe(
                    pipe,
                    prompt=prompts,
                    negative_prompt=(negative_prompt if negative_prompt.strip() else None),
                    image=init_images,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    noise_level=int(eff_noise_level),
                    generator=gens,
                    num_images_per_prompt=int(k),
                )
                dt = time.time() - t0

                imgs = out.images
                B = len(batch)
                K = int(k)
                if len(imgs) != B * K:
                    raise RuntimeError(f"[CC3M] Unexpected output images: got={len(imgs)}, expected={B*K}")

                per_item_dt = dt / max(1, B)
                for b in range(B):
                    idx = indices[b]
                    key = keys[b]
                    caption = captions[b]
                    prompt = prompts[b]

                    gen_paths = []
                    for kk in range(K):
                        img = imgs[b * K + kk]
                        ext = "jpg" if save_format.lower() in ("jpg", "jpeg") else "png"
                        out_filename = f"{idx:06d}_{Path(key).name}_k{kk}.{ext}"
                        out_path = outdir / "images" / out_filename
                        save_img(img, out_path, save_format, png_compress_level, jpg_quality)
                        gen_paths.append(str(out_path))

                    rec = {
                        "task": "unclip_caption_variation_v2",
                        "dataset": "cc3m",
                        "index": int(idx),
                        "key": str(key),
                        "caption": caption,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "input_path": input_paths[b],
                        "gen_paths": gen_paths,
                        "steps": int(steps),
                        "guidance_scale": float(guidance_scale),
                        "noise_level": int(eff_noise_level),
                        "variation_strength": float(variation_strength_value),
                        "noise_scale": int(noise_scale_value),
                        "seed_base": int(seed),
                        "time_sec": float(per_item_dt),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                processed += B
                chunk_processed += B
                pbar.update(B)

                if processed >= limit:
                    break

            if between_chunk_cleanup and processed < limit:
                print(f"[CC3M] Chunk {chunk_id} done. Running chunk cleanup...")
                maybe_between_dataset_cleanup(cleanup_device, True)

        pbar.close()

    print(f"[CC3M DONE] Generated {processed} records at: {records_path}")


@torch.inference_mode()
def generate_flickr30k(
    pipe,
    outdir: Path,
    flickr_root: Path,
    split: str,
    limit: int,
    start_index: int,
    k: int,
    batch_inputs: int,
    prefetch: int,
    steps: int,
    guidance_scale: float,
    eff_noise_level: int,
    variation_strength_value: float,
    noise_scale_value: int,
    seed: int,
    prompt_template: str,
    negative_prompt: str,
    token_file: Optional[Path],
    csv_file: Optional[Path],
    images_root_override: Optional[Path],
    save_format: str,
    png_compress_level: int,
    jpg_quality: int,
    chunk_size: int,
    cleanup_device: str,
    between_chunk_cleanup: bool,
):
    ensure_outdir(outdir, fresh=False)

    images_root = images_root_override if images_root_override else resolve_images_root(flickr_root)
    if not images_root.exists():
        raise FileNotFoundError(f"images_root not found: {images_root}")

    token_path = token_file if token_file else (flickr_root / "results_20130124.token")
    csv_path = csv_file if csv_file else (flickr_root / "flickr_annotations_30k.csv")

    if token_path.exists():
        print(f"[Flickr] Using token captions: {token_path}")
        img2caps = load_pairs_from_token(token_path)
        print("[Flickr] WARN: token file usually has no split; --split may not take effect.")
    else:
        print(f"[Flickr] token missing, fallback to CSV: {csv_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV captions not found: {csv_path}")
        img2caps = load_pairs_from_csv(csv_path, split=split)

    image_names = sorted(img2caps.keys())
    if not image_names:
        raise RuntimeError(f"[Flickr] No samples found. split={split} may not match, or captions parsing is empty.")

    start = max(0, int(start_index))
    end = min(len(image_names), start + int(limit)) if int(limit) > 0 else len(image_names)
    image_names = image_names[start:end]

    records_path = outdir / "records.jsonl"
    print(f"[Flickr] Using images_root: {images_root}")
    print(f"[Flickr] Writing records to: {records_path}")

    def sample_iter() -> Iterator[Tuple[int, str, Path, str]]:
        for local_i, image_name in enumerate(image_names):
            caps = img2caps.get(image_name, [])
            caption = pick_caption_deterministic(caps, seed, start + local_i)
            if not caption:
                continue
            img_path = resolve_image_path(images_root, image_name)
            if not img_path.exists():
                continue
            yield (start + local_i), Path(image_name).name, img_path, caption

    # Use deterministic mode to ensure reproducibility
    it = prefetch_iter(sample_iter(), prefetch, deterministic=True)

    processed = 0
    total = len(image_names)
    chunk_id = 0

    with records_path.open("a", encoding="utf-8") as f:
        pbar = tqdm(total=total, desc="Flickr Generating")

        while processed < total:
            current_chunk = min(chunk_size, total - processed)
            chunk_begin_abs = start + processed
            chunk_end_abs = chunk_begin_abs + current_chunk
            chunk_id += 1

            print(
                f"[Flickr] Chunk {chunk_id}: generating {current_chunk} samples "
                f"(absolute range [{chunk_begin_abs}, {chunk_end_abs}))"
            )

            chunk_processed = 0
            while chunk_processed < current_chunk:
                batch: List[Tuple[int, str, Path, str]] = []
                for _ in range(min(max(1, int(batch_inputs)), current_chunk - chunk_processed)):
                    try:
                        batch.append(next(it))
                    except StopIteration:
                        break

                if not batch:
                    break

                prompts, init_images, input_paths, image_stems, indices, captions = [], [], [], [], [], []
                for idx, img_name, img_path, caption in batch:
                    with Image.open(img_path) as im:
                        pil_img = im.convert("RGB").copy()

                    input_path = outdir / "inputs" / f"{idx:06d}_{img_name}"
                    if not input_path.exists():
                        pil_img.save(input_path)

                    init_image = pil_to_rgb512(pil_img, size=512)
                    prompt = prompt_template.format(caption=caption, image_name=img_name)

                    indices.append(int(idx))
                    captions.append(caption)
                    prompts.append(prompt)
                    init_images.append(init_image)
                    input_paths.append(str(input_path))
                    image_stems.append(Path(img_name).stem)

                gens: List[torch.Generator] = []
                for idx in indices:
                    for kk in range(int(k)):
                        g = torch.Generator(device=pipe._gen_device)
                        g.manual_seed(seed + idx * 100000 + kk)
                        gens.append(g)

                t0 = time.time()
                out = adapt_and_call_pipe(
                    pipe,
                    prompt=prompts,
                    negative_prompt=(negative_prompt if negative_prompt.strip() else None),
                    image=init_images,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    noise_level=int(eff_noise_level),
                    generator=gens,
                    num_images_per_prompt=int(k),
                )
                dt = time.time() - t0

                imgs = out.images
                B = len(batch)
                K = int(k)
                if len(imgs) != B * K:
                    raise RuntimeError(f"[Flickr] Unexpected output images: got={len(imgs)}, expected={B*K}")

                per_item_dt = dt / max(1, B)
                for b in range(B):
                    idx = indices[b]
                    img_stem = image_stems[b]
                    caption = captions[b]
                    prompt = prompts[b]

                    gen_paths = []
                    for kk in range(K):
                        img = imgs[b * K + kk]
                        ext = "jpg" if save_format.lower() in ("jpg", "jpeg") else "png"
                        out_path = outdir / "images" / f"{idx:06d}_{img_stem}_k{kk}.{ext}"
                        save_img(img, out_path, save_format, png_compress_level, jpg_quality)
                        gen_paths.append(str(out_path))

                    rec = {
                        "task": "unclip_caption_variation_v2",
                        "dataset": "flickr30k",
                        "split": split,
                        "flickr_index": int(idx),
                        "image_name": batch[b][1],
                        "caption": caption,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "input_path": input_paths[b],
                        "gen_paths": gen_paths,
                        "steps": int(steps),
                        "guidance_scale": float(guidance_scale),
                        "noise_level": int(eff_noise_level),
                        "variation_strength": float(variation_strength_value),
                        "noise_scale": int(noise_scale_value),
                        "seed_base": int(seed),
                        "time_sec": float(per_item_dt),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                processed += B
                chunk_processed += B
                pbar.update(B)

                if processed >= total:
                    break

            if between_chunk_cleanup and processed < total:
                print(f"[Flickr] Chunk {chunk_id} done. Running chunk cleanup...")
                maybe_between_dataset_cleanup(cleanup_device, True)

        pbar.close()

    print(f"[Flickr DONE] records: {records_path}")


@torch.inference_mode()
def generate_coco(
    pipe,
    outdir: Path,
    coco_root: Path,
    split: str,
    ann_file: Optional[Path],
    limit: int,
    start_index: int,
    k: int,
    batch_inputs: int,
    prefetch: int,
    steps: int,
    guidance_scale: float,
    eff_noise_level: int,
    variation_strength_value: float,
    noise_scale_value: int,
    seed: int,
    caption_mode: str,
    prompt_template: str,
    negative_prompt: str,
    save_format: str,
    png_compress_level: int,
    jpg_quality: int,
    chunk_size: int,
    cleanup_device: str,
    between_chunk_cleanup: bool,
):
    ensure_outdir(outdir, fresh=False)

    images_root = coco_root / split
    ann_path = ann_file if ann_file else (coco_root / "annotations" / f"captions_{split}.json")
    if not images_root.exists():
        raise FileNotFoundError(f"images_root not found: {images_root}")
    if not ann_path.exists():
        raise FileNotFoundError(f"ann_file not found: {ann_path}")

    try:
        from torchvision.datasets import CocoCaptions
    except Exception as e:
        raise RuntimeError("torchvision.datasets.CocoCaptions import failed. Check torchvision installation.") from e

    try:
        ds = CocoCaptions(root=str(images_root), annFile=str(ann_path))
    except Exception as e:
        raise RuntimeError("Building CocoCaptions failed (often missing pycocotools).") from e

    start = max(0, int(start_index))
    end = min(len(ds), start + int(limit))
    total = end - start
    records_path = outdir / "records.jsonl"

    print(f"[COCO] Using split={split}, subset [{start}:{end}) => n={total}")
    print(f"[COCO] Writing records to: {records_path}")

    def sample_iter() -> Iterator[Tuple[int, int, Image.Image, str]]:
        for i in range(start, end):
            pil_img, captions = ds[i]
            image_id = int(ds.ids[i]) if hasattr(ds, "ids") else int(i)

            caption = ""
            if captions:
                if caption_mode == "random":
                    j = (seed + image_id) % len(captions)
                    caption = str(captions[j])
                else:
                    caption = str(captions[0])
            yield i, image_id, pil_img, caption

    # Use deterministic mode to ensure reproducibility
    it = prefetch_iter(sample_iter(), prefetch, deterministic=True)

    processed = 0
    chunk_id = 0

    with records_path.open("a", encoding="utf-8") as f:
        pbar = tqdm(total=total, desc="COCO Generating")

        while processed < total:
            current_chunk = min(chunk_size, total - processed)
            chunk_begin_abs = start + processed
            chunk_end_abs = chunk_begin_abs + current_chunk
            chunk_id += 1

            print(
                f"[COCO] Chunk {chunk_id}: generating {current_chunk} samples "
                f"(absolute range [{chunk_begin_abs}, {chunk_end_abs}))"
            )

            chunk_processed = 0
            while chunk_processed < current_chunk:
                batch: List[Tuple[int, int, Image.Image, str]] = []
                for _ in range(min(max(1, int(batch_inputs)), current_chunk - chunk_processed)):
                    try:
                        batch.append(next(it))
                    except StopIteration:
                        break

                if not batch:
                    break

                prompts, init_images, input_paths, image_ids, indices, captions = [], [], [], [], [], []
                for i, image_id, pil_img, caption in batch:
                    input_path = outdir / "inputs" / f"{i:05d}_id{image_id}.jpg"
                    if not input_path.exists():
                        pil_img.convert("RGB").save(input_path, quality=95)

                    init_image = pil_to_rgb512(pil_img, size=512)
                    prompt = prompt_template.format(caption=caption, image_id=image_id)

                    indices.append(int(i))
                    image_ids.append(int(image_id))
                    captions.append(caption)
                    prompts.append(prompt)
                    init_images.append(init_image)
                    input_paths.append(str(input_path))

                gens: List[torch.Generator] = []
                for image_id in image_ids:
                    for kk in range(int(k)):
                        g = torch.Generator(device=pipe._gen_device)
                        g.manual_seed(seed + image_id * 100000 + kk)
                        gens.append(g)

                t0 = time.time()
                out = adapt_and_call_pipe(
                    pipe,
                    prompt=prompts,
                    negative_prompt=(negative_prompt if negative_prompt.strip() else None),
                    image=init_images,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    noise_level=int(eff_noise_level),
                    generator=gens,
                    num_images_per_prompt=int(k),
                )
                dt = time.time() - t0

                imgs = out.images
                B = len(batch)
                K = int(k)
                if len(imgs) != B * K:
                    raise RuntimeError(f"[COCO] Unexpected output images: got={len(imgs)}, expected={B*K}")

                per_item_dt = dt / max(1, B)
                for b in range(B):
                    i = indices[b]
                    image_id = image_ids[b]
                    caption = captions[b]
                    prompt = prompts[b]

                    gen_paths = []
                    for kk in range(K):
                        img = imgs[b * K + kk]
                        ext = "jpg" if save_format.lower() in ("jpg", "jpeg") else "png"
                        out_path = outdir / "images" / f"{i:05d}_id{image_id}_k{kk}.{ext}"
                        save_img(img, out_path, save_format, png_compress_level, jpg_quality)
                        gen_paths.append(str(out_path))

                    rec = {
                        "task": "unclip_caption_variation_v2",
                        "dataset": "coco",
                        "split": split,
                        "index": int(i),
                        "image_id": int(image_id),
                        "caption": caption,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "input_path": input_paths[b],
                        "gen_paths": gen_paths,
                        "steps": int(steps),
                        "guidance_scale": float(guidance_scale),
                        "noise_level": int(eff_noise_level),
                        "variation_strength": float(variation_strength_value),
                        "noise_scale": int(noise_scale_value),
                        "seed_base": int(seed),
                        "time_sec": float(per_item_dt),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                processed += B
                chunk_processed += B
                pbar.update(B)

                if processed >= total:
                    break

            if between_chunk_cleanup and processed < total:
                print(f"[COCO] Chunk {chunk_id} done. Running chunk cleanup...")
                maybe_between_dataset_cleanup(cleanup_device, True)

        pbar.close()

    print(f"[COCO DONE] records: {records_path}")


def load_pipeline_once(
    model: str,
    device: str,
    dtype: str,
    disable_safety_checker: bool,
    disable_progress: bool,
    tf32: bool,
    channels_last: bool,
    compile_unet: bool,
):
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.backends.cudnn.benchmark = True

    print(f"[PIPE] Loading pipeline once: {model} dtype={torch_dtype} device={device}")
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(model, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    if disable_progress:
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

    if disable_safety_checker:
        try:
            pipe.safety_checker = None
        except Exception:
            pass
        if hasattr(pipe, "requires_safety_checker"):
            try:
                pipe.requires_safety_checker = False
            except Exception:
                pass

    if channels_last:
        try:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.vae.to(memory_format=torch.channels_last)
        except Exception:
            pass

    if compile_unet:
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
            print("[PIPE] torch.compile(unet) enabled.")
        except Exception as e:
            print(f"[PIPE] WARN: torch.compile failed: {e}")

    pipe._gen_device = torch.device(device)

    try:
        unet_dtype = next(pipe.unet.parameters()).dtype
        vae_dtype = next(pipe.vae.parameters()).dtype
        print(f"[PIPE] unet dtype={unet_dtype}, vae dtype={vae_dtype}")
    except Exception:
        pass

    return pipe


def main():
    ap = argparse.ArgumentParser()

    # global
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance_scale", type=float, default=7.0)

    ap.add_argument("--noise_level", type=int, default=-1)
    ap.add_argument("--variation_strength", type=float, default=0.35)
    ap.add_argument("--noise_scale", type=int, default=50)
    ap.add_argument("--strength", type=float, default=None)

    ap.add_argument("--num_images_per_input", type=int, default=3)
    ap.add_argument("--cc3m_k", type=int, default=-1)
    ap.add_argument("--flickr_k", type=int, default=-1)
    ap.add_argument("--coco_k", type=int, default=-1)

    ap.add_argument("--batch_inputs", type=int, default=1)
    ap.add_argument("--prefetch", type=int, default=32)

    # logical chunk/window
    ap.add_argument("--chunk_size", type=int, default=10,
                    help="Logical generation window size. Model stays loaded; generation runs chunk by chunk.")
    ap.add_argument("--cc3m_chunk_size", type=int, default=-1)
    ap.add_argument("--flickr_chunk_size", type=int, default=-1)
    ap.add_argument("--coco_chunk_size", type=int, default=-1)
    ap.add_argument("--between_chunk_cleanup", action="store_true",
                    help="Run gc + empty_cache after each chunk/window while keeping the model loaded.")

    ap.add_argument("--prompt_template", type=str, default="{caption}")
    ap.add_argument("--negative_prompt", type=str, default="")

    ap.add_argument("--save_format", type=str, default="jpg", choices=["png", "jpg", "jpeg"])
    ap.add_argument("--png_compress_level", type=int, default=1)
    ap.add_argument("--jpg_quality", type=int, default=95)

    # perf toggles
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--channels_last", action="store_true")
    ap.add_argument("--compile_unet", action="store_true")
    ap.add_argument("--disable_progress", action="store_true")
    ap.add_argument("--disable_safety_checker", action="store_true", default=True,
                    help="Disable safety checker (recommended to avoid black image replacement).")
    ap.add_argument("--between_dataset_cleanup", action="store_true",
                    help="Run gc + empty_cache between datasets (keeps weights).")

    # datasets roots
    ap.add_argument("--cc3m_root", type=str, required=True)
    ap.add_argument("--flickr_root", type=str, required=True)
    ap.add_argument("--coco_root", type=str, required=True)

    # dataset outdirs
    ap.add_argument("--cc3m_outdir", type=str, required=True)
    ap.add_argument("--flickr_outdir", type=str, required=True)
    ap.add_argument("--coco_outdir", type=str, required=True)

    # limits
    ap.add_argument("--cc3m_limit", type=int, default=1000)
    ap.add_argument("--flickr_limit", type=int, default=1000)
    ap.add_argument("--coco_limit", type=int, default=1000)

    ap.add_argument("--cc3m_start_index", type=int, default=0)
    ap.add_argument("--flickr_start_index", type=int, default=0)
    ap.add_argument("--coco_start_index", type=int, default=0)

    # splits/ann
    ap.add_argument("--flickr_split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--coco_split", type=str, default="val2017", choices=["val2017", "train2017"])
    ap.add_argument("--coco_ann_file", type=str, default="")
    ap.add_argument("--coco_caption_mode", type=str, default="first", choices=["first", "random"])

    # flickr optional paths
    ap.add_argument("--flickr_token_file", type=str, default="")
    ap.add_argument("--flickr_csv_file", type=str, default="")
    ap.add_argument("--flickr_images_root", type=str, default="")

    # fresh per dataset
    ap.add_argument("--fresh", action="store_true")

    # order
    ap.add_argument("--order", type=str, default="cc3m,flickr,coco",
                    help="Comma separated order: cc3m,flickr,coco (any permutation).")

    args = ap.parse_args()

    set_seed(args.seed)

    eff_noise_level = derive_noise_level(args.noise_level, args.variation_strength, args.strength, args.noise_scale)
    print(f"[RUN] noise_level={eff_noise_level} (derived), steps={args.steps}, guidance={args.guidance_scale}")
    print(f"[RUN] batch_inputs={args.batch_inputs}, K={args.num_images_per_input}, prefetch={args.prefetch}")

    cc3m_k = int(args.cc3m_k) if int(args.cc3m_k) > 0 else int(args.num_images_per_input)
    flickr_k = int(args.flickr_k) if int(args.flickr_k) > 0 else int(args.num_images_per_input)
    coco_k = int(args.coco_k) if int(args.coco_k) > 0 else int(args.num_images_per_input)

    cc3m_chunk_size = resolve_chunk_size(args.chunk_size, args.cc3m_chunk_size)
    flickr_chunk_size = resolve_chunk_size(args.chunk_size, args.flickr_chunk_size)
    coco_chunk_size = resolve_chunk_size(args.chunk_size, args.coco_chunk_size)

    print(f"[RUN] per-dataset K: cc3m={cc3m_k}, flickr={flickr_k}, coco={coco_k}")
    print(f"[RUN] per-dataset chunk_size: cc3m={cc3m_chunk_size}, flickr={flickr_chunk_size}, coco={coco_chunk_size}")
    print(f"[RUN] between_chunk_cleanup={args.between_chunk_cleanup}")

    cc3m_out = Path(args.cc3m_outdir)
    flickr_out = Path(args.flickr_outdir)
    coco_out = Path(args.coco_outdir)
    for od in (cc3m_out, flickr_out, coco_out):
        ensure_outdir(od, fresh=args.fresh)

    pipe = load_pipeline_once(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        disable_safety_checker=args.disable_safety_checker,
        disable_progress=args.disable_progress,
        tf32=args.tf32,
        channels_last=args.channels_last,
        compile_unet=args.compile_unet,
    )

    order = [x.strip().lower() for x in args.order.split(",") if x.strip()]
    valid = {"cc3m", "flickr", "coco"}
    if not order or any(x not in valid for x in order):
        raise ValueError(f"--order must be a comma-separated list of {sorted(valid)}; got: {args.order}")

    for name in order:
        if name == "cc3m":
            print("[RUN] ===== CC3M =====")
            generate_cc3m(
                pipe=pipe,
                outdir=cc3m_out,
                data_root=Path(args.cc3m_root),
                limit=int(args.cc3m_limit),
                start_index=int(args.cc3m_start_index),
                k=int(cc3m_k),
                batch_inputs=int(args.batch_inputs),
                prefetch=int(args.prefetch),
                steps=int(args.steps),
                guidance_scale=float(args.guidance_scale),
                eff_noise_level=int(eff_noise_level),
                variation_strength_value=float(args.variation_strength if args.strength is None else args.strength),
                noise_scale_value=int(args.noise_scale),
                seed=int(args.seed),
                prompt_template=str(args.prompt_template),
                negative_prompt=str(args.negative_prompt),
                save_format=str(args.save_format),
                png_compress_level=int(args.png_compress_level),
                jpg_quality=int(args.jpg_quality),
                chunk_size=int(cc3m_chunk_size),
                cleanup_device=str(args.device),
                between_chunk_cleanup=bool(args.between_chunk_cleanup),
            )

        elif name == "flickr":
            print("[RUN] ===== Flickr30k =====")
            generate_flickr30k(
                pipe=pipe,
                outdir=flickr_out,
                flickr_root=Path(args.flickr_root),
                split=str(args.flickr_split),
                limit=int(args.flickr_limit),
                start_index=int(args.flickr_start_index),
                k=int(flickr_k),
                batch_inputs=int(args.batch_inputs),
                prefetch=int(args.prefetch),
                steps=int(args.steps),
                guidance_scale=float(args.guidance_scale),
                eff_noise_level=int(eff_noise_level),
                variation_strength_value=float(args.variation_strength if args.strength is None else args.strength),
                noise_scale_value=int(args.noise_scale),
                seed=int(args.seed),
                prompt_template=str(args.prompt_template),
                negative_prompt=str(args.negative_prompt),
                token_file=(Path(args.flickr_token_file) if args.flickr_token_file else None),
                csv_file=(Path(args.flickr_csv_file) if args.flickr_csv_file else None),
                images_root_override=(Path(args.flickr_images_root) if args.flickr_images_root else None),
                save_format=str(args.save_format),
                png_compress_level=int(args.png_compress_level),
                jpg_quality=int(args.jpg_quality),
                chunk_size=int(flickr_chunk_size),
                cleanup_device=str(args.device),
                between_chunk_cleanup=bool(args.between_chunk_cleanup),
            )

        elif name == "coco":
            print("[RUN] ===== COCO =====")
            generate_coco(
                pipe=pipe,
                outdir=coco_out,
                coco_root=Path(args.coco_root),
                split=str(args.coco_split),
                ann_file=(Path(args.coco_ann_file) if args.coco_ann_file else None),
                limit=int(args.coco_limit),
                start_index=int(args.coco_start_index),
                k=int(coco_k),
                batch_inputs=int(args.batch_inputs),
                prefetch=int(args.prefetch),
                steps=int(args.steps),
                guidance_scale=float(args.guidance_scale),
                eff_noise_level=int(eff_noise_level),
                variation_strength_value=float(args.variation_strength if args.strength is None else args.strength),
                noise_scale_value=int(args.noise_scale),
                seed=int(args.seed),
                caption_mode=str(args.coco_caption_mode),
                prompt_template=str(args.prompt_template),
                negative_prompt=str(args.negative_prompt),
                save_format=str(args.save_format),
                png_compress_level=int(args.png_compress_level),
                jpg_quality=int(args.jpg_quality),
                chunk_size=int(coco_chunk_size),
                cleanup_device=str(args.device),
                between_chunk_cleanup=bool(args.between_chunk_cleanup),
            )

        maybe_between_dataset_cleanup(args.device, args.between_dataset_cleanup)

    print("[RUN] ALL DONE")


if __name__ == "__main__":
    main()

