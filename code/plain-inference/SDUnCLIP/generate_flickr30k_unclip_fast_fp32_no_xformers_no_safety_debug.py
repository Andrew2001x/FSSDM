#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task (v2): Caption-guided image variation with StableUnCLIPImg2ImgPipeline (Flickr30k)

Speed-optimized version:
- Batch multiple *inputs* per pipeline call (--batch_inputs)
- Generate K variations in one call (num_images_per_prompt=K)
- Optional prefetch thread for image loading (--prefetch)
- Faster saving options (--save_format jpg / --png_compress_level)
"""

import json
import time
import csv
import ast
import argparse
import threading
import queue
from pathlib import Path
from typing import Dict, List, Iterator, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from diffusers import StableUnCLIPImg2ImgPipeline


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pil_to_rgb512(pil_img: Image.Image, size=512) -> Image.Image:
    return pil_img.convert("RGB").resize((size, size), resample=Image.BICUBIC)


def adapt_and_call_pipe(pipe, **kwargs):
    import inspect
    sig = inspect.signature(pipe.__call__)
    allowed = set(sig.parameters.keys())

    if "image" in kwargs and "image" not in allowed:
        for alt in ("init_image", "input_image", "images"):
            if alt in allowed:
                kwargs[alt] = kwargs.pop("image")
                break

    filtered = {k: v for k, v in kwargs.items() if k in allowed}
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


def prefetch_iter(it: Iterator, prefetch: int) -> Iterator:
    if prefetch <= 0:
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
        img.convert("RGB").save(path, quality=jpg_quality, optimize=False)
    else:
        img.save(path, compress_level=int(png_compress_level))


# -----------------------------
# Captions loaders
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


def load_pairs_from_csv(csv_path: Path, split: str = None) -> Dict[str, List[str]]:
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


def derive_noise_level(noise_level: int, variation_strength: float, strength_alias, noise_scale: int) -> int:
    if noise_level is not None and noise_level >= 0:
        return int(noise_level)
    base = variation_strength
    if strength_alias is not None:
        base = float(strength_alias)
    base = max(0.0, min(1.0, float(base)))
    return int(round(base * noise_scale))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--flickr_root", type=str, required=True, help="See script help for this option."--split", type=str, default="test", choices=["train", "val", "test"],
                    help="See script help for this option."--limit", type=int, default=100)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--num_images_per_input", type=int, default=2)

    # NEW: batching knobs
    ap.add_argument("--batch_inputs", type=int, default=1)
    ap.add_argument("--prefetch", type=int, default=0)

    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=7.5)

    ap.add_argument("--noise_level", type=int, default=-1, help="See script help for this option.")
    ap.add_argument("--variation_strength", type=float, default=0.35, help="0..1 -> noise_level")
    ap.add_argument("--noise_scale", type=int, default=50)
    ap.add_argument("--strength", type=float, default=None, help="See script help for this option.")

    ap.add_argument("--prompt_template", type=str, default="{caption}",
                    help="See script help for this option.")
    ap.add_argument("--negative_prompt", type=str, default="")

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--fresh", action="store_true")

    ap.add_argument("--token_file", type=str, default="", help="See script help for this option.")
    ap.add_argument("--csv_file", type=str, default="", help="See script help for this option.")
    ap.add_argument("--images_root", type=str, default="", help="See script help for this option.")

    # saving
    ap.add_argument("--save_format", type=str, default="png", choices=["png", "jpg", "jpeg"])
    ap.add_argument("--png_compress_level", type=int, default=1)
    ap.add_argument("--jpg_quality", type=int, default=95)

    # perf toggles
    ap.add_argument("--disable_progress", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--channels_last", action="store_true")
    ap.add_argument("--compile_unet", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "images").mkdir(parents=True, exist_ok=True)
    (outdir / "inputs").mkdir(parents=True, exist_ok=True)
    maybe_fresh_dir(outdir, args.fresh)
    set_seed(args.seed)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.backends.cudnn.benchmark = True

    flickr_root = Path(args.flickr_root)
    images_root = Path(args.images_root) if args.images_root else resolve_images_root(flickr_root)
    if not images_root.exists():
        raise FileNotFoundError(f"images_root not found: {images_root}")

    token_path = Path(args.token_file) if args.token_file else (flickr_root / "results_20130124.token")
    csv_path = Path(args.csv_file) if args.csv_file else (flickr_root / "flickr_annotations_30k.csv")

    if token_path.exists():
        print(f"[INFO] Using token captions: {token_path}")
        img2caps = load_pairs_from_token(token_path)
        print("[WARN] token  split ℃?-split € CSV captions?)
    else:
        print(f"[INFO] token missing, fallback to CSV captions: {csv_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV captions not found: {csv_path}")
        img2caps = load_pairs_from_csv(csv_path, split=args.split)

    image_names = sorted(img2caps.keys())
    if not image_names:
        raise RuntimeError(f"No samples found. split={args.split} ?captions ｆ┖?)

    start = max(0, args.start_index)
    end = min(len(image_names), start + args.limit) if args.limit > 0 else len(image_names)
    image_names = image_names[start:end]

    # dtype
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    eff_noise_level = derive_noise_level(args.noise_level, args.variation_strength, args.strength, args.noise_scale)
    print(f"[INFO] Using images_root: {images_root}")
    print(f"[INFO] Using split={args.split}, subset [{start}:{end}] -> n={len(image_names)}")
    print(f"[INFO] Variation: noise_level={eff_noise_level}")
    print(f"[INFO] batch_inputs={args.batch_inputs}, K={args.num_images_per_input}, save_format={args.save_format}")

    print(f"[INFO] Loading pipeline: {args.model}")
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(args.model, torch_dtype=torch_dtype).to(args.device)


    # ---- safety checker: disable to avoid black-image replacement ----
    try:
        if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
            pipe.safety_checker = None
        if hasattr(pipe, "requires_safety_checker"):
            pipe.requires_safety_checker = False
    except Exception as e:
        print(f"[WARN] Failed to disable safety_checker: {e}")
    print(f"[INFO] safety_checker disabled: {getattr(pipe, 'safety_checker', None) is None}")
    print(f"[INFO] unet dtype: {next(pipe.unet.parameters()).dtype} | vae dtype: {next(pipe.vae.parameters()).dtype}")

    # [debug] xFormers disabled

    if args.disable_progress:
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

    if args.channels_last:
        try:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.vae.to(memory_format=torch.channels_last)
        except Exception:
            pass

    if args.compile_unet:
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
            print("[INFO] torch.compile(unet) enabled.")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")

    records_path = outdir / "records.jsonl"
    print(f"[INFO] Writing records to: {records_path}")

    printed_debug = False

    def sample_iter() -> Iterator[Tuple[int, str, Path, str]]:
        for local_i, image_name in enumerate(image_names):
            caps = img2caps.get(image_name, [])
            caption = pick_caption_deterministic(caps, args.seed, start + local_i)
            if not caption:
                continue
            img_path = resolve_image_path(images_root, image_name)
            if not img_path.exists():
                continue
            yield (start + local_i), Path(image_name).name, img_path, caption

    it = prefetch_iter(sample_iter(), args.prefetch)

    with records_path.open("a", encoding="utf-8") as f, torch.inference_mode():
        pbar = tqdm(total=len(image_names), desc="Generating")
        while True:
            batch: List[Tuple[int, str, Path, str]] = []
            for _ in range(max(1, int(args.batch_inputs))):
                try:
                    batch.append(next(it))
                except StopIteration:
                    break
            if not batch:
                break

            prompts: List[str] = []
            init_images: List[Image.Image] = []
            input_paths: List[str] = []
            image_stems: List[str] = []
            indices: List[int] = []
            captions: List[str] = []

            # load images for batch
            for idx, img_name, img_path, caption in batch:
                pil_img = Image.open(img_path).convert("RGB")
                input_path = outdir / "inputs" / f"{idx:06d}_{img_name}"
                if not input_path.exists():
                    # keep original file extension for traceability
                    pil_img.save(input_path)

                init_image = pil_to_rgb512(pil_img, size=512)
                prompt = args.prompt_template.format(caption=caption, image_name=img_name)

                indices.append(int(idx))
                captions.append(caption)
                prompts.append(prompt)
                init_images.append(init_image)
                input_paths.append(str(input_path))
                image_stems.append(Path(img_name).stem)

            gens: List[torch.Generator] = []
            for b, idx in enumerate(indices):
                for k in range(int(args.num_images_per_input)):
                    g = torch.Generator(device=args.device)
                    g.manual_seed(args.seed + idx * 100000 + k)
                    gens.append(g)

            t0 = time.time()
            out = adapt_and_call_pipe(
                pipe,
                prompt=prompts,
                negative_prompt=(args.negative_prompt if args.negative_prompt.strip() else None),
                image=init_images,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                noise_level=int(eff_noise_level),
                generator=gens,
                num_images_per_prompt=int(args.num_images_per_input),
            )
            dt = time.time() - t0

            imgs = out.images
            if not printed_debug:
                printed_debug = True
                try:
                    nsfw = getattr(out, "nsfw_content_detected", None)
                    print(f"[DEBUG] nsfw_content_detected: {nsfw}")
                except Exception as e:
                    print(f"[DEBUG] nsfw_content_detected read failed: {e}")
                try:
                    ex = imgs[0].getextrema()  # RGB -> ((min,max),(min,max),(min,max))
                    print(f"[DEBUG] first_out_image extrema: {ex}")
                except Exception as e:
                    print(f"[DEBUG] extrema read failed: {e}")
            B = len(batch)
            K = int(args.num_images_per_input)
            assert len(imgs) == B * K, f"Unexpected output images: got={len(imgs)}, expected={B*K}"

            per_item_dt = dt / max(1, B)
            for b in range(B):
                idx = indices[b]
                img_stem = image_stems[b]
                caption = captions[b]
                prompt = prompts[b]

                gen_paths = []
                for k in range(K):
                    img = imgs[b * K + k]
                    ext = "jpg" if args.save_format.lower() in ("jpg", "jpeg") else "png"
                    out_path = outdir / "images" / f"{idx:06d}_{img_stem}_k{k}.{ext}"
                    save_img(img, out_path, args.save_format, args.png_compress_level, args.jpg_quality)
                    gen_paths.append(str(out_path))

                rec = {
                    "task": "unclip_caption_variation_v2",
                    "dataset": "flickr30k",
                    "split": args.split,
                    "flickr_index": int(idx),
                    "image_name": batch[b][1],
                    "caption": caption,
                    "prompt": prompt,
                    "negative_prompt": args.negative_prompt,
                    "input_path": input_paths[b],
                    "gen_paths": gen_paths,
                    "steps": int(args.steps),
                    "guidance_scale": float(args.guidance_scale),
                    "noise_level": int(eff_noise_level),
                    "variation_strength": float(args.variation_strength if args.strength is None else args.strength),
                    "noise_scale": int(args.noise_scale),
                    "seed_base": int(args.seed),
                    "time_sec": float(per_item_dt),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            pbar.update(B)
        pbar.close()

    print(f"[DONE] records: {records_path}")
    print(f"[DONE] images dir: {outdir/'images'}")
    print(f"[DONE] inputs dir: {outdir/'inputs'}")


if __name__ == "__main__":
    main()


