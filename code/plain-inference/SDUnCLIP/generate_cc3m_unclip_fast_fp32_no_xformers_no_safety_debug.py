#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task (v2): Caption-guided image variation with StableUnCLIPImg2ImgPipeline (CC3M WebDataset tars)

Speed-optimized version:
- Batch multiple *inputs* per pipeline call (--batch_inputs)
- Generate K variations in one call (num_images_per_prompt=K)
- Prefetch thread overlaps tar IO/decode with GPU compute (--prefetch)
- Faster saving options (--save_format jpg / --png_compress_level)
"""

import json
import time
import argparse
import tarfile
import io
import threading
import queue
from pathlib import Path
from typing import List, Generator, Tuple, Iterator, Optional
import random

import torch
from PIL import Image
from tqdm import tqdm

from diffusers import StableUnCLIPImg2ImgPipeline


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
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


def derive_noise_level(noise_level: int, variation_strength: float, strength_alias, noise_scale: int) -> int:
    if noise_level is not None and noise_level >= 0:
        return int(noise_level)
    base = variation_strength
    if strength_alias is not None:
        base = float(strength_alias)
    base = max(0.0, min(1.0, float(base)))
    return int(round(base * noise_scale))


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
# WebDataset tar reading
# -----------------------------
def get_caption_from_json(json_bytes: bytes) -> str:
    try:
        data = json.loads(json_bytes.decode("utf-8"))
        return str(data.get("caption", data.get("text", ""))).strip()
    except Exception:
        return ""


def yield_samples_from_tars(tar_paths: List[Path]) -> Generator[Tuple[str, Image.Image, str], None, None]:
    """
    Iterate shards, yield (key, PIL_Image, caption)
    Expect <key>.jpg + <key>.json (or .txt)
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
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

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
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--data_root", type=str, required=True, help="See script help for this option."--limit", type=int, default=100)
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

    ap.add_argument("--prompt_template", type=str, default="{caption}", help="See script help for this option.")
    ap.add_argument("--negative_prompt", type=str, default="")

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--fresh", action="store_true")

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

    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    tar_files = sorted(list(data_root.glob("*.tar")))
    if not tar_files:
        raise FileNotFoundError(f"No .tar files found in {data_root}")
    print(f"[INFO] Found {len(tar_files)} tar files.")

    eff_noise_level = derive_noise_level(args.noise_level, args.variation_strength, args.strength, args.noise_scale)
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

    data_gen = yield_samples_from_tars(tar_files)

    # skip start_index
    if args.start_index > 0:
        print(f"[INFO] Skipping first {args.start_index} samples...")
        for _ in range(args.start_index):
            try:
                next(data_gen)
            except StopIteration:
                print("[WARN] start_index exceeded dataset size.")
                break

    # prefetch to overlap tar IO with GPU
    it = prefetch_iter(data_gen, args.prefetch)

    processed_count = 0
    with records_path.open("a", encoding="utf-8") as f, torch.inference_mode():
        pbar = tqdm(total=args.limit, desc="Generating")
        while processed_count < args.limit:
            batch: List[Tuple[int, str, Image.Image, str]] = []

            # collect a batch of valid (captioned) samples
            while len(batch) < max(1, int(args.batch_inputs)) and processed_count + len(batch) < args.limit:
                try:
                    key, pil_img, caption = next(it)
                except StopIteration:
                    break
                caption = (caption or "").strip()
                if not caption:
                    continue
                idx = processed_count + len(batch)
                batch.append((idx, key, pil_img, caption))

            if not batch:
                break

            prompts: List[str] = []
            init_images: List[Image.Image] = []
            input_paths: List[str] = []
            keys: List[str] = []
            indices: List[int] = []
            captions: List[str] = []

            for idx, key, pil_img, caption in batch:
                input_filename = f"{idx:06d}_{Path(key).name}.jpg"
                input_path = outdir / "inputs" / input_filename
                if not input_path.exists():
                    pil_img.convert("RGB").save(input_path, quality=95)

                init_image = pil_to_rgb512(pil_img, size=512)
                prompt = args.prompt_template.format(caption=caption, key=Path(key).name)

                indices.append(int(idx))
                keys.append(str(key))
                captions.append(caption)
                prompts.append(prompt)
                init_images.append(init_image)
                input_paths.append(str(input_path))

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
                key = keys[b]
                caption = captions[b]
                prompt = prompts[b]

                gen_paths = []
                for k in range(K):
                    img = imgs[b * K + k]
                    ext = "jpg" if args.save_format.lower() in ("jpg", "jpeg") else "png"
                    out_filename = f"{idx:06d}_{Path(key).name}_k{k}.{ext}"
                    out_path = outdir / "images" / out_filename
                    save_img(img, out_path, args.save_format, args.png_compress_level, args.jpg_quality)
                    gen_paths.append(str(out_path))

                rec = {
                    "task": "unclip_caption_variation_v2",
                    "dataset": "cc3m",
                    "index": int(idx),
                    "key": str(key),
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

            processed_count += B
            pbar.update(B)

        pbar.close()

    print(f"[DONE] Generated {processed_count} records.")
    print(f"[DONE] records: {records_path}")


if __name__ == "__main__":
    main()


