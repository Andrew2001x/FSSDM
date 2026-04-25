#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task (v2): Caption-guided image variation with StableUnCLIPImg2ImgPipeline

Speed-optimized version:
- Batch multiple *inputs* per pipeline call (--batch_inputs)
- Generate K variations in a single call via num_images_per_prompt=K
- Optional prefetch thread to overlap CPU image loading with GPU compute (--prefetch)
- Faster image saving options (--save_format jpg / --png_compress_level)
- Optional TF32 + channels_last + torch.compile (helps on Ampere/Hopper)

This keeps the original record format (records.jsonl) and naming convention.
"""

import json
import time
import argparse
import threading
import queue
from pathlib import Path
from typing import Iterator, Tuple, List

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
    """
    diffusers  + image ?    """
    import inspect
    sig = inspect.signature(pipe.__call__)
    allowed = set(sig.parameters.keys())

    # English note: localized comment removed.
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
    """
    - If --noise_level >= 0, use it directly.
    - Else derive from variation strength in [0,1] mapped to [0, noise_scale].
    - If user provided legacy --strength, it overrides variation_strength.
    """
    if noise_level is not None and noise_level >= 0:
        return int(noise_level)
    base = variation_strength
    if strength_alias is not None:
        base = float(strength_alias)
    base = max(0.0, min(1.0, float(base)))
    return int(round(base * noise_scale))


def prefetch_iter(it: Iterator, prefetch: int) -> Iterator:
    """
    Background prefetch to overlap CPU IO/decode with GPU compute.
    """
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
        # PNG is CPU-heavy; use low compress_level for speed
        img.save(path, compress_level=int(png_compress_level))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="See script help for this option.")
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--coco_root", type=str, required=True,
                    help="See script help for this option."--ann_file", type=str, default="",
                    help="See script help for this option.")
    ap.add_argument("--split", type=str, default="val2017", choices=["val2017", "train2017"])

    ap.add_argument("--limit", type=int, default=100, help="See script help for this option."--start_index", type=int, default=0)
    ap.add_argument("--num_images_per_input", type=int, default=2)

    # NEW: batching knobs
    ap.add_argument("--batch_inputs", type=int, default=1,
                    help="See script help for this option.")
    ap.add_argument("--prefetch", type=int, default=0,
                    help="See script help for this option."--steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=7.5)

    # Variation controls
    ap.add_argument("--noise_level", type=int, default=-1,
                    help="See script help for this option."--variation_strength", type=float, default=0.35,
                    help="See script help for this option."--noise_scale", type=int, default=50,
                    help="See script help for this option."--strength", type=float, default=None,
                    help="See script help for this option."--caption_mode", type=str, default="first", choices=["first", "random"])
    ap.add_argument("--prompt_template", type=str, default="{caption}",
                    help="See script help for this option.")
    ap.add_argument("--negative_prompt", type=str, default="",
                    help="See script help for this option.")

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--fresh", action="store_true")

    # NEW: perf toggles
    ap.add_argument("--tf32", action="store_true", help="See script help for this option.")
    ap.add_argument("--channels_last", action="store_true", help="See script help for this option.")
    ap.add_argument("--compile_unet", action="store_true", help="See script help for this option.")
    ap.add_argument("--disable_progress", action="store_true", help="See script help for this option.")

    # NEW: faster saving
    ap.add_argument("--save_format", type=str, default="png", choices=["png", "jpg", "jpeg"])
    ap.add_argument("--png_compress_level", type=int, default=1)
    ap.add_argument("--jpg_quality", type=int, default=95)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "images").mkdir(parents=True, exist_ok=True)
    (outdir / "inputs").mkdir(parents=True, exist_ok=True)
    maybe_fresh_dir(outdir, args.fresh)
    set_seed(args.seed)

    # dtype
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # perf toggles
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.backends.cudnn.benchmark = True

    coco_root = Path(args.coco_root)
    images_root = coco_root / args.split
    ann_file = Path(args.ann_file) if args.ann_file else (coco_root / "annotations" / f"captions_{args.split}.json")

    if not images_root.exists():
        raise FileNotFoundError(f"images_root not found: {images_root}")
    if not ann_file.exists():
        raise FileNotFoundError(f"ann_file not found: {ann_file}")

    try:
        from torchvision.datasets import CocoCaptions
    except Exception as e:
        raise RuntimeError("torchvision.datasets.CocoCaptions  torchvision ｇ‘?) from e

    try:
        ds = CocoCaptions(root=str(images_root), annFile=str(ann_file))
    except Exception as e:
        raise RuntimeError(" CocoCaptions €€?pycocotoolsip install pycocotools€?) from e

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

    start = max(0, args.start_index)
    end = min(len(ds), start + args.limit)
    records_path = outdir / "records.jsonl"

    eff_noise_level = derive_noise_level(args.noise_level, args.variation_strength, args.strength, args.noise_scale)
    print(f"[INFO] Using split={args.split}, subset [{start}:{end}) => n={end-start}")
    print(f"[INFO] Variation: noise_level={eff_noise_level} (noise_scale={args.noise_scale})")
    print(f"[INFO] batch_inputs={args.batch_inputs}, K={args.num_images_per_input}, save_format={args.save_format}")
    print(f"[INFO] Writing records to: {records_path}")

    printed_debug = False

    def sample_iter() -> Iterator[Tuple[int, int, Image.Image, str]]:
        for i in range(start, end):
            pil_img, captions = ds[i]
            image_id = int(ds.ids[i]) if hasattr(ds, "ids") else int(i)

            caption = ""
            if captions:
                if args.caption_mode == "random":
                    j = (args.seed + image_id) % len(captions)
                    caption = str(captions[j])
                else:
                    caption = str(captions[0])
            yield i, image_id, pil_img, caption

    it = prefetch_iter(sample_iter(), args.prefetch)

    with records_path.open("a", encoding="utf-8") as f, torch.inference_mode():
        pbar = tqdm(total=(end - start), desc="Generating")
        while True:
            batch: List[Tuple[int, int, Image.Image, str]] = []
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
            image_ids: List[int] = []
            indices: List[int] = []
            captions: List[str] = []

            # CPU work (load/resize) before GPU call
            for i, image_id, pil_img, caption in batch:
                input_path = outdir / "inputs" / f"{i:05d}_id{image_id}.jpg"
                if not input_path.exists():
                    pil_img.convert("RGB").save(input_path, quality=95)

                init_image = pil_to_rgb512(pil_img, size=512)
                prompt = args.prompt_template.format(caption=caption, image_id=image_id)

                indices.append(int(i))
                image_ids.append(int(image_id))
                captions.append(caption)
                prompts.append(prompt)
                init_images.append(init_image)
                input_paths.append(str(input_path))

            # generators for determinism: length = B*K
            gens: List[torch.Generator] = []
            for b, image_id in enumerate(image_ids):
                for k in range(args.num_images_per_input):
                    g = torch.Generator(device=args.device)
                    g.manual_seed(args.seed + image_id * 100000 + k)
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

            # save + write records
            per_item_dt = dt / max(1, B)
            for b in range(B):
                i = indices[b]
                image_id = image_ids[b]
                caption = captions[b]
                prompt = prompts[b]

                gen_paths = []
                for k in range(K):
                    img = imgs[b * K + k]
                    ext = "jpg" if args.save_format.lower() in ("jpg", "jpeg") else "png"
                    out_path = outdir / "images" / f"{i:05d}_id{image_id}_k{k}.{ext}"
                    save_img(img, out_path, args.save_format, args.png_compress_level, args.jpg_quality)
                    gen_paths.append(str(out_path))

                rec = {
                    "task": "unclip_caption_variation_v2",
                    "dataset": "coco",
                    "split": args.split,
                    "index": int(i),
                    "image_id": int(image_id),
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


