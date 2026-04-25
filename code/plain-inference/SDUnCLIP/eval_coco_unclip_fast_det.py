#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation (v2) for StableUnCLIP caption-guided image variation - DETERMINISTIC VERSION.

This evaluation matches the *intended* behavior of StableUnCLIPImg2Img:
it generates new images conditioned on an input image embedding + prompt.
So pixel-level reconstruction metrics (MSE/PSNR/SSIM) are optional;
the primary metrics are semantic:

- clip_it: CLIP cosine similarity between generated image and caption (higher=better)
- clip_ii: CLIP cosine similarity between generated image and input image (higher=closer to input)
- div_clip: diversity among K samples based on CLIP image embeddings
            (1 - mean_pairwise_cosine_similarity; higher=more diverse)
- fid: distribution-level FID between generated images and input images (lower=better)

Aggregation options (when K>1):
- mean
- best_clip_it
- best_clip_ii
- best_tradeoff  (maximize clip_it + alpha * clip_ii)
- best_ssim / best_mse (if you also compute pixel metrics)
"""

import re
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# -----------------------------
# Path utils
# -----------------------------
def resolve_existing_path(p: str, run_dir: Path) -> str:
    pp = Path(str(p)).expanduser()
    if pp.is_absolute() and pp.exists():
        return str(pp)

    if pp.exists():  # relative to cwd
        return str(pp)

    pp2 = (run_dir / pp).resolve()
    if pp2.exists():
        return str(pp2)

    # try last 2 path parts
    parts = pp.parts
    if len(parts) >= 2:
        tail2 = Path(*parts[-2:])
        pp3 = (run_dir / tail2).resolve()
        if pp3.exists():
            return str(pp3)

    return str(pp)


# -----------------------------
# Optional pixel metrics
# -----------------------------
def pil_to_tensor01(pil_img: Image.Image, size: int = 256) -> torch.Tensor:
    img = pil_img.convert("RGB").resize((size, size), resample=Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.mse_loss(a, b).item())


def psnr_from_mse(m: float, max_val: float = 1.0) -> float:
    if m <= 0:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / m))


def rgb_to_gray01(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[0], x[1], x[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.clamp(0, 1)


def ssim_simple_gray(a: np.ndarray, b: np.ndarray) -> float:
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    mu_a = a.mean()
    mu_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(num / den)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_g = rgb_to_gray01(a).cpu().numpy()
    b_g = rgb_to_gray01(b).cpu().numpy()
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        return float(sk_ssim(a_g, b_g, data_range=1.0))
    except Exception:
        return ssim_simple_gray(a_g, b_g)


# -----------------------------
# CLIP embeddings
# -----------------------------
def get_clip(device: str, model_name: str):
    from transformers import CLIPModel, CLIPProcessor
    try:
        processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
    except TypeError:
        processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    return model, processor


@torch.no_grad()
@torch.no_grad()
def _unwrap_clip_feats(model, feats, inputs: Dict[str, torch.Tensor], kind: str) -> torch.Tensor:
    """
    Transformers compatibility:
    - Newer transformers: get_*_features returns a Tensor
    - Some versions/builds: may return BaseModelOutputWithPooling or other output objects
    """
    if feats is None:
        # force a forward pass as a last resort
        out = model(**inputs)
        feats = getattr(out, "image_embeds" if kind == "image" else "text_embeds", out)

    if torch.is_tensor(feats):
        return feats

    # Common output attributes
    for attr in (("image_embeds", "text_embeds") if kind == "image" else ("text_embeds", "image_embeds")):
        if hasattr(feats, attr):
            v = getattr(feats, attr)
            if torch.is_tensor(v):
                return v

    for attr in ("pooler_output", "last_hidden_state"):
        if hasattr(feats, attr):
            v = getattr(feats, attr)
            if torch.is_tensor(v):
                return v[:, 0] if attr == "last_hidden_state" else v

    # tuple/list outputs
    if isinstance(feats, (tuple, list)):
        for v in feats:
            if torch.is_tensor(v):
                return v

    # Fallback: run sub-models directly and project if available
    if kind == "image" and hasattr(model, "vision_model"):
        vout = model.vision_model(pixel_values=inputs.get("pixel_values"))
        pooled = getattr(vout, "pooler_output", None)
        if pooled is None and hasattr(vout, "last_hidden_state") and torch.is_tensor(vout.last_hidden_state):
            pooled = vout.last_hidden_state[:, 0]
        if pooled is None and isinstance(vout, (tuple, list)) and len(vout) > 1 and torch.is_tensor(vout[1]):
            pooled = vout[1]
        if pooled is None:
            raise TypeError(f"Cannot unwrap vision output type: {type(vout)}")
        proj = getattr(model, "visual_projection", None)
        if callable(proj):
            pooled = proj(pooled)
        return pooled

    if kind == "text" and hasattr(model, "text_model"):
        tout = model.text_model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
        )
        pooled = getattr(tout, "pooler_output", None)
        if pooled is None and hasattr(tout, "last_hidden_state") and torch.is_tensor(tout.last_hidden_state):
            pooled = tout.last_hidden_state[:, 0]
        if pooled is None and isinstance(tout, (tuple, list)) and len(tout) > 1 and torch.is_tensor(tout[1]):
            pooled = tout[1]
        if pooled is None:
            raise TypeError(f"Cannot unwrap text output type: {type(tout)}")
        proj = getattr(model, "text_projection", None)
        if callable(proj):
            pooled = proj(pooled)
        return pooled

    raise TypeError(f"Unexpected CLIP feature type: {type(feats)}")

def clip_encode_images(model, processor, device: str, pil_list: List[Image.Image]) -> torch.Tensor:
    """
    Robust CLIP image embedding extraction across transformers versions.
    Returns L2-normalized features [B, D].
    """
    inputs = processor(images=[im.convert("RGB") for im in pil_list], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Prefer official helper if available
    feats = None
    try:
        feats = model.get_image_features(**inputs)
    except Exception:
        try:
            out = model(**inputs)
            feats = getattr(out, "image_embeds", out)
        except Exception:
            feats = None

    feats = _unwrap_clip_feats(model, feats, inputs, kind="image")
    feats = F.normalize(feats.float(), dim=-1)
    return feats


@torch.no_grad()
def clip_encode_texts(model, processor, device: str, texts: List[str]) -> torch.Tensor:
    """
    Robust CLIP text embedding extraction across transformers versions.
    Returns L2-normalized features [B, D].
    """
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    feats = None
    try:
        feats = model.get_text_features(**inputs)
    except Exception:
        try:
            out = model(**inputs)
            feats = getattr(out, "text_embeds", out)
        except Exception:
            feats = None

    feats = _unwrap_clip_feats(model, feats, inputs, kind="text")
    feats = F.normalize(feats.float(), dim=-1)
    return feats
# -----------------------------
# FID via Inception (avgpool features)
# -----------------------------
def get_inception(device: str):
    from torchvision.models import inception_v3, Inception_V3_Weights
    from torchvision.models.feature_extraction import create_feature_extractor

    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights)  # avoid aux_logits arg for compatibility
    model.eval().to(device)

    extractor = create_feature_extractor(model, return_nodes={"avgpool": "feat"})
    extractor.eval().to(device)
    preprocess = weights.transforms()
    return extractor, preprocess


@torch.no_grad()
def inception_feats_from_paths(extractor, preprocess, device: str, paths: List[str], batch: int = 16) -> np.ndarray:
    feats = []
    for i in range(0, len(paths), batch):
        chunk_paths = paths[i:i + batch]
        ims = []
        for p in chunk_paths:
            ims.append(Image.open(p).convert("RGB"))
        x = torch.stack([preprocess(im) for im in ims], dim=0).to(device)
        out = extractor(x)["feat"]     # [B,2048,1,1]
        out = out.flatten(1)           # [B,2048]
        feats.append(out.detach().cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 2048), dtype=np.float32)


def cov_np(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    return (x.T @ x) / max(1, (x.shape[0] - 1))


def sqrtm_psd(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    mat = (mat + mat.T) / 2.0
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, eps, None)
    return (v * np.sqrt(w)) @ v.T


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    mu1 = mu1.astype(np.float64)
    mu2 = mu2.astype(np.float64)
    sigma1 = sigma1.astype(np.float64)
    sigma2 = sigma2.astype(np.float64)
    diff = mu1 - mu2
    covmean = sqrtm_psd((sigma1 @ sigma2 + sigma2 @ sigma1) / 2.0 + np.eye(sigma1.shape[0]) * eps, eps=eps)
    return float(np.real(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)))


# -----------------------------
# Records IO / scanning fallback
# -----------------------------
def read_records(records_path: Path, limit_records: int) -> List[Dict]:
    recs = []
    with records_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
            if limit_records > 0 and len(recs) >= limit_records:
                break
    return recs


def scan_records(run_dir: Path, limit_records: int) -> List[Dict]:
    images_dir = run_dir / "images"
    inputs_dir = run_dir / "inputs"
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not inputs_dir.exists():
        raise FileNotFoundError(f"inputs_dir not found: {inputs_dir}")

    img_re = re.compile(r"(?P<prefix>.+)_k(?P<k>\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)

    # map prefix -> list of gen paths
    gen_map: Dict[str, List[str]] = {}
    for p in images_dir.iterdir():
        m = img_re.match(p.name)
        if not m:
            continue
        prefix = m.group("prefix")
        gen_map.setdefault(prefix, []).append(str(p))

    # find input path by prefix
    in_files = list(inputs_dir.iterdir())
    recs = []
    for prefix, gpaths in gen_map.items():
        gpaths.sort(key=lambda s: int(img_re.match(Path(s).name).group("k")))
        input_path = ""
        for inf in in_files:
            if inf.name.startswith(prefix):
                input_path = str(inf)
                break
        recs.append({
            "index": int(prefix.split("_", 1)[0]) if prefix.split("_", 1)[0].isdigit() else 0,
            "caption": "",
            "input_path": input_path,
            "gen_paths": gpaths,
        })

    recs.sort(key=lambda r: r.get("index", 0))
    if limit_records > 0:
        recs = recs[:limit_records]
    return recs


def safe_nanmean(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


# -----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--limit_records", type=int, default=2000)

    ap.add_argument("--metrics", type=str, default="clip_it,clip_ii,div_clip,fid",
                    help="See script help for this option.")
    ap.add_argument("--aggregate", type=str, default="mean",
                    choices=["mean", "best_clip_it", "best_clip_ii", "best_tradeoff", "best_ssim", "best_mse"])
    ap.add_argument("--tradeoff_alpha", type=float, default=1.0,
                    help="best_tradeoff: maximize clip_it + alpha * clip_ii")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--image_size", type=int, default=256, help="pixel metrics resize size")
    ap.add_argument("--inception_batch", type=int, default=64)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    records_path = run_dir / "records.jsonl"

    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]

    # load records
    if records_path.exists():
        recs = read_records(records_path, args.limit_records)
        print(f"[INFO] Loaded records: {records_path} (n={len(recs)})")
    else:
        print(f"[WARN] records.jsonl missing. Scanning images under: {run_dir}")
        recs = scan_records(run_dir, args.limit_records)
        print(f"[INFO] Built records by scanning (n={len(recs)})")

    need_clip = any(m in metrics for m in ["clip_it", "clip_ii", "div_clip"])
    need_fid = ("fid" in metrics)
    need_pixel = any(m in metrics for m in ["mse", "psnr", "ssim"])

    clip_model = clip_proc = None
    if need_clip:
        clip_model, clip_proc = get_clip(args.device, args.clip_model)

    inception = preprocess = None
    if need_fid:
        inception, preprocess = get_inception(args.device)

    per_rows = []
    fid_gen_paths: List[str] = []
    fid_real_paths: List[str] = []

    # for global summary
    s_clip_it = []
    s_clip_ii = []
    s_div = []
    s_mse = []
    s_psnr = []
    s_ssim = []
    n_records = 0

    for rec in tqdm(recs, desc="Evaluating"):
        idx = int(rec.get("index", rec.get("flickr_index", rec.get("cc3m_index", 0))))
        caption = str(rec.get("caption", "") or rec.get("prompt", "")).strip()
        input_path = resolve_existing_path(rec.get("input_path", ""), run_dir)
        gen_paths = rec.get("gen_paths", [])

        if not input_path or not Path(input_path).exists():
            # last resort: try inputs/<idx*>.jpg
            cands = sorted((run_dir / "inputs").glob(f"{idx:05d}*"))
            if cands:
                input_path = str(cands[0])
            else:
                raise FileNotFoundError(f"input image not found for idx={idx}: {rec.get('input_path', '')}")

        gen_paths = [resolve_existing_path(p, run_dir) for p in gen_paths]
        gen_paths = [p for p in gen_paths if Path(p).exists()]
        if not gen_paths:
            continue

        in_pil = Image.open(input_path).convert("RGB")

        # Load generated images once (avoid re-opening on disk in multiple metrics)
        gen_pils = [Image.open(p).convert("RGB") for p in gen_paths] if (need_clip or need_pixel) else []

        # optional pixel GT tensor
        gt = pil_to_tensor01(in_pil, size=args.image_size) if need_pixel else None

        # CLIP embeddings
        clip_it_list: List[float] = []
        clip_ii_list: List[float] = []
        div_clip = float("nan")

        if need_clip:
            # Encode input + all generated images in ONE CLIP forward pass to reduce overhead
            all_embs = clip_encode_images(clip_model, clip_proc, args.device, [in_pil] + gen_pils)  # [1+K, D]
            in_emb = all_embs[:1]      # [1, D]
            gen_embs = all_embs[1:]    # [K, D]

            # clip_ii
            clip_ii = (gen_embs * in_emb.expand_as(gen_embs)).sum(dim=-1)  # [K]
            clip_ii_list = [float(x) for x in clip_ii.detach().cpu().tolist()]

            # clip_it (needs caption)
            if caption:
                txt_emb = clip_encode_texts(clip_model, clip_proc, args.device, [caption])  # [1, D]
                clip_it = (gen_embs * txt_emb.expand_as(gen_embs)).sum(dim=-1)  # [K]
                clip_it_list = [float(x) for x in clip_it.detach().cpu().tolist()]
            else:
                clip_it_list = [float("nan")] * len(gen_paths)

            # diversity among K samples
            if len(gen_paths) >= 2:
                sim = gen_embs @ gen_embs.T  # [K, K]
                k = sim.shape[0]
                mean_offdiag = (sim.sum() - torch.trace(sim)) / (k * (k - 1))
                div_clip = float(1.0 - mean_offdiag.item())
            else:
                div_clip = 0.0

        # per-image pixel metrics + packing items
        items = []
        for j, p in enumerate(gen_paths):
            row = {"gen_path": p}
            if need_clip:
                row["clip_ii"] = clip_ii_list[j]
                row["clip_it"] = clip_it_list[j]
            if need_pixel:
                gp = gen_pils[j]
                gen_t = pil_to_tensor01(gp, size=args.image_size)
                if "mse" in metrics or "psnr" in metrics:
                    m = mse(gen_t, gt)
                    row["mse"] = m
                    if "psnr" in metrics:
                        row["psnr"] = psnr_from_mse(m)
                if "ssim" in metrics:
                    row["ssim"] = ssim(gen_t, gt)
            items.append(row)

        # aggregate over K
        def pick_best(items: List[Dict]) -> Dict:
            if args.aggregate == "mean":
                out = {}
                # mean over numeric keys (nanmean)
                keys = [k for k in items[0].keys() if k != "gen_path"]
                for k in keys:
                    out[k] = safe_nanmean([it.get(k, float("nan")) for it in items])
                return out

            if args.aggregate == "best_clip_it":
                best = max(items, key=lambda d: d.get("clip_it", float("-inf")))
                return {k: v for k, v in best.items() if k != "gen_path"}

            if args.aggregate == "best_clip_ii":
                best = max(items, key=lambda d: d.get("clip_ii", float("-inf")))
                return {k: v for k, v in best.items() if k != "gen_path"}

            if args.aggregate == "best_tradeoff":
                def score(d):
                    return float(d.get("clip_it", float("-inf"))) + args.tradeoff_alpha * float(d.get("clip_ii", float("-inf")))
                best = max(items, key=score)
                return {k: v for k, v in best.items() if k != "gen_path"}

            if args.aggregate == "best_ssim":
                best = max(items, key=lambda d: d.get("ssim", float("-inf")))
                return {k: v for k, v in best.items() if k != "gen_path"}

            if args.aggregate == "best_mse":
                best = min(items, key=lambda d: d.get("mse", float("inf")))
                return {k: v for k, v in best.items() if k != "gen_path"}

            raise ValueError("bad aggregate")

        agg = pick_best(items)

        out_row = {
            "index": idx,
            "n_gen": len(gen_paths),
            "div_clip": div_clip,
        }
        # add aggregated
        for k, v in agg.items():
            out_row[k] = v

        per_rows.append(out_row)
        n_records += 1

        # summary accumulators
        if "clip_it" in out_row:
            s_clip_it.append(out_row["clip_it"])
        if "clip_ii" in out_row:
            s_clip_ii.append(out_row["clip_ii"])
        if "div_clip" in metrics:
            s_div.append(div_clip)

        if "mse" in out_row:
            s_mse.append(out_row["mse"])
        if "psnr" in out_row:
            s_psnr.append(out_row["psnr"])
        if "ssim" in out_row:
            s_ssim.append(out_row["ssim"])

        if need_fid:
            fid_real_paths.append(input_path)
            fid_gen_paths.extend(gen_paths)

    summary = {"n_records": n_records, "aggregate": args.aggregate}

    if n_records > 0:
        if need_clip:
            summary["clip_it"] = safe_nanmean(s_clip_it)
            summary["clip_ii"] = safe_nanmean(s_clip_ii)
            summary["div_clip"] = safe_nanmean(s_div) if "div_clip" in metrics else None
        if need_pixel:
            if "mse" in metrics:
                summary["mse"] = safe_nanmean(s_mse)
            if "psnr" in metrics:
                summary["psnr"] = safe_nanmean(s_psnr)
            if "ssim" in metrics:
                summary["ssim"] = safe_nanmean(s_ssim)

    if need_fid:
        if len(fid_gen_paths) >= 2 and len(fid_real_paths) >= 2:
            # compute features
            real_feats = inception_feats_from_paths(inception, preprocess, args.device, fid_real_paths, batch=args.inception_batch)
            gen_feats = inception_feats_from_paths(inception, preprocess, args.device, fid_gen_paths, batch=args.inception_batch)

            if real_feats.shape[0] >= 2 and gen_feats.shape[0] >= 2:
                mu_r, mu_g = real_feats.mean(axis=0), gen_feats.mean(axis=0)
                sr, sg = cov_np(real_feats), cov_np(gen_feats)
                summary["fid"] = frechet_distance(mu_r, sr, mu_g, sg)
            else:
                summary["fid"] = None
        else:
            summary["fid"] = None

    out_json = run_dir / "eval_summary.json"
    out_csv = run_dir / "eval_per_record.csv"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if per_rows:
        keys = sorted(per_rows[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_rows:
                w.writerow(r)

    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[DONE] {out_json}")
    print(f"[DONE] {out_csv}")


if __name__ == "__main__":
    main()


