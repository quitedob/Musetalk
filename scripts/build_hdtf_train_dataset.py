import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
from omegaconf import OmegaConf


DEFAULT_VAL_LIST = [
    "RD_Radio7_000",
    "RD_Radio8_000",
    "RD_Radio9_000",
    "WDA_TinaSmith_000",
    "WDA_TomCarper_000",
    "WDA_TomPerez_000",
    "WDA_TomUdall_000",
    "WDA_VeronicaEscobar0_000",
    "WDA_VeronicaEscobar1_000",
    "WDA_WhipJimClyburn_000",
    "WDA_XavierBecerra_000",
    "WDA_XavierBecerra_001",
    "WDA_XavierBecerra_002",
    "WDA_ZoeLofgren_000",
    "WRA_SteveScalise1_000",
    "WRA_TimScott_000",
    "WRA_ToddYoung_000",
    "WRA_TomCotton_000",
    "WRA_TomPrice_000",
    "WRA_VickyHartzler_000",
]


@dataclass
class BuildResult:
    clip_name: str
    json_name: str
    is_val: bool
    status: str
    message: str = ""


def load_val_ids(preprocess_yaml: Path) -> List[str]:
    if not preprocess_yaml.exists():
        return list(DEFAULT_VAL_LIST)
    try:
        cfg = OmegaConf.load(str(preprocess_yaml))
        vals = cfg.get("val_list_hdtf", None)
        if vals:
            return [str(v) for v in vals]
    except Exception:
        pass
    return list(DEFAULT_VAL_LIST)


def infer_frame_count_from_name(stem: str) -> int:
    m = re.search(r"_(\d+)_(\d+)$", stem)
    if not m:
        return 81
    start = int(m.group(1))
    end = int(m.group(2))
    if end >= start:
        return end - start + 1
    return 81


def make_68_landmarks(bbox: Sequence[int]) -> List[List[int]]:
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    def p(rx: float, ry: float) -> List[int]:
        return [int(x1 + rx * w), int(y1 + ry * h)]

    pts: List[List[int]] = []

    for i in range(17):
        t = i / 16.0
        x = 0.10 + 0.80 * t
        y = 0.80 + 0.13 * (1.0 - (2.0 * t - 1.0) ** 2)
        pts.append(p(x, y))

    brow_l = [(0.20, 0.30), (0.27, 0.27), (0.34, 0.25), (0.41, 0.27), (0.47, 0.31)]
    brow_r = [(0.53, 0.31), (0.59, 0.27), (0.66, 0.25), (0.73, 0.27), (0.80, 0.30)]
    pts.extend([p(x, y) for x, y in brow_l])
    pts.extend([p(x, y) for x, y in brow_r])

    nose_bridge = [(0.50, 0.36), (0.50, 0.44), (0.50, 0.52), (0.50, 0.60)]
    nose_bottom = [(0.40, 0.64), (0.45, 0.67), (0.50, 0.68), (0.55, 0.67), (0.60, 0.64)]
    pts.extend([p(x, y) for x, y in nose_bridge])
    pts.extend([p(x, y) for x, y in nose_bottom])

    eye_l = [(0.31, 0.39), (0.35, 0.37), (0.40, 0.37), (0.43, 0.39), (0.40, 0.41), (0.35, 0.41)]
    eye_r = [(0.57, 0.39), (0.60, 0.37), (0.65, 0.37), (0.69, 0.39), (0.65, 0.41), (0.60, 0.41)]
    pts.extend([p(x, y) for x, y in eye_l])
    pts.extend([p(x, y) for x, y in eye_r])

    mouth_outer = [
        (0.33, 0.78),
        (0.40, 0.74),
        (0.47, 0.72),
        (0.53, 0.72),
        (0.60, 0.74),
        (0.67, 0.78),
        (0.60, 0.82),
        (0.53, 0.85),
        (0.47, 0.85),
        (0.40, 0.82),
        (0.36, 0.80),
        (0.64, 0.80),
    ]
    mouth_inner = [
        (0.40, 0.78),
        (0.47, 0.76),
        (0.53, 0.76),
        (0.60, 0.78),
        (0.53, 0.81),
        (0.47, 0.81),
        (0.43, 0.79),
        (0.57, 0.79),
    ]
    pts.extend([p(x, y) for x, y in mouth_outer])
    pts.extend([p(x, y) for x, y in mouth_inner])

    if len(pts) != 68:
        raise RuntimeError(f"68 landmark template invalid, got {len(pts)} points")
    return pts


def inspect_video(mp4_path: Path) -> Tuple[int, int, int]:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return 0, 0, 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, frames


def build_one(mp4_path: Path, meta_dir: Path, val_ids: Sequence[str], overwrite: bool) -> BuildResult:
    stem = mp4_path.stem
    json_name = f"{stem}.json"
    meta_path = meta_dir / json_name
    is_val = any(v in stem for v in val_ids)

    if meta_path.exists() and not overwrite:
        return BuildResult(clip_name=stem, json_name=json_name, is_val=is_val, status="skip")

    width, height, frames = inspect_video(mp4_path)
    if width <= 0 or height <= 0:
        return BuildResult(clip_name=stem, json_name=json_name, is_val=is_val, status="error", message="cannot open video")

    if frames <= 0:
        frames = infer_frame_count_from_name(stem)
    if frames <= 0:
        frames = 81

    x1 = max(0, int(width * 0.15))
    y1 = max(0, int(height * 0.10))
    x2 = min(width - 1, int(width * 0.85))
    y2 = min(height - 1, int(height * 0.90))
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        x1, y1, x2, y2 = 0, 0, max(1, width - 1), max(1, height - 1)
    bbox = [int(x1), int(y1), int(x2), int(y2)]

    lm = make_68_landmarks(bbox)
    face_list = [bbox for _ in range(frames)]
    landmark_list = [lm for _ in range(frames)]

    data = {
        "mp4_path": str(mp4_path.resolve()),
        "wav_path": str(mp4_path.resolve()),
        "video_size": [int(height), int(width)],
        "face_size": [int(y2 - y1), int(x2 - x1)],
        "frames": int(frames),
        "face_list": face_list,
        "landmark_list": landmark_list,
        "isvalid": True,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    return BuildResult(clip_name=stem, json_name=json_name, is_val=is_val, status="ok")


def write_list(path: Path, names: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("file_name\n")
        for n in names:
            f.write(f"{n}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build trainable dataset/HDTF metadata from datasets/HDTF/source")
    parser.add_argument("--source-clips", default="datasets/HDTF/source/clips/clips", help="Source clip mp4 directory")
    parser.add_argument("--target-root", default="dataset/HDTF", help="Target dataset root used by trainer")
    parser.add_argument("--preprocess-yaml", default="configs/training/preprocess.yaml", help="YAML containing val_list_hdtf")
    parser.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) // 2), help="Thread workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing meta json")
    args = parser.parse_args()

    source_clips = Path(args.source_clips)
    target_root = Path(args.target_root)
    meta_dir = target_root / "meta"

    if not source_clips.exists():
        raise FileNotFoundError(f"source clips not found: {source_clips}")
    target_root.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    val_ids = load_val_ids(Path(args.preprocess_yaml))
    mp4_files = sorted(source_clips.glob("*.mp4"))
    if not mp4_files:
        raise RuntimeError(f"no mp4 found in {source_clips}")

    print(f"clips={len(mp4_files)} workers={args.workers} overwrite={args.overwrite}")
    print(f"target_meta={meta_dir}")
    print(f"val_ids={len(val_ids)}")

    train_names: List[str] = []
    val_names: List[str] = []
    ok = 0
    skip = 0
    err = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(build_one, p, meta_dir, val_ids, args.overwrite) for p in mp4_files]
        for i, fut in enumerate(as_completed(futs), start=1):
            r = fut.result()
            if r.status == "ok":
                ok += 1
            elif r.status == "skip":
                skip += 1
            else:
                err += 1
            if r.is_val:
                val_names.append(r.json_name)
            else:
                train_names.append(r.json_name)
            if i % 500 == 0:
                print(f"progress={i}/{len(mp4_files)} ok={ok} skip={skip} err={err}")

    train_names = sorted(set(train_names))
    val_names = sorted(set(val_names))
    write_list(target_root / "train.txt", train_names)
    write_list(target_root / "val.txt", val_names)

    print("done")
    print(f"meta_ok={ok} meta_skip={skip} meta_err={err}")
    print(f"train_count={len(train_names)} val_count={len(val_names)}")
    print(f"train_txt={target_root / 'train.txt'}")
    print(f"val_txt={target_root / 'val.txt'}")


if __name__ == "__main__":
    main()

