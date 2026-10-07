"""Build train/validation/test metadata from the seven NISQA corpus subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


SUBSETS = [
    "NISQA_TRAIN_LIVE",
    "NISQA_TRAIN_SIM",
    "NISQA_VAL_LIVE",
    "NISQA_VAL_SIM",
    "NISQA_TEST_FOR",
    "NISQA_TEST_LIVETALK",
    "NISQA_TEST_P501",
]
RECORD_COLUMNS = [
    "subset", "split", "domain", "db", "mos", "deg_path", "ref_path",
    "filename_deg", "filename_ref", "lang", "duration_sec", "exists_deg", "exists_ref",
]


def corpus_path(root: Path, value: str) -> Path:
    """Accept either slash convention used in the source CSVs."""
    path = Path(str(value).replace("\\", "/")).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def safe_duration(audio_path: Path) -> float | None:
    try:
        info = sf.info(str(audio_path))
        return float(info.frames / info.samplerate)
    except (OSError, RuntimeError, ZeroDivisionError):
        return None


def load_one_subset(root: Path, subset: str) -> pd.DataFrame:
    csv_path = root / subset / f"{subset}_file.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    required = ["db", "mos", "filepath_deg", "filepath_ref", "filename_deg", "filename_ref"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    split = "train" if "TRAIN" in subset else "val" if "VAL" in subset else "test"
    domain = "live" if "LIVE" in subset else "sim"
    df["mos"] = pd.to_numeric(df["mos"], errors="coerce")
    records = []
    for _, row in df.iterrows():
        if pd.isna(row["filepath_deg"]) or not np.isfinite(row["mos"]):
            continue
        deg_path = corpus_path(root, row["filepath_deg"])
        ref_path = None if pd.isna(row["filepath_ref"]) else corpus_path(root, row["filepath_ref"])
        records.append({
            "subset": subset,
            "split": split,
            "domain": domain,
            "db": row["db"],
            "mos": float(row["mos"]),
            "deg_path": str(deg_path),
            "ref_path": str(ref_path) if ref_path is not None else "",
            "filename_deg": row["filename_deg"],
            "filename_ref": row["filename_ref"],
            "lang": row.get("lang", ""),
            "duration_sec": safe_duration(deg_path) if deg_path.is_file() else None,
            "exists_deg": deg_path.is_file(),
            "exists_ref": ref_path.is_file() if ref_path is not None else False,
        })
    return pd.DataFrame(records, columns=RECORD_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nisqa-root", type=Path, required=True, help="Path to NISQA_Corpus.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    root = args.nisqa_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"NISQA root is not a directory: {root}")

    try:
        df = pd.concat([load_one_subset(root, subset) for subset in SUBSETS], ignore_index=True)
        # Training uses degraded speech only. Exclude missing, unreadable, and < 0.3 s audio.
        df = df[df["exists_deg"] & (df["duration_sec"] >= 0.3)].reset_index(drop=True)
        missing_splits = [split for split in ("train", "val", "test") if not (df["split"] == split).any()]
        if missing_splits:
            raise ValueError(f"No usable samples for splits: {', '.join(missing_splits)}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    out_csv = args.output_csv.expanduser().resolve()
    out_json = args.output_json.expanduser().resolve()
    if out_csv == out_json:
        parser.error("--output-csv and --output-json must name different files")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    summary = {
        "num_samples": int(len(df)),
        "split_counts": df["split"].value_counts().to_dict(),
        "domain_counts": df["domain"].value_counts().to_dict(),
        "subset_counts": df["subset"].value_counts().to_dict(),
        "nisqa_root": str(root),
        "output_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
