"""Extract fixed-length AST filterbanks and split manifests from NISQA metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio


def resolve_audio_path(value: str, metadata_dir: Path) -> Path:
    path = Path(str(value).replace("\\", "/")).expanduser()
    return (path if path.is_absolute() else metadata_dir / path).resolve()


def load_audio_mono(path: str, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(path, always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1, dtype=np.float32)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
    if not wav.size or not np.isfinite(wav).all():
        raise ValueError(f"Audio is empty or contains non-finite samples: {path}")
    return wav.astype(np.float32)


def compute_kaldi_fbank_ast(wav: np.ndarray, sr: int = 16000, n_mels: int = 128) -> np.ndarray:
    """Use mean-centered audio, a 25 ms Hanning window, and a 10 ms hop.

    This preserves the original implementation's Kaldi frontend exactly;
    the output has shape [frequency, time].
    """
    waveform = torch.from_numpy(wav).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"Unexpected waveform ndim: {waveform.ndim}")
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=n_mels,
        dither=0.0,
        frame_length=25,
        frame_shift=10,
    )
    return fbank.transpose(0, 1).cpu().numpy().astype(np.float32)


def pad_or_truncate_fbank(fbank: np.ndarray, target_length: int) -> np.ndarray:
    frames = fbank.shape[1]
    if frames < target_length:
        pad = np.zeros((fbank.shape[0], target_length - frames), dtype=np.float32)
        fbank = np.concatenate([fbank, pad], axis=1)
    elif frames > target_length:
        fbank = fbank[:, :target_length]
    return fbank.astype(np.float32)


def load_metadata(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = ["subset", "split", "domain", "mos", "deg_path"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in metadata: {missing}")
    if df.empty:
        raise ValueError("Input metadata has no samples")
    if df[required].isna().any().any():
        raise ValueError("Required metadata fields must not be empty")
    if not df["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Metadata split must be train, val, or test")
    if not df["domain"].isin(["live", "sim"]).all():
        raise ValueError("Metadata domain must be live or sim")
    for split in ("train", "val", "test"):
        if not (df["split"] == split).any():
            raise ValueError(f"Input metadata has no {split} samples")
    df["mos"] = pd.to_numeric(df["mos"], errors="raise")
    if not np.isfinite(df["mos"]).all():
        raise ValueError("MOS values must be finite numbers")
    df["deg_path"] = df["deg_path"].map(lambda value: str(resolve_audio_path(value, input_csv.parent)))
    for audio_path in df["deg_path"]:
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"Missing degraded audio: {audio_path}")
    if "ref_path" in df.columns:
        df["ref_path"] = df["ref_path"].map(
            lambda value: "" if pd.isna(value) or not str(value).strip()
            else str(resolve_audio_path(value, input_csv.parent))
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument(
        "--stats-split", choices=("train", "all"), default="train",
        help="Use training features for normalization (default); all reproduces legacy all-split statistics.",
    )
    args = parser.parse_args()
    if args.target_sr <= 0 or args.target_length <= 0:
        parser.error("--target-sr and --target-length must be positive")
    input_csv = args.input_csv.expanduser().resolve()
    try:
        df = load_metadata(input_csv)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    out_dir = args.output_dir.expanduser().resolve()
    out_csv = out_dir / "fbank_metadata.csv"
    out_stats = out_dir / "fbank_stats.json"
    rows = []
    total_sum = total_sq_sum = 0.0
    total_count = stats_num_samples = 0
    for idx, row in df.iterrows():
        deg_path = str(row["deg_path"])
        try:
            wav = load_audio_mono(deg_path, args.target_sr)
            if wav.size < round(args.target_sr * 0.025):
                raise ValueError("Audio is shorter than the 25 ms analysis window")
            fbank = compute_kaldi_fbank_ast(wav, sr=args.target_sr)
            if not fbank.size or not np.isfinite(fbank).all():
                raise ValueError("Filterbank is empty or contains non-finite values")
        except (OSError, RuntimeError, ValueError, AssertionError) as exc:
            parser.error(f"Feature extraction failed for {deg_path}: {exc}")
        raw_time_frames = int(fbank.shape[1])
        fbank = pad_or_truncate_fbank(fbank, args.target_length)
        npy_path = out_dir / "npy" / row["split"] / row["domain"] / f"{idx:08d}.npy"
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, fbank)

        # Include padding, as in the original pipeline; choose training-only by default.
        if args.stats_split == "all" or row["split"] == "train":
            total_sum += float(fbank.sum(dtype=np.float64))
            total_sq_sum += float(np.square(fbank, dtype=np.float64).sum(dtype=np.float64))
            total_count += int(fbank.size)
            stats_num_samples += 1
        rows.append({
            "subset": row["subset"],
            "split": row["split"],
            "domain": row["domain"],
            "mos": float(row["mos"]),
            "deg_path": deg_path,
            "ref_path": row.get("ref_path", ""),
            "fbank_path": str(npy_path),
            "freq_bins": int(fbank.shape[0]),
            "time_frames_raw": raw_time_frames,
            "time_frames_final": int(fbank.shape[1]),
            "storage_shape": "F,T",
            "sample_rate": args.target_sr,
            "target_length": args.target_length,
            "feature_type": "kaldi_fbank_ast_paper",
        })

    if total_count == 0:
        parser.error(f"No feature values available for --stats-split {args.stats_split}")
    mean = total_sum / total_count
    variance = max(total_sq_sum / total_count - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        parser.error("Normalization statistics must have finite mean and positive standard deviation")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    split_outputs = {}
    for split in ("train", "val", "test"):
        split_csv = out_dir / f"fbank_{split}.csv"
        out_df[out_df["split"] == split].to_csv(split_csv, index=False)
        split_outputs[f"fbank_{split}_csv"] = str(split_csv)
    test_bench_csv = out_dir / "fbank_test_bench100.csv"
    out_df[out_df["split"] == "test"].head(100).to_csv(test_bench_csv, index=False)
    split_outputs["fbank_test_bench100_csv"] = str(test_bench_csv)
    stats = {
        "num_samples": int(len(out_df)),
        "sample_rate": args.target_sr,
        "feature_type": "kaldi_fbank_ast_paper",
        "n_mels": 128,
        "frame_length_ms": 25,
        "frame_shift_ms": 10,
        "window_type": "hanning",
        "use_energy": False,
        "dither": 0.0,
        "target_length": args.target_length,
        "stats_split": args.stats_split,
        "stats_num_samples": stats_num_samples,
        "stats_num_values": total_count,
        "stats_include_padding": True,
        "dataset_fbank_mean": float(mean),
        "dataset_fbank_std": std,
        "output_csv": str(out_csv),
        **split_outputs,
    }
    out_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
