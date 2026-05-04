"""
Deep Stem Isolation Tool
========================
Runs a second Demucs pass on selected stems to remove bleed.
Uses existing stems from htdemucs_6s (produced by app.py) — does NOT
re-run the initial separation.

Only drums and guitar benefit from double-pass isolation.
All other stems are skipped (use the originals).

Usage:
    python tools/deep_isolate.py separated/htdemucs_6s/my_song
    python tools/deep_isolate.py separated/htdemucs_6s/my_song --gpu
    python tools/deep_isolate.py separated/htdemucs_6s/my_song --adaptive-bandpass

Output:
    Saves *-deep-isolate.wav files alongside the originals:
    separated/htdemucs_6s/my_song/
    ├── drums.wav
    ├── drums-deep-isolate.wav
    ├── guitar.wav
    └── guitar-deep-isolate.wav

Note:
    --bootleg in app.py runs deep isolation automatically.
    This standalone tool is for testing or re-running independently.

Requirements:
    demucs, librosa, numpy, scipy, soundfile, torch
"""

import os
import sys
import shutil
import argparse
import subprocess
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

# Only these stems benefit from double-pass
DOUBLE_PASS_STEMS = {"drums", "guitar"}

# htdemucs (4-stem) outputs: bass, drums, vocals, other
# Map each 6s stem to the closest 4-stem bucket
STEM_TO_4S = {
    "drums": "drums",
    "guitar": "other",
}


def run_demucs(input_path, output_dir, device="cpu"):
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", "htdemucs",
        "-d", device,
        "-o", output_dir,
        input_path
    ]
    subprocess.run(cmd, check=True)


def find_spectral_bounds(y, sr, energy_threshold=0.02):
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    energy_per_bin = np.mean(S ** 2, axis=1)
    total_energy = np.sum(energy_per_bin)
    if total_energy < 1e-10:
        return None, None
    cumulative = np.cumsum(energy_per_bin) / total_energy
    low_idx = np.searchsorted(cumulative, energy_threshold)
    high_idx = np.searchsorted(cumulative, 1.0 - energy_threshold)
    low_hz = max(freqs[low_idx], 20)
    high_hz = min(freqs[min(high_idx, len(freqs) - 1)], sr / 2 - 1)
    return float(low_hz), float(high_hz)


def adaptive_bandpass(y, sr):
    low, high = find_spectral_bounds(y, sr)
    if low is None or high is None or high <= low:
        return y
    sos = butter(4, [low, high], btype='band', fs=sr, output='sos')
    return sosfilt(sos, y)


def is_silent(y):
    return np.max(np.abs(y)) < 1e-3 or np.sqrt(np.mean(y ** 2)) < 1e-4


def main():
    parser = argparse.ArgumentParser(description="Double-pass Demucs on existing stems")
    parser.add_argument("stem_dir", help="Path to existing stems folder (e.g. separated/htdemucs_6s/MySong)")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA GPU")
    parser.add_argument("--adaptive-bandpass", action="store_true",
                        help="Apply adaptive frequency filtering after isolation")
    args = parser.parse_args()

    if not os.path.isdir(args.stem_dir):
        print(f"❌ Folder not found: {args.stem_dir}")
        sys.exit(1)

    device = "cuda" if args.gpu else "cpu"
    tmp_dir = os.path.join("separated", "_deep_tmp")

    print(f"\n🔬 Deep isolating stems in: {args.stem_dir} ({device.upper()})\n")

    for stem_name in DOUBLE_PASS_STEMS:
        stem_path = os.path.join(args.stem_dir, f"{stem_name}.wav")
        if not os.path.exists(stem_path):
            print(f"  ⏭️ {stem_name}.wav not found, skipping.")
            continue

        y, sr = librosa.load(stem_path, sr=None, mono=True)
        if is_silent(y):
            print(f"  ⏭️ {stem_name} is silent, skipping.")
            continue

        print(f"  🔄 Re-separating {stem_name}...")
        try:
            run_demucs(stem_path, tmp_dir, device=device)

            target = STEM_TO_4S[stem_name]
            resep_path = os.path.join(tmp_dir, "htdemucs", stem_name, f"{target}.wav")

            if os.path.exists(resep_path):
                y_clean, sr_clean = librosa.load(resep_path, sr=None, mono=True)

                if args.adaptive_bandpass and not is_silent(y_clean):
                    low, high = find_spectral_bounds(y_clean, sr_clean)
                    if low and high:
                        print(f"    📐 Bandpass: {low:.0f}–{high:.0f} Hz")
                        y_clean = adaptive_bandpass(y_clean, sr_clean)

                # Normalize to original peak
                orig_peak = np.max(np.abs(y))
                clean_peak = np.max(np.abs(y_clean))
                if clean_peak > 0 and orig_peak > 0:
                    y_clean = y_clean / clean_peak * orig_peak

                out_path = os.path.join(args.stem_dir, f"{stem_name}-deep-isolate.wav")
                sf.write(out_path, y_clean, sr_clean)
                print(f"  ✅ {stem_name}-deep-isolate.wav")
            else:
                print(f"  ⚠️ {stem_name}: pass 2 produced no output, skipping.")
        except Exception as e:
            print(f"  ⚠️ {stem_name} failed: {e}")

    # Cleanup temp files
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n🎯 Done. Compare with: python tools/compare_stems.py {args.stem_dir}")


if __name__ == "__main__":
    main()
