"""
Stem Comparison Tool
====================
Compares original vs enhanced stems in a given folder and outputs
a scored comparison table with SNR improvement, noise removed,
correlation, and an overall quality verdict.

Usage:
    python tools/compare_stems.py separated/htdemucs_6s/Paradise_City

Requirements:
    librosa, numpy, scipy (already in project requirements)
"""

import os
import sys
import argparse
import numpy as np
import librosa

STEMS = ["bass", "drums", "guitar", "piano", "vocals", "other"]


def compute_metrics(original, enhanced):
    # Trim to same length
    length = min(len(original), len(enhanced))
    orig = original[:length]
    enh = enhanced[:length]

    # Level-match enhanced to original before comparison
    orig_rms = np.sqrt(np.mean(orig ** 2)) + 1e-10
    enh_rms = np.sqrt(np.mean(enh ** 2)) + 1e-10
    enh_matched = enh * (orig_rms / enh_rms)

    diff = orig - enh_matched

    # SNR improvement: compare noise floors
    orig_power = np.mean(orig ** 2) + 1e-10
    diff_power = np.mean(diff ** 2) + 1e-10
    snr_improvement = 10 * np.log10(orig_power / diff_power)

    # Noise removed: energy of the difference as % of original
    noise_removed = (diff_power / orig_power) * 100

    # Correlation (signal preservation — 1.0 = identical)
    correlation = np.corrcoef(orig, enh_matched)[0, 1] if len(orig) > 1 else 1.0

    # Peak difference (on raw, not level-matched)
    peak_diff = np.max(np.abs(enh)) - np.max(np.abs(orig))

    return {
        "snr_improvement": round(snr_improvement, 2),
        "noise_removed_pct": round(noise_removed, 1),
        "correlation": round(correlation, 4),
        "peak_diff": round(peak_diff, 3),
    }


def grade(metrics):
    corr = metrics["correlation"]
    snr = metrics["snr_improvement"]
    if corr >= 0.98 and snr > 0:
        return "✅ Good"
    elif corr >= 0.95 and snr > 0:
        return "⚠️ OK"
    else:
        return "❌ Degraded"


def main():
    parser = argparse.ArgumentParser(description="Compare original vs enhanced stems")
    parser.add_argument("folder", help="Path to stems folder (e.g. separated/htdemucs_6s/MySong)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"❌ Folder not found: {args.folder}")
        sys.exit(1)

    print(f"\n🔍 Comparing stems in: {args.folder}\n")
    header = f"{'Stem':<10} {'SNR Δ (dB)':>10} {'Noise Removed':>14} {'Correlation':>12} {'Peak Δ':>8} {'Verdict':>12}"
    print(header)
    print("─" * len(header))

    found = 0
    comparisons = [
        ("enhanced", None, "Original vs Enhanced"),
        ("deep-isolate", None, "Original vs Deep Isolate"),
        ("enhanced", "deep-isolate", "Enhanced vs Deep Isolate"),
    ]
    for suffix_a, suffix_b, label in comparisons:
        pairs = []
        for stem in STEMS:
            if suffix_b:
                path_a = os.path.join(args.folder, f"{stem}-{suffix_a}.wav")
                path_b = os.path.join(args.folder, f"{stem}-{suffix_b}.wav")
            else:
                path_a = os.path.join(args.folder, f"{stem}.wav")
                path_b = os.path.join(args.folder, f"{stem}-{suffix_a}.wav")
            if os.path.exists(path_a) and os.path.exists(path_b):
                pairs.append((stem, path_a, path_b))

        if not pairs:
            continue

        if found > 0:
            print()
        print(f"  ── {label} ──")
        print(header)
        print("─" * len(header))

        for stem, path_a, path_b in pairs:
            found += 1
            a, sr = librosa.load(path_a, sr=None, mono=True)
            b, _ = librosa.load(path_b, sr=None, mono=True)
            m = compute_metrics(a, b)
            verdict = grade(m)
            print(
                f"{stem:<10} {m['snr_improvement']:>+9.2f}  {m['noise_removed_pct']:>12.1f}%  {m['correlation']:>11.4f} {m['peak_diff']:>+7.3f}  {verdict}"
            )

    if found == 0:
        print("⚠️ No original/enhanced stem pairs found.")
    else:
        print(f"\n📊 Compared {found} stem(s).")
        print("   Correlation close to 1.0 = signal well preserved.")
        print("   Noise Removed 5-15% = healthy cleanup range.\n")


if __name__ == "__main__":
    main()
