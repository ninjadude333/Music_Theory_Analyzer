import os
import json
import shutil
import subprocess
import librosa
import numpy as np
import sys
import time
import argparse
import torch
from dotenv import load_dotenv

load_dotenv()

# --- TENSORFLOW / KERAS LEGACY BRIDGE ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    import pretty_midi
    import soundfile as sf
    import noisereduce as nr
    from scipy.signal import butter, sosfilt
    from ollama import Client
    from httpx import Timeout
    from basic_pitch.inference import predict as bp_predict
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma4:e4b")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SYSTEM_PROMPT = """
You are an expert Music Theory Professor and Studio Producer. 
Identify the Key/Mode, provide Roman Numeral analysis, describe song form, and give soloing advice.
"""

# Initialize the remote client with a 5-minute timeout
client = Client(host=OLLAMA_HOST, timeout=Timeout(connect=10, read=300, write=300, pool=300))

# Per-instrument frequency bands (low_hz, high_hz) for bandpass filtering
STEM_FREQ_BANDS = {
    "bass":    (30, 300),
    "drums":   (40, 12000),
    "guitar":  (80, 6000),
    "piano":   (60, 8000),
    "vocals":  (120, 10000),
    "other":   (None, None),
}

# Stems that benefit from double-pass isolation
DEEP_ISOLATE_STEMS = {"drums", "guitar"}
STEM_TO_4S = {"drums": "drums", "guitar": "other"}


def _bandpass(y, sr, low, high):
    sos = butter(5, [low, high], btype='band', fs=sr, output='sos')
    return sosfilt(sos, y)


def enhance_stems(stem_dir, use_bandpass=False):
    print("\n✨ Enhancing stems (noise reduction + normalization)...")
    for stem_name, (low, high) in STEM_FREQ_BANDS.items():
        stem_path = os.path.join(stem_dir, f"{stem_name}.wav")
        if not os.path.exists(stem_path):
            continue
        try:
            y, sr = librosa.load(stem_path, sr=None, mono=False)
            mono = y.mean(axis=0) if y.ndim > 1 else y

            if np.max(np.abs(mono)) < 1e-4:
                print(f"  ⏭️ {stem_name} is silent, skipping.")
                continue

            cleaned = nr.reduce_noise(y=mono, sr=sr, prop_decrease=0.2, stationary=True)

            if use_bandpass and low and high:
                cleaned = _bandpass(cleaned, sr, low, high)

            orig_peak = np.max(np.abs(mono))
            peak = np.max(np.abs(cleaned))
            if peak > 0 and orig_peak > 0:
                cleaned = cleaned / peak * orig_peak

            out_path = os.path.join(stem_dir, f"{stem_name}-enhanced.wav")
            sf.write(out_path, cleaned, sr)
            print(f"  ✅ {stem_name}-enhanced.wav")
        except Exception as e:
            print(f"  ⚠️ Failed to enhance {stem_name}: {e}")


def deep_isolate_stems(stem_dir, device="cuda"):
    print(f"\n🔬 Deep isolating stems ({device.upper()})...")
    tmp_dir = os.path.join("separated", "_deep_tmp")

    for stem_name in DEEP_ISOLATE_STEMS:
        stem_path = os.path.join(stem_dir, f"{stem_name}.wav")
        if not os.path.exists(stem_path):
            print(f"  ⏭️ {stem_name}.wav not found, skipping.")
            continue

        y, sr = librosa.load(stem_path, sr=None, mono=True)
        if np.max(np.abs(y)) < 1e-3:
            print(f"  ⏭️ {stem_name} is silent, skipping.")
            continue

        print(f"  🔄 Re-separating {stem_name}...")
        try:
            cmd = [
                sys.executable, "-m", "demucs.separate",
                "-n", "htdemucs", "-d", device, "-o", tmp_dir, stem_path
            ]
            subprocess.run(cmd, check=True)

            target = STEM_TO_4S[stem_name]
            resep_path = os.path.join(tmp_dir, "htdemucs", stem_name, f"{target}.wav")

            if os.path.exists(resep_path):
                y_clean, sr_clean = librosa.load(resep_path, sr=None, mono=True)
                orig_peak = np.max(np.abs(y))
                clean_peak = np.max(np.abs(y_clean))
                if clean_peak > 0 and orig_peak > 0:
                    y_clean = y_clean / clean_peak * orig_peak
                sf.write(os.path.join(stem_dir, f"{stem_name}-deep-isolate.wav"), y_clean, sr_clean)
                print(f"  ✅ {stem_name}-deep-isolate.wav")
            else:
                print(f"  ⚠️ {stem_name}: pass 2 produced no output, skipping.")
        except Exception as e:
            print(f"  ⚠️ {stem_name} failed: {e}")

    shutil.rmtree(tmp_dir, ignore_errors=True)


def extract_rich_data(audio_path, use_gpu=True):
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"\n[1/2] 🥁 Isolating instruments using {device.upper()}...")
    
    os.system(f"python -m demucs.separate -n htdemucs_6s -d {device} \"{audio_path}\"")
    
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join("separated", "htdemucs_6s", filename)
    
    stems_found = []
    if os.path.exists(stem_dir):
        for s in ["bass.wav", "drums.wav", "guitar.wav", "piano.wav", "other.wav", "vocals.wav"]:
            if os.path.exists(os.path.join(stem_dir, s)):
                stems_found.append(s.split('.')[0])

    print(f"[2/2] 🎹 Analyzing audio features and transcribing...")
    y, sr = librosa.load(audio_path)
    
    tempo_track = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo_track[0] if isinstance(tempo_track, (list, tuple, np.ndarray)) else tempo_track
    
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    keys = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    estimated_key = keys[np.argmax(np.mean(chroma, axis=1))]

    notes = []
    midi_data = None
    try:
        _, midi_data, _ = bp_predict(audio_path)
        if hasattr(midi_data, 'instruments') and len(midi_data.instruments) > 0:
            raw_notes = midi_data.instruments[0].notes
            for n in raw_notes[:100]:
                notes.append({
                    "p": librosa.midi_to_note(n.pitch),
                    "s": round(float(n.start), 2),
                    "d": round(float(n.end - n.start), 2)
                })
    except Exception as e:
        print(f"⚠️ Transcription failed: {e}")

    return {
        "metadata": {
            "tempo_bpm": round(float(np.atleast_1d(tempo)[0]), 1),
            "estimated_key": estimated_key,
            "instruments": stems_found,
            "processing_device": device
        },
        "transcription_preview": notes,
        "midi_data": midi_data
    }


def get_ollama_analysis(data):
    print(f"\n🧠 Connecting to Ollama at {OLLAMA_HOST}...")

    try:
        test_client = Client(host=OLLAMA_HOST, timeout=Timeout(connect=5, read=5, write=5, pool=5))
        test_client.list()
        print(f"  ✅ Server reachable.")
    except Exception as e:
        return f"❌ Cannot reach Ollama at {OLLAMA_HOST}: {e}"

    try:
        models = [m.model for m in client.list().models]
        if not any(MODEL_NAME in m for m in models):
            available = ", ".join(models[:10]) or "(none)"
            return f"❌ Model '{MODEL_NAME}' not found. Available: {available}"
        print(f"  ✅ Model '{MODEL_NAME}' found.")
    except Exception as e:
        return f"❌ Failed to list models: {e}"

    print(f"  📡 Streaming response (this may take a while)...\n")
    try:
        start = time.time()
        chunks = []
        first_token = True
        for chunk in client.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': json.dumps(data)}
            ],
            stream=True
        ):
            token = chunk['message']['content']
            if first_token:
                elapsed = time.time() - start
                print(f"  ⏱️ First token in {elapsed:.1f}s\n")
                first_token = False
            sys.stdout.write(token)
            sys.stdout.flush()
            chunks.append(token)

        elapsed = time.time() - start
        result = "".join(chunks)
        print(f"\n\n  ✅ Done in {elapsed:.1f}s ({len(result)} chars)")
        return result
    except Exception as e:
        return f"❌ Streaming error: {e}"


def run_ollama_only(input_path):
    """Run Ollama analysis on an existing JSON data file or audio file's output."""
    filename = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(OUTPUT_DIR, filename)
    json_path = os.path.join(output_dir, f"{filename}_data.json")

    if not os.path.exists(json_path):
        print(f"❌ No data file found at: {json_path}")
        print(f"   Run the full pipeline first: python app.py {input_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📄 Loaded existing data from: {json_path}")
    analysis = get_ollama_analysis(data)

    print("\n" + "="*60)
    print("🎼 MUSIC THEORY ANALYSIS REPORT")
    print("="*60 + "\n")
    print(analysis)

    report_path = os.path.join(output_dir, f"{filename}_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 🎼 Music Theory Analysis: {filename}\n\n")
        f.write(f"**Tempo:** {data['metadata']['tempo_bpm']} BPM\n")
        f.write(f"**Estimated Key:** {data['metadata']['estimated_key']}\n")
        f.write(f"**Instruments:** {', '.join(data['metadata']['instruments'])}\n\n")
        f.write("---\n\n")
        f.write(analysis)
    print(f"\n💾 Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--ollama", action="store_true", help="Enable LLM analysis via Ollama")
    parser.add_argument("--ollama-only", action="store_true", help="Run only Ollama analysis on existing data (skip separation)")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU, use CPU only")
    parser.add_argument("--export-midi", action="store_true", help="Export MIDI file of the transcription")
    parser.add_argument("--export-multitrack-midi", action="store_true", help="Export multitrack MIDI with separate tracks per stem")
    parser.add_argument("--midi-from-enhanced", action="store_true", help="Use enhanced stems for MIDI transcription (implies --enhance_stems)")
    parser.add_argument("--enhance_stems", action="store_true", help="Noise reduce and normalize separated stems")
    parser.add_argument("--bandpass", action="store_true", help="Apply per-instrument frequency filtering (use with --enhance_stems)")
    parser.add_argument("--bootleg", action="store_true", help="Bootleg mode: enhance stems + deep isolate drums/guitar")
    parser.add_argument("--cleanup", action="store_true", help="Remove separated stems after processing")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ File '{args.input}' not found.")
        return

    # --ollama-only: skip everything, just run LLM on existing data
    if args.ollama_only:
        run_ollama_only(args.input)
        return

    use_gpu = not args.no_gpu

    # --bootleg activates enhance_stems + deep isolate
    if args.bootleg:
        args.enhance_stems = True

    # --midi-from-enhanced implies --enhance_stems
    if args.midi_from_enhanced:
        args.enhance_stems = True

    rich_data = extract_rich_data(args.input, use_gpu=use_gpu)
    filename = os.path.splitext(os.path.basename(args.input))[0]
    stem_dir = os.path.join("separated", "htdemucs_6s", filename)
    output_dir = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(output_dir, exist_ok=True)

    # Enhance stems if requested
    if args.enhance_stems and os.path.exists(stem_dir):
        enhance_stems(stem_dir, use_bandpass=args.bandpass)

    # Deep isolate for bootleg mode
    if args.bootleg and os.path.exists(stem_dir):
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        deep_isolate_stems(stem_dir, device=device)

    # Export MIDI if requested
    if args.export_midi and rich_data.get("midi_data"):
        midi_path = os.path.join(output_dir, f"{filename}.mid")
        rich_data["midi_data"].write(midi_path)
        print(f"🎹 MIDI exported to: {midi_path}")

    # Export multitrack MIDI — transcribe each stem separately
    if args.export_multitrack_midi and os.path.exists(stem_dir):
        print("🎶 Transcribing stems to multitrack MIDI...")
        multitrack = pretty_midi.PrettyMIDI()
        for stem_name in ["bass", "drums", "guitar", "piano", "vocals", "other"]:
            stem_suffix = "-enhanced" if args.midi_from_enhanced else ""
            stem_path = os.path.join(stem_dir, f"{stem_name}{stem_suffix}.wav")
            if not os.path.exists(stem_path):
                continue
            try:
                print(f"  ♪ Transcribing {stem_name}...")
                _, stem_midi, _ = bp_predict(stem_path)
                if hasattr(stem_midi, 'instruments') and stem_midi.instruments:
                    track = stem_midi.instruments[0]
                    track.name = stem_name
                    multitrack.instruments.append(track)
            except Exception as e:
                print(f"  ⚠️ Skipping {stem_name}: {e}")
        if multitrack.instruments:
            mt_path = os.path.join(output_dir, f"{filename}_multitrack.mid")
            multitrack.write(mt_path)
            print(f"🎹 Multitrack MIDI exported to: {mt_path}")

    # Always save JSON data to output dir
    serializable_data = {k: v for k, v in rich_data.items() if k != "midi_data"}
    json_path = os.path.join(output_dir, f"{filename}_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=4)
    print(f"📄 JSON saved to: {json_path}")

    # Ollama analysis if requested
    if args.ollama:
        analysis = get_ollama_analysis(serializable_data)
        print("\n" + "="*60)
        print("🎼 MUSIC THEORY ANALYSIS REPORT")
        print("="*60 + "\n")
        print(analysis)

        report_path = os.path.join(output_dir, f"{filename}_analysis.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# 🎼 Music Theory Analysis: {filename}\n\n")
            f.write(f"**Tempo:** {rich_data['metadata']['tempo_bpm']} BPM\n")
            f.write(f"**Estimated Key:** {rich_data['metadata']['estimated_key']}\n")
            f.write(f"**Instruments:** {', '.join(rich_data['metadata']['instruments'])}\n\n")
            f.write("---\n\n")
            f.write(analysis)
        print(f"\n💾 Report saved to: {report_path}")

    # Cleanup stems if requested
    if args.cleanup and os.path.exists(stem_dir):
        shutil.rmtree(stem_dir)
        print(f"🧹 Cleaned up stems: {stem_dir}")


if __name__ == "__main__":
    main()
