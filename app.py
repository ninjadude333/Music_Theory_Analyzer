import os
import json
import shutil
import librosa
import numpy as np
import sys
import argparse
import torch
from dotenv import load_dotenv

load_dotenv()

# --- TENSORFLOW / KERAS LEGACY BRIDGE ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    import pretty_midi
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

def extract_rich_data(audio_path, use_gpu=False):
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"\n[1/2] 🥁 Isolating instruments using {device.upper()}...")
    
    os.system(f"python -m demucs.separate -n htdemucs -d {device} \"{audio_path}\"")
    
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join("separated", "htdemucs", filename)
    
    stems_found = []
    if os.path.exists(stem_dir):
        for s in ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]:
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
    print(f"\n[3/3] 🧠 Connecting to Ollama at {OLLAMA_HOST}...")
    try:
        # Quick connectivity check
        client.list()
        print(f"  ✅ Connected. Sending to {MODEL_NAME} (this may take a while)...")
        response = client.chat(model=MODEL_NAME, messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': json.dumps(data)}
        ])
        return response['message']['content']
    except Exception as e:
        return f"❌ Connection error: {e}. Check if Ollama is running at {OLLAMA_HOST}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--no-ollama", action="store_true")
    parser.add_argument("--gpu_on", action="store_true")
    parser.add_argument("--export-midi", action="store_true", help="Export MIDI file of the transcription")
    parser.add_argument("--export-multitrack-midi", action="store_true", help="Export multitrack MIDI with separate tracks per stem")
    parser.add_argument("--cleanup", action="store_true", help="Remove separated stems after processing")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ File '{args.input}' not found.")
        return

    rich_data = extract_rich_data(args.input, args.gpu_on)
    filename = os.path.splitext(os.path.basename(args.input))[0]
    stem_dir = os.path.join("separated", "htdemucs", filename)
    output_dir = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(output_dir, exist_ok=True)

    # Export MIDI if requested
    if args.export_midi and rich_data.get("midi_data"):
        midi_path = os.path.join(output_dir, f"{filename}.mid")
        rich_data["midi_data"].write(midi_path)
        print(f"🎹 MIDI exported to: {midi_path}")

    # Export multitrack MIDI — transcribe each stem separately
    if args.export_multitrack_midi and os.path.exists(stem_dir):
        print("🎶 Transcribing stems to multitrack MIDI...")
        multitrack = pretty_midi.PrettyMIDI()
        for stem_name in ["bass", "drums", "vocals", "other"]:
            stem_path = os.path.join(stem_dir, f"{stem_name}.wav")
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

    # Strip non-serializable midi_data before JSON usage
    serializable_data = {k: v for k, v in rich_data.items() if k != "midi_data"}

    if args.no_ollama:
        json_path = os.path.join(output_dir, f"{filename}_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=4)
        print(f"📄 JSON saved to: {json_path}")
    else:
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