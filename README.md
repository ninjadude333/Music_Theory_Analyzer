# 🎼 Music Theory Analyzer

An AI-powered music analysis pipeline that separates instruments, transcribes audio, and generates detailed music theory reports using a local LLM.

Drop in any audio file and get back:
- 🥁 **Stem separation** (bass, drums, guitar, piano, vocals, other) via [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs_6s`)
- 🎹 **MIDI transcription** via [Basic Pitch](https://github.com/spotify/basic-pitch)
- 🎵 **Audio feature extraction** (tempo, key detection) via [Librosa](https://librosa.org/)
- 🎶 **Chord detection** per bar from MIDI transcription
- 🧠 **Expert music theory analysis** (key/mode, Roman numeral analysis, song form, soloing advice) via [Ollama](https://ollama.com/) LLM
- 🌐 **Interactive HTML report** with audio players, chord chart, theme switcher

Reports are saved as Markdown and HTML files alongside the separated stems for easy reference.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌───────────────┐
│  Audio File │────▶│   Demucs     │────▶│  Librosa +      │────▶│  Ollama LLM   │
│  (.mp3/wav) │     │  (Stems)     │     │  Basic Pitch    │     │  (Analysis)   │
└─────────────┘     └──────────────┘     └─────────────────┘     └───────────────┘
                           │                      │                        │
                           ▼                      ▼                        ▼
                    separated/htdemucs_6s/   output/<song>/           output/<song>/
                    <song>/bass.wav          <song>.mid               <song>_analysis.md
                    <song>/drums.wav         <song>_data.json         <song>_report.html
                    <song>/guitar.wav        <song>_multitrack.mid
                    <song>/piano.wav
                    <song>/vocals.wav
                    <song>/other.wav
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running on a reachable host (local or remote)
- A pulled model (default: `gemma4:e4b`) — any model works, just update `MODEL_NAME` in `app.py`
- (Optional) NVIDIA GPU + CUDA for faster stem separation
- (Optional) Chromaprint for song identification:
  - **Windows:** `fpcalc.exe` in project root or PATH ([download](https://acoustid.org/chromaprint))
  - **Linux:** `sudo apt install libchromaprint-tools`
  - **macOS:** `brew install chromaprint`

### Installation

```bash
git clone https://github.com/ninjadude333/Music_Theory_Analyzer.git
cd Music_Theory_Analyzer

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Edit the top of `app.py` to point to your Ollama instance:

```python
OLLAMA_HOST = "http://localhost:11434"  # or your remote server
MODEL_NAME = "gemma4:e4b"              # any Ollama model
```

---

## 🎯 Usage by Recording Type

### 🎙️ Studio Recording (CD, Spotify rip, official release)

Demucs handles these well out of the box. No enhancement needed.

```bash
# Basic: separate stems + extract audio data
python app.py input/my_song.mp3

# With MIDI export
python app.py input/my_song.mp3 --export-midi

# With multitrack MIDI (one track per instrument)
python app.py input/my_song.mp3 --export-multitrack-midi

# Full pipeline with LLM analysis
python app.py input/my_song.mp3 --ollama --export-midi --export-multitrack-midi
```

### 🎤 Bootleg / Live Recording (concert recording, audience capture)

Live recordings have bleed between instruments. Use `--bootleg` to run noise reduction + double-pass isolation on drums and guitar.

```bash
# Bootleg mode: enhance stems + deep isolate drums/guitar
python app.py input/live_show.mp3 --bootleg

# Bootleg + MIDI + LLM report
python app.py input/live_show.mp3 --bootleg --export-multitrack-midi --ollama

# Compare original vs enhanced vs deep-isolated stems
python tools/compare_stems.py separated/htdemucs_6s/live_show
```

### 📱 Phone Recording (voice memo, rehearsal capture)

Phone recordings are similar to bootlegs but often mono with more background noise. Same approach applies.

```bash
# Bootleg mode works well for phone recordings too
python app.py input/rehearsal.mp3 --bootleg

# If you want aggressive frequency filtering on top
python app.py input/rehearsal.mp3 --bootleg --bandpass
```

### 🧠 LLM Analysis Only (already have stems/data)

If you've already run the pipeline and just want to generate or regenerate the LLM report:

```bash
# Run Ollama on existing data (no re-separation needed)
python app.py input/my_song.mp3 --ollama-only
```

This reads the existing `output/<song>/<song>_data.json` and sends it to Ollama. Useful for trying different models or re-running after tweaking the system prompt.

---

## 🎛️ Command Line Options

| Flag | Description |
|---|---|
| `input` | Path to audio file (required) |
| `--ollama` | Enable LLM analysis via Ollama (off by default) |
| `--ollama-only` | Run only Ollama analysis on existing data (skip separation) |
| `--no_gpu` | Disable GPU, use CPU only (GPU is default) |
| `--export-midi` | Export MIDI file of the transcription to the output directory |
| `--export-multitrack-midi` | Export multitrack MIDI with separate tracks per stem |
| `--enhance_stems` | Noise reduce and normalize separated stems |
| `--bandpass` | Apply per-instrument frequency filtering (use with --enhance_stems) |
| `--midi-from-enhanced` | Use enhanced stems for MIDI transcription (implies --enhance_stems) |
| `--bootleg` | Bootleg mode: enhance stems + deep isolate drums/guitar |
| `--cleanup` | Remove separated stem files after processing |

---

## 📂 Output

After a successful run, you'll find:

```
separated/htdemucs_6s/my_song/
├── bass.wav
├── bass-enhanced.wav           ← with --enhance_stems
├── drums.wav
├── drums-enhanced.wav
├── drums-deep-isolate.wav      ← with --bootleg
├── guitar.wav
├── guitar-enhanced.wav
├── guitar-deep-isolate.wav     ← with --bootleg
├── piano.wav
├── piano-enhanced.wav
├── vocals.wav
├── vocals-enhanced.wav
├── other.wav
└── other-enhanced.wav

output/my_song/
├── my_song_data.json           ← Audio metadata + transcription (always created)
├── my_song_analysis.md         ← LLM music theory report (with --ollama)
├── my_song_report.html         ← Interactive HTML report (always created)
├── my_song.mid                 ← MIDI transcription (with --export-midi)
└── my_song_multitrack.mid      ← Multitrack MIDI (with --export-multitrack-midi)
```

### HTML Report

The HTML report (`_report.html`) is a self-contained interactive page featuring:

- **Theme switcher** — Dark Neo, Minimal, Editorial
- **Layout modes:**
  - **Analysis** — LLM report prominent, audio player sidebar
  - **Player** — Audio player prominent, analysis sidebar
  - **Chords** — Full-width chord chart only (great for playing along)
  - **Compact** — Single column, everything stacked
- **Stats bar** — Tempo, Key, Instruments, Notes detected
- **Audio players** — Original mix, all stems (original/enhanced/deep-isolate), MIDI download
- **Chord progression** — Per-bar chord detection from MIDI, displayed as a grid
- **Transcription preview** — First 30 notes with pitch, timing, and duration bars
- **Print support** — Clean print layout with audio players hidden

---

## 📋 Requirements

| Dependency | Purpose |
|---|---|
| `librosa` | Audio feature extraction (tempo, chroma, key) |
| `basic-pitch` | AI-powered MIDI transcription |
| `demucs` | Neural network stem separation |
| `torch` / `torchaudio` | Deep learning backend |
| `onnxruntime` | ONNX inference backend for Basic Pitch |
| `ollama` | Python client for Ollama LLM API |
| `noisereduce` | Spectral gating noise reduction for stem enhancement |
| `soundfile` | Audio file I/O for enhanced stem export |
| `scipy` | Signal processing (bandpass filtering) |
| `httpx` | HTTP client with timeout support |
| `pretty_midi` | MIDI file creation and multitrack merging |

---

## 🧰 Tools

### Stem Comparison (`tools/compare_stems.py`)

Compares original vs enhanced stems side-by-side with quality metrics.

```bash
python tools/compare_stems.py separated/htdemucs_6s/my_song
```

Output:
```
Stem       SNR Δ (dB)  Noise Removed  Correlation   Peak Δ      Verdict
──────────────────────────────────────────────────────────────────────────
bass           +1.20          8.3%       0.9912  +0.012    ✅ Good
drums          +0.85          5.1%       0.9967  -0.003    ✅ Good
guitar         +1.50         10.2%       0.9845  +0.008    ⚠️ OK
...
```

| Metric | Meaning |
|---|---|
| SNR Δ | Signal-to-noise improvement in dB (higher = cleaner) |
| Noise Removed | % of signal energy removed (5-15% = healthy) |
| Correlation | Signal preservation (1.0 = identical to original) |
| Peak Δ | Peak level difference after normalization |
| Verdict | ✅ Good (corr ≥ 0.98), ⚠️ OK (corr ≥ 0.95), ❌ Degraded |

Automatically detects and compares all available variants:
- Original vs Enhanced
- Original vs Deep Isolate
- Enhanced vs Deep Isolate

### Deep Stem Isolation (`tools/deep_isolate.py`)

Runs a second Demucs pass on drums and guitar stems to remove bleed. Uses existing stems — does not re-run the initial separation.

```bash
# Run on existing stems
python tools/deep_isolate.py separated/htdemucs_6s/my_song

# With GPU
python tools/deep_isolate.py separated/htdemucs_6s/my_song --gpu

# With adaptive bandpass filtering
python tools/deep_isolate.py separated/htdemucs_6s/my_song --gpu --adaptive-bandpass
```

> **Note:** `--bootleg` in `app.py` runs deep isolation automatically. This standalone tool is for testing or re-running isolation independently.

---

## 🤝 Contributing

Contributions are welcome and appreciated! Here's how to get involved:

### Ways to Contribute

- 🐛 **Bug Reports** — Found something broken? Open an issue with reproduction steps
- 💡 **Feature Requests** — Ideas for new analysis types, output formats, or integrations
- 🔧 **Code Contributions** — PRs for bug fixes, new features, or improvements
- 📖 **Documentation** — Improve README, add examples, write tutorials
- 🎵 **Testing** — Try different genres/formats and report accuracy issues

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Make your changes
4. Test with a few audio files to verify nothing breaks
5. Commit with a descriptive message: `git commit -m "Add: chord progression timeline output"`
6. Push and open a Pull Request

### Ideas for Contributions

- [ ] Chord progression timeline visualization
- [ ] Batch processing of multiple files
- [ ] Web UI / API server mode
- [ ] Support for additional LLM providers (OpenAI, Anthropic)
- [ ] Improved key detection using multiple algorithms
- [ ] Genre classification
- [ ] Configurable system prompts for different analysis styles
- [ ] Docker container for easy deployment

---

## 📄 License

MIT — do whatever you want with it.

---

## 🙏 Acknowledgments

- [Meta's Demucs](https://github.com/facebookresearch/demucs) for state-of-the-art source separation
- [Spotify's Basic Pitch](https://github.com/spotify/basic-pitch) for audio-to-MIDI transcription
- [Librosa](https://librosa.org/) for audio analysis
- [Ollama](https://ollama.com/) for making local LLMs accessible
