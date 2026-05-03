# 🎼 Music Theory Analyzer

An AI-powered music analysis pipeline that separates instruments, transcribes audio, and generates detailed music theory reports using a local LLM.

Drop in any audio file and get back:
- 🥁 **Stem separation** (bass, drums, vocals, other) via [Demucs](https://github.com/facebookresearch/demucs)
- 🎹 **MIDI transcription** via [Basic Pitch](https://github.com/spotify/basic-pitch)
- 🎵 **Audio feature extraction** (tempo, key detection) via [Librosa](https://librosa.org/)
- 🧠 **Expert music theory analysis** (key/mode, Roman numeral analysis, song form, soloing advice) via [Ollama](https://ollama.com/) LLM

Reports are saved as Markdown files alongside the separated stems for easy reference.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌───────────────┐
│  Audio File │────▶│   Demucs     │────▶│  Librosa +      │────▶│  Ollama LLM   │
│  (.mp3/wav) │     │  (Stems)     │     │  Basic Pitch    │     │  (Analysis)   │
└─────────────┘     └──────────────┘     └─────────────────┘     └───────────────┘
                           │                      │                        │
                           ▼                      ▼                        ▼
                    separated/htdemucs/    output/<song>/           output/<song>/
                    <song>/bass.wav        <song>.mid               <song>_analysis.md
                    <song>/drums.wav
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

### Usage

```bash
# Full analysis (separation + transcription + LLM report)
python app.py input/my_song.mp3

# With GPU acceleration for stem separation
python app.py input/my_song.mp3 --gpu_on

# Skip LLM analysis (just extract audio data as JSON)
python app.py input/my_song.mp3 --no-ollama

# Export MIDI transcription alongside stems
python app.py input/my_song.mp3 --export-midi

# Export multitrack MIDI (one track per instrument)
python app.py input/my_song.mp3 --export-multitrack-midi

# Clean up separated stems after processing
python app.py input/my_song.mp3 --cleanup
```

### Output

After a successful run, you'll find:

```
separated/htdemucs/my_song/
├── bass.wav
├── drums.wav
├── vocals.wav
└── other.wav

output/my_song/
├── my_song_analysis.md         ← The music theory report
├── my_song.mid                 ← MIDI transcription (with --export-midi)
└── my_song_multitrack.mid      ← Multitrack MIDI (with --export-multitrack-midi)
```

---

## 📋 Requirements

| Dependency | Purpose |
|---|---|
| `librosa` | Audio feature extraction (tempo, chroma, key) |
| `basic-pitch` | AI-powered MIDI transcription |
| `demucs` | Neural network stem separation |
| `torch` / `torchaudio` | Deep learning backend |
| `tensorflow` | Required by Basic Pitch |
| `ollama` | Python client for Ollama LLM API |
| `httpx` | HTTP client with timeout support |
| `pretty_midi` | MIDI file creation and multitrack merging |

---

## 🎛️ Command Line Options

| Flag | Description |
|---|---|
| `input` | Path to audio file (required) |
| `--gpu_on` | Use CUDA GPU for Demucs stem separation |
| `--no-ollama` | Skip LLM analysis, output raw JSON data only |
| `--export-midi` | Export MIDI file of the transcription to the output directory |
| `--export-multitrack-midi` | Export multitrack MIDI with separate tracks per stem |
| `--cleanup` | Remove separated stem files after processing |

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

- [ ] Support for additional output formats (PDF, HTML)
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
