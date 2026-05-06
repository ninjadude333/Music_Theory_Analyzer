# 📋 TODO — Music Theory Analyzer

> Ideas ingested from `ideas.txt` — sorted by effort (easiest first)

---

## 🟢 Trivial

- [x] **Add `--cleanup` flag to remove stems after processing**
  Add an argparse flag that deletes the separated stem directory after the report is generated. ~5 lines with `shutil.rmtree()`.

## 🟢 Easy

- [ ] **Support more input audio types: mp4, m4a, wav, flac**
  Demucs and librosa already handle wav/flac natively. For mp4/m4a, ensure ffmpeg is available (Demucs uses it internally). May just need to remove any file-extension filtering in `app.py`.

- [x] **Export MIDI file of chord progression and melody**
  Basic Pitch already produces `midi_data` — just save it with `midi_data.write()`. Minimal code change in `extract_rich_data()`.

## 🟡 Easy–Medium

- [x] **Export multitrack MIDI with separate tracks per instrument**
  Run Basic Pitch on each separated stem, merge resulting MIDI instruments into one `pretty_midi` file. Building blocks exist, needs a loop + merge.

## 🟡 Medium

- [x] **Output a formatted dynamic HTML page**
  Interactive HTML report with Dark Neo/Minimal/Editorial themes, audio players, chord chart, layout switcher, and print support.

- [ ] **Generate guitar tab output**
  Convert MIDI note data to tab format. Requires mapping MIDI pitches → string/fret positions and a text formatter. Non-trivial string assignment logic.

## 🟠 Medium–Hard

- [ ] **Dockerize the app to receive audio input and return HTML output**
  Create a Dockerfile with all dependencies (Python, ffmpeg, torch, Demucs, Basic Pitch). Expose an endpoint or volume mount that accepts audio and returns the generated HTML report. Needs careful handling of GPU passthrough and large image size.

- [ ] **Generate piano sheet music output**
  Needs a notation library (`music21` or `lilypond`). Converting MIDI → readable sheet music with proper rhythm quantization is significantly harder than tabs.

- [ ] **Detect song and get more info online**
  Integrate an audio fingerprinting API (AcoustID/MusicBrainz, Shazam) for song identification and metadata enrichment. External API dependency.

## 🔴 Hard

- [ ] **Find similar songs with the same chord progression**
  Requires a chord progression database or API. No off-the-shelf solution — likely need to build/source a dataset or use a music knowledge API.

- [ ] **Create a playlist of songs with similar chord progressions**
  Builds on chord-similarity search plus integration with a streaming service API (e.g., Spotify). Inherits all complexity from the item above.

- [ ] **Create a web UI**
  Full-stack: Flask/FastAPI backend, file upload, async job processing (Demucs is slow), progress feedback, results display. Biggest scope.
