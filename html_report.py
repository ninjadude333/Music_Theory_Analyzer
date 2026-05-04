"""
HTML Report Generator
=====================
Generates a self-contained HTML music theory analysis report
using the dark neo design system with theme/layout switchers.

Reads _data.json and _analysis.md from the output folder,
discovers stems and MIDI files, and produces a single .html file.
"""

import os
import re
import json
from datetime import datetime


STEM_ICONS = {
    "bass": "🎸", "drums": "🥁", "guitar": "🎸", "piano": "🎹",
    "vocals": "🎤", "other": "🎛️",
}

STEM_ORDER = ["bass", "drums", "guitar", "piano", "vocals", "other"]

# Chord detection
_CHORD_TEMPLATES = {
    "": [0,4,7], "m": [0,3,7], "7": [0,4,7,10], "maj7": [0,4,7,11],
    "m7": [0,3,7,10], "dim": [0,3,6], "aug": [0,4,8], "sus4": [0,5,7], "sus2": [0,2,7],
}
_NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
_NOTE_TO_PC = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11, 'B#': 0, 'Cb': 11,
}

def _build_all_chords():
    chords = {}
    for suffix, intervals in _CHORD_TEMPLATES.items():
        for root in range(12):
            pcs = set((i + root) % 12 for i in intervals)
            chords[_NOTE_NAMES[root] + suffix] = pcs
    return chords

_ALL_CHORDS = _build_all_chords()

def _note_name_to_pc(name):
    n = name.replace('\u266f', '#').replace('\u266d', 'b').rstrip('0123456789')
    return _NOTE_TO_PC.get(n, 0)

def _detect_chords(notes, tempo_bpm, time_sig=4):
    if not notes or tempo_bpm <= 0:
        return []
    bar_dur = (60.0 / tempo_bpm) * time_sig
    max_t = max(n["s"] + n["d"] for n in notes)
    bars = []
    for i in range(int(max_t / bar_dur) + 1):
        bs, be = i * bar_dur, (i + 1) * bar_dur
        w = [0.0] * 12
        for n in notes:
            ns, ne = n["s"], n["s"] + n["d"]
            overlap = max(0, min(ne, be) - max(ns, bs))
            if overlap > 0:
                w[_note_name_to_pc(n["p"])] += overlap
        active = set(j for j, v in enumerate(w) if v > 0)
        if not active:
            bars.append({"bar": i + 1, "chord": "-", "time": round(bs, 2)})
            continue
        best, best_s = "?", -999
        for name, pcs in _ALL_CHORDS.items():
            s = len(active & pcs) / len(pcs) - 0.1 * len(active - pcs)
            if s > best_s:
                best_s, best = s, name
        bars.append({"bar": i + 1, "chord": best, "time": round(bs, 2)})
    return bars


def _md_to_html(md):
    """Minimal markdown to HTML for the LLM analysis content."""
    html = md
    # Headers (process before escaping so # isn't affected)
    lines = html.split("\n")
    processed = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### "):
            processed.append(f'<h3>{stripped[4:]}</h3>')
        elif stripped.startswith("## "):
            processed.append(f'<h2>{stripped[3:]}</h2>')
        elif stripped.startswith("# "):
            processed.append(f'<h1>{stripped[2:]}</h1>')
        else:
            processed.append(line)
    html = "\n".join(processed)

    # LaTeX math blocks — convert to readable text before escaping
    html = re.sub(r'\$\$(.+?)\$\$', lambda m: '<div class="math">' + m.group(1)
        .replace(r'\text{', '').replace('}', '').replace(r'\rightarrow', ' → ')
        .replace(r'\quad', '  ').replace(r'\flat', '♭')
        .replace('|', '|') + '</div>', html, flags=re.DOTALL)
    html = re.sub(r'\$(.+?)\$', lambda m: '<code>' + m.group(1)
        .replace(r'\text{', '').replace('}', '').replace(r'\rightarrow', ' → ')
        .replace(r'\flat', '♭').replace(r'\#', '♯') + '</code>', html)

    # Escape HTML entities (after math processing)
    html = html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Restore our HTML tags
    for tag in ['h1', 'h2', 'h3', 'p', 'li', 'hr', 'strong', 'em', 'code', 'div']:
        html = html.replace(f'&lt;{tag}&gt;', f'<{tag}>').replace(f'&lt;{tag} ', f'<{tag} ')
        html = html.replace(f'&lt;/{tag}&gt;', f'</{tag}>')
    html = html.replace('class=&quot;math&quot;', 'class="math"')

    # Bold and italic
    html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'(?<!\\)\*(.+?)\*', r'<em>\1</em>', html)

    # Bullet lists
    html = re.sub(r'^\*\s+(.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^-\s+(.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

    # Numbered lists
    html = re.sub(r'^\d+\.\s+(.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

    # Horizontal rules
    html = re.sub(r'^---+$', r'<hr>', html, flags=re.MULTILINE)

    # Paragraphs (double newline)
    html = re.sub(r'\n\n+', r'</p><p>', html)
    html = f'<p>{html}</p>'

    # Clean up empty paragraphs and fix nesting
    html = html.replace('<p></p>', '')
    html = html.replace('<p><hr></p>', '<hr>')
    html = re.sub(r'<p>(<h[123]>)', r'\1', html)
    html = re.sub(r'(</h[123]>)</p>', r'\1', html)
    html = re.sub(r'<p>(<div)', r'\1', html)
    html = re.sub(r'(</div>)</p>', r'\1', html)
    return html


def _discover_audio_files(stem_dir, output_dir, filename, input_path):
    """Find all playable audio files (stems, MIDI, original)."""
    files = []

    # Original input file
    if os.path.exists(input_path):
        files.append({"label": "🎵 Original Mix", "path": os.path.abspath(input_path), "type": "audio"})

    # Stems
    if os.path.exists(stem_dir):
        for stem in STEM_ORDER:
            for suffix, tag in [("", ""), ("-enhanced", " (Enhanced)"), ("-deep-isolate", " (Deep Isolate)")]:
                p = os.path.join(stem_dir, f"{stem}{suffix}.wav")
                if os.path.exists(p):
                    icon = STEM_ICONS.get(stem, "🎵")
                    files.append({"label": f"{icon} {stem.title()}{tag}", "path": os.path.abspath(p), "type": "audio"})

    # MIDI files
    for suffix, label in [("", "🎹 MIDI"), ("_multitrack", "🎹 Multitrack MIDI")]:
        p = os.path.join(output_dir, f"{filename}{suffix}.mid")
        if os.path.exists(p):
            files.append({"label": label, "path": os.path.abspath(p), "type": "midi"})

    return files


def generate_html_report(input_path, output_dir, stem_dir, filename):
    """Generate the HTML report file."""
    json_path = os.path.join(output_dir, f"{filename}_data.json")
    md_path = os.path.join(output_dir, f"{filename}_analysis.md")

    if not os.path.exists(json_path):
        print(f"  ⚠️ No data file found, skipping HTML generation.")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    notes = data.get("transcription_preview", [])

    # Read analysis markdown if it exists
    analysis_html = "<p>No LLM analysis available. Run with <code>--ollama</code> to generate.</p>"
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        # Strip the header block (first 6 lines are metadata we already show)
        lines = md_content.split("\n")
        body_start = 0
        for i, line in enumerate(lines):
            if line.strip() == "---":
                body_start = i + 1
                break
        analysis_html = _md_to_html("\n".join(lines[body_start:]))

    # Discover audio files
    audio_files = _discover_audio_files(stem_dir, output_dir, filename, input_path)

    # Build audio player rows
    audio_rows = ""
    for af in audio_files:
        path = af["path"].replace("\\", "/")
        if af["type"] == "midi":
            audio_rows += f'''<div class="audio-row">
              <span class="audio-label">{af["label"]}</span>
              <a href="file:///{path}" class="btn ghost" download>Download</a>
            </div>\n'''
        else:
            audio_rows += f'''<div class="audio-row">
              <span class="audio-label">{af["label"]}</span>
              <audio controls preload="none"><source src="file:///{path}" type="audio/wav"></audio>
            </div>\n'''

    # Build note preview rows
    note_rows = ""
    for n in notes[:30]:
        pct = min(n["d"] / 2.0 * 100, 100)
        note_rows += f'''<div class="note-row">
          <span class="note-pitch">{n["p"]}</span>
          <span class="note-time">{n["s"]}s</span>
          <div class="bar"><span style="--val:{pct:.0f}%"></span></div>
          <span class="note-dur">{n["d"]}s</span>
        </div>\n'''

    # Build chord-per-bar grid (compute from notes if not in data)
    chords = data.get("chords_per_bar", [])
    if not chords and notes and meta.get("tempo_bpm", 0) > 0:
        chords = _detect_chords(notes, meta["tempo_bpm"])
    chord_blocks = ""
    for c in chords:
        chord_blocks += f'''<div class="chord-bar" data-chord="{c["chord"]}">
          <div class="bar-num">Bar {c["bar"]}</div>
          <div class="chord-name">{c["chord"]}</div>
          <div class="bar-time">{c["time"]}s</div>
        </div>\n'''

    # Instrument chips
    inst_chips = ""
    for inst in meta.get("instruments", []):
        icon = STEM_ICONS.get(inst, "🎵")
        inst_chips += f'<span class="chip"><span class="dot"></span>{icon} {inst.title()}</span>\n'

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    display_name = filename.replace("_", " ")

    html = f'''<!DOCTYPE html>
<html lang="en" class="layout-analysis">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>🎼 {display_name} — Music Theory Analysis</title>
  <style>
    :root {{
      --bg: #0b0c0e; --surface: #111318; --muted: #171a20; --text: #e8eaed;
      --subtle: #aab0b6; --accent: #8ab4f8; --accent-2: #c58af9;
      --ok: #34c759; --warn: #ffd60a; --danger: #ff453a;
      --chip: #20232b; --border: rgba(255,255,255,0.07);
      --shadow: 0 12px 36px rgba(0,0,0,.35); --radius: 18px;
    }}
    [data-theme="minimal"] {{
      --bg: #f7f7fb; --surface: #ffffff; --muted: #f3f4f7; --text: #0f172a;
      --subtle: #667085; --accent: #2563eb; --accent-2: #7c3aed;
      --ok: #16a34a; --warn: #f59e0b; --danger: #ef4444; --chip: #eef2ff;
      --border: #e5e7eb; --shadow: 0 10px 30px rgba(16,24,40,.06);
    }}
    [data-theme="editorial"] {{
      --bg: #fffdf8; --surface: #ffffff; --muted: #fff7e6; --text: #1f2937;
      --subtle: #6b7280; --accent: #0ea5e9; --accent-2: #ef4444;
      --ok: #16a34a; --warn: #d97706; --danger: #b91c1c; --chip: #f3f4f6;
      --border: #efe7d2; --shadow: 0 20px 50px rgba(149,121,67,.15);
    }}
    *{{box-sizing:border-box}} html,body{{height:100%}}
    body{{
      margin:0; background: radial-gradient(1200px 500px at 60% -200px, rgba(124,58,237,.08), transparent 55%), var(--bg);
      color:var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial; line-height:1.45;
    }}
    .app{{max-width:1200px;margin:0 auto;padding:28px 20px 84px}}
    .topbar{{display:grid;grid-template-columns:1fr auto auto;gap:14px;align-items:center;margin-bottom:18px}}
    .title{{font-weight:800;letter-spacing:-.02em;line-height:1.05;font-size:clamp(26px,3vw,36px)}}
    .title small{{display:block;font-weight:500;color:var(--subtle);font-size:.6em}}
    .control{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:8px;display:flex;gap:6px;box-shadow:var(--shadow)}}
    .btn{{border:1px solid transparent;background:transparent;color:var(--text);padding:8px 12px;border-radius:10px;cursor:pointer;font-weight:600;font-size:13px;text-decoration:none}}
    .btn[aria-pressed="true"],.btn.primary{{background:var(--muted);border-color:var(--border)}}
    .btn:hover{{background:var(--muted)}}
    .meta{{display:flex;flex-wrap:wrap;gap:10px;margin:6px 0 10px}}
    .chip{{background:var(--chip);color:var(--text);border:1px solid var(--border);padding:6px 10px;border-radius:999px;font-size:12px;display:inline-flex;align-items:center;gap:6px;white-space:nowrap}}
    .chip .dot{{width:8px;height:8px;border-radius:999px;background:var(--accent);display:inline-block}}
    .stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0 18px}}
    .stat{{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:16px;box-shadow:var(--shadow);display:grid;gap:6px}}
    .stat .k{{font-size:28px;font-weight:800;letter-spacing:-.02em}}.stat .l{{font-size:12px;color:var(--subtle);text-transform:uppercase;letter-spacing:.14em}}
    .columns{{display:grid;gap:16px;grid-template-columns:3fr 2fr}}
    .layout-compact .columns{{grid-template-columns:1fr}}
    .layout-player .columns{{grid-template-columns:2fr 3fr}}
    .layout-chords .columns{{grid-template-columns:1fr}}
    .layout-chords .chord-grid{{grid-template-columns:repeat(8,1fr)}}
    .layout-chords .card:not(.chord-card){{display:none}}
    .layout-chords .chord-card{{display:block}}
    .layout-chords .stats{{display:none}}
    .layout-chords .meta{{display:none}}
    .layout-chords .chord-collapsed .chord-grid{{max-height:none}}
    .layout-chords .chord-fade{{display:none!important}}
    .full-width{{grid-column:1/-1}}
    .card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow)}}
    .card .hd{{padding:16px 18px 0;color:var(--subtle);font-weight:700;text-transform:uppercase;font-size:12px;letter-spacing:.14em}}
    .card .bd{{padding:18px}}
    .muted{{color:var(--subtle)}} .ghost{{opacity:.85}}
    .content{{line-height:1.6}}
    .content h1{{font-size:20px;font-weight:700;margin:18px 0 8px;color:var(--accent)}}
    .content h2{{font-size:18px;font-weight:700;margin:16px 0 8px;color:var(--text)}}
    .content h3{{font-size:16px;font-weight:600;margin:12px 0 6px;color:var(--text)}}
    .content p{{margin:8px 0;color:var(--text)}}
    .content li{{margin:4px 0;color:var(--text);list-style:disc;margin-left:20px}}
    .content strong{{font-weight:600;color:var(--accent)}}
    .content em{{color:var(--accent-2)}}
    .content code{{background:var(--muted);padding:2px 6px;border-radius:6px;font-size:13px;color:var(--accent-2)}}
    .content .math{{background:var(--muted);padding:12px 16px;border-radius:10px;font-size:15px;font-weight:600;color:var(--accent);margin:12px 0;text-align:center;letter-spacing:.05em}}
    .content hr{{border:none;height:1px;background:var(--border);margin:16px 0}}
    .audio-row{{display:grid;grid-template-columns:180px 1fr;gap:12px;align-items:center;padding:8px 0;border-bottom:1px solid var(--border)}}
    .audio-row:last-child{{border-bottom:none}}
    .audio-label{{font-weight:600;font-size:13px;white-space:nowrap}}
    .audio-row audio{{width:100%;height:36px;border-radius:8px}}
    .note-row{{display:grid;grid-template-columns:60px 50px 1fr 50px;gap:8px;align-items:center;padding:4px 0;font-size:13px}}
    .note-pitch{{font-weight:700;color:var(--accent)}}.note-time,.note-dur{{color:var(--subtle);font-size:12px}}
    .bar{{height:8px;background:var(--muted);border-radius:999px;position:relative;overflow:hidden}}
    .bar>span{{position:absolute;inset:0;width:var(--val,40%);background:linear-gradient(90deg,var(--accent),var(--accent-2));border-radius:999px}}
    .chord-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
    .chord-bar{{background:var(--muted);border:1px solid var(--border);border-radius:10px;padding:10px 8px;text-align:center;display:grid;gap:4px;transition:transform .1s}}
    .chord-bar:hover{{transform:scale(1.05)}}
    .chord-bar .chord-name{{font-size:18px;font-weight:800;color:var(--accent);letter-spacing:-.01em}}
    .chord-bar .bar-num{{font-size:10px;color:var(--subtle);text-transform:uppercase;letter-spacing:.1em}}
    .chord-bar .bar-time{{font-size:10px;color:var(--subtle)}}
    .chord-bar[data-chord="-"]{{opacity:.3}}
    .chord-section{{position:relative}}
    .chord-toggle{{background:var(--muted);border:1px solid var(--border);color:var(--text);padding:6px 14px;border-radius:10px;cursor:pointer;font-size:12px;font-weight:600;float:right}}
    .chord-collapsed .chord-grid{{max-height:200px;overflow:hidden}}
    .chord-collapsed .chord-fade{{display:block}}
    .chord-fade{{display:none;position:absolute;bottom:50px;left:0;right:0;height:60px;background:linear-gradient(transparent,var(--surface));pointer-events:none}}
    [data-theme="editorial"] .title{{font-family: Georgia, "Times New Roman", serif; letter-spacing:-.01em}}
    [data-theme="editorial"] .card{{border-radius:22px}}
    @media (max-width:768px){{.columns{{grid-template-columns:1fr}}.stats{{grid-template-columns:repeat(2,1fr)}}}}
    @media print{{.control,.topbar .btn{{display:none!important}}.app{{max-width:auto;padding:0}}body{{background:#fff}}audio{{display:none}}}}
  </style>
</head>
<body>
  <div class="app">
    <div class="topbar">
      <div class="title">
        🎼 {display_name}
        <small>Music Theory Analysis • Generated: {now}</small>
      </div>
      <div class="control" role="group" aria-label="Theme">
        <button class="btn" data-theme="minimal">Minimal</button>
        <button class="btn" data-theme="dark" aria-pressed="true">Dark Neo</button>
        <button class="btn" data-theme="editorial">Editorial</button>
      </div>
      <div class="control" role="group" aria-label="Layout">
        <button class="btn" data-layout="analysis" aria-pressed="true">Analysis</button>
        <button class="btn" data-layout="player">Player</button>
        <button class="btn" data-layout="chords">Chords</button>
        <button class="btn" data-layout="compact">Compact</button>
      </div>
    </div>

    <div class="meta">
      {inst_chips}
      <button class="btn ghost" style="margin-left:auto" onclick="window.print()">Print</button>
    </div>

    <div class="stats">
      <div class="stat"><div class="l">Tempo</div><div class="k">{meta["tempo_bpm"]} BPM</div><div class="muted">Detected by Librosa</div></div>
      <div class="stat"><div class="l">Key</div><div class="k">{meta["estimated_key"]}</div><div class="muted">Estimated from chroma</div></div>
      <div class="stat"><div class="l">Instruments</div><div class="k">{len(meta.get("instruments", []))}</div><div class="muted">Separated by Demucs</div></div>
      <div class="stat"><div class="l">Notes Detected</div><div class="k">{len(notes)}</div><div class="muted">Transcribed by Basic Pitch</div></div>
    </div>

    <div class="columns">
      <div>
        <section class="card">
          <div class="hd">Music Theory Analysis</div>
          <div class="bd content">{analysis_html}</div>
        </section>

        <section class="card chord-card" style="margin-top:16px">
          <div class="hd">Chord Progression (Per Bar)
            {f'<button class="chord-toggle" onclick="toggleChords()">Show All ({len(chords)} bars)</button>' if len(chords) > 16 else ''}
          </div>
          </div>
          <div class="bd chord-section{' chord-collapsed' if len(chords) > 16 else ''}" id="chord-section">
            {f'<div class="chord-grid">{chord_blocks}</div><div class="chord-fade"></div>' if chord_blocks else '<p class="muted">No chord data available. Requires MIDI transcription.</p>'}
          </div>
        </section>

        <section class="card" style="margin-top:16px">
          <div class="hd">Transcription Preview</div>
          <div class="bd">
            {note_rows if note_rows else '<p class="muted">No transcription data available.</p>'}
          </div>
        </section>
      </div>

      <div>
        <section class="card">
          <div class="hd">🎧 Audio Player</div>
          <div class="bd">
            {audio_rows if audio_rows else '<p class="muted">No audio files found.</p>'}
          </div>
        </section>
      </div>
    </div>
  </div>

  <script>
    document.querySelectorAll('[data-theme]').forEach(btn =>
      btn.addEventListener('click', () => {{
        const theme = btn.dataset.theme;
        if (theme === 'dark') document.documentElement.removeAttribute('data-theme');
        else document.documentElement.setAttribute('data-theme', theme);
        document.querySelectorAll('[data-theme]').forEach(b => b.setAttribute('aria-pressed', b===btn));
      }})
    );
    document.querySelectorAll('[data-layout]').forEach(btn =>
      btn.addEventListener('click', () => {{
        document.documentElement.classList.remove('layout-analysis','layout-player','layout-compact','layout-chords');
        document.documentElement.classList.add('layout-' + btn.dataset.layout);
        document.querySelectorAll('[data-layout]').forEach(b => b.setAttribute('aria-pressed', b===btn));
      }})
    );
    function toggleChords() {{
      const sec = document.getElementById('chord-section');
      const btn = document.querySelector('.chord-toggle');
      sec.classList.toggle('chord-collapsed');
      btn.textContent = sec.classList.contains('chord-collapsed') ? 'Show All ({len(chords)} bars)' : 'Collapse';
    }}
  </script>
</body>
</html>'''

    html_path = os.path.join(output_dir, f"{filename}_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path
