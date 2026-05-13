"""
visualise_pairs.py
Generates a browsable HTML gallery from image-text paired JSON files.

Supports switching between:
  - Simple view: image + text only
  - Enriched view: image + text + all metadata fields

Usage:
    python3 visualise_pairs.py \
        --pairs outputs/pairs/image_text_pairs.json \
        --enriched outputs/pairs/image_text_enriched.json \
        --image_base predictions/lora \
        --output pairs/output/gallery.html

    # Per-book
    python3 visualise_pairs.py \
        --pairs outputs/pairs/bdj_qm/bdj_qm_image_text_pairs.json \
        --enriched outputs/pairs/bdj_qm/bdj_qm_image_text_enriched.json \
        --image_base predictions/lora \
        --output outputs/pairs/bdj_qm/gallery.html
"""

import argparse
import json
import base64
from pathlib import Path


def encode_image(image_path: Path) -> str | None:
    """Encode image to base64 data URI for self-contained HTML."""
    if not image_path.exists():
        return None
    suffix = image_path.suffix.lower()
    mime = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.bmp': 'image/bmp'}.get(suffix, 'image/jpeg')
    data = base64.b64encode(image_path.read_bytes()).decode('utf-8')
    return f"data:{mime};base64,{data}"


def build_html(pairs: list, enriched: list, image_base: Path, title: str) -> str:
    """Build self-contained HTML gallery."""

    # Build card data
    cards = []
    enriched_by_path = {r['image_path']: r for r in enriched} if enriched else {}

    for item in pairs:
        img_rel  = item.get('image_path', '')
        text     = item.get('text', '')

        # Try to find image
        img_src  = None
        for candidate in [
            image_base / img_rel,
            Path(img_rel),
        ]:
            uri = encode_image(candidate)
            if uri:
                img_src = uri
                break

        # Enriched fields
        enrich = enriched_by_path.get(img_rel, {})
        meta_fields = {k: v for k, v in enrich.items()
                       if k not in ('image_path', 'text')}

        cards.append({
            'img_src':  img_src,
            'img_path': img_rel,
            'text':     text,
            'meta':     meta_fields,
        })

    # Serialize cards to JS
    cards_js = json.dumps(cards, ensure_ascii=False)
    total    = len(cards)
    matched  = sum(1 for c in cards if c['img_src'])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg: #0f0f0f;
    --surface: #1a1a1a;
    --border: #2e2e2e;
    --accent: #c8a96e;
    --text: #e8e0d0;
    --muted: #888;
    --meta-bg: #141414;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Georgia', serif;
    min-height: 100vh;
  }}

  header {{
    padding: 2rem 2.5rem 1rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 1.5rem;
    flex-wrap: wrap;
  }}

  header h1 {{
    font-size: 1.4rem;
    font-weight: normal;
    letter-spacing: 0.05em;
    color: var(--accent);
  }}

  header .stats {{
    font-size: 0.8rem;
    color: var(--muted);
    font-family: monospace;
  }}

  .controls {{
    padding: 1rem 2.5rem;
    display: flex;
    gap: 1rem;
    align-items: center;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--border);
  }}

  .toggle-group {{
    display: flex;
    gap: 0;
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }}

  .toggle-group button {{
    background: var(--surface);
    color: var(--muted);
    border: none;
    padding: 0.4rem 1rem;
    font-size: 0.8rem;
    cursor: pointer;
    font-family: monospace;
    transition: all 0.15s;
  }}

  .toggle-group button.active {{
    background: var(--accent);
    color: #000;
  }}

  .search-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.4rem 0.8rem;
    font-size: 0.85rem;
    border-radius: 4px;
    width: 260px;
    font-family: inherit;
  }}

  .search-box:focus {{ outline: none; border-color: var(--accent); }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 1.5rem;
    padding: 2rem 2.5rem;
  }}

  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: border-color 0.2s;
  }}

  .card:hover {{ border-color: var(--accent); }}

  .card-img {{
    width: 100%;
    aspect-ratio: 3/4;
    object-fit: contain;
    background: #111;
    padding: 0.5rem;
  }}

  .card-img.missing {{
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: 0.75rem;
    font-family: monospace;
    min-height: 160px;
  }}

  .card-body {{
    padding: 0.75rem;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }}

  .card-text {{
    font-size: 1.05rem;
    line-height: 1.6;
    color: var(--text);
    word-break: break-all;
  }}

  .card-path {{
    font-size: 0.65rem;
    color: var(--muted);
    font-family: monospace;
    word-break: break-all;
  }}

  .meta-table {{
    display: none;
    margin-top: 0.5rem;
    background: var(--meta-bg);
    border-radius: 4px;
    overflow: hidden;
    font-size: 0.72rem;
    font-family: monospace;
  }}

  .meta-table.visible {{ display: table; width: 100%; border-collapse: collapse; }}

  .meta-table td {{
    padding: 0.25rem 0.5rem;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }}

  .meta-table td:first-child {{
    color: var(--accent);
    white-space: nowrap;
    width: 40%;
  }}

  .meta-table td:last-child {{ color: var(--muted); }}

  .no-results {{
    grid-column: 1 / -1;
    text-align: center;
    padding: 4rem;
    color: var(--muted);
    font-family: monospace;
  }}
</style>
</head>
<body>

<header>
  <h1>MILens — Paired Gallery</h1>
  <span class="stats">{total} pairs · {matched} with images · {total - matched} missing</span>
</header>

<div class="controls">
  <div class="toggle-group">
    <button class="active" onclick="setView('simple')" id="btn-simple">Simple</button>
    <button onclick="setView('enriched')" id="btn-enriched">Enriched</button>
  </div>
  <input class="search-box" type="text" placeholder="Search text or filename…"
         oninput="filterCards(this.value)" />
  <span class="stats" id="count-label">{total} shown</span>
</div>

<div class="grid" id="grid"></div>

<script>
const CARDS = {cards_js};
let currentView = 'simple';

function setView(view) {{
  currentView = view;
  document.getElementById('btn-simple').classList.toggle('active', view === 'simple');
  document.getElementById('btn-enriched').classList.toggle('active', view === 'enriched');
  renderCards(document.querySelector('.search-box').value);
}}

function filterCards(query) {{
  renderCards(query);
}}

function renderCards(query) {{
  const grid = document.getElementById('grid');
  const q = query.toLowerCase().trim();
  const filtered = q
    ? CARDS.filter(c => c.text.toLowerCase().includes(q) || c.img_path.toLowerCase().includes(q))
    : CARDS;

  document.getElementById('count-label').textContent = filtered.length + ' shown';

  if (filtered.length === 0) {{
    grid.innerHTML = '<div class="no-results">No results found.</div>';
    return;
  }}

  grid.innerHTML = filtered.map(card => {{
    const imgHtml = card.img_src
      ? `<img class="card-img" src="${{card.img_src}}" alt="${{card.img_path}}" loading="lazy">`
      : `<div class="card-img missing">image not found<br>${{card.img_path.split('/').pop()}}</div>`;

    const metaRows = Object.entries(card.meta)
      .filter(([k]) => k !== 'source_csv' || currentView === 'enriched')
      .map(([k, v]) => `<tr><td>${{k}}</td><td>${{v || '—'}}</td></tr>`)
      .join('');

    const metaTable = metaRows
      ? `<table class="meta-table ${{currentView === 'enriched' ? 'visible' : ''}}">${{metaRows}}</table>`
      : '';

    return `
      <div class="card" data-text="${{card.text}}" data-path="${{card.img_path}}">
        ${{imgHtml}}
        <div class="card-body">
          <div class="card-text">${{card.text || '—'}}</div>
          <div class="card-path">${{card.img_path.split('/').pop()}}</div>
          ${{metaTable}}
        </div>
      </div>`;
  }}).join('');
}}

renderCards('');
</script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs",      required=True,
                        help="Path to image_text_pairs.json")
    parser.add_argument("--enriched",   default=None,
                        help="Path to image_text_enriched.json (optional, enables enriched view)")
    parser.add_argument("--image_base", default=".",
                        help="Base folder for resolving relative image paths (default: .)")
    parser.add_argument("--output",     default="pairs/output/gallery.html",
                        help="Output HTML file (default: outputs/pairs/gallery.html)")
    parser.add_argument("--title",      default="MILens — Paired Gallery")
    args = parser.parse_args()

    pairs_path   = Path(args.pairs)
    enriched_path = Path(args.enriched) if args.enriched else None
    image_base   = Path(args.image_base)
    output_path  = Path(args.output)

    print(f"Loading pairs from {pairs_path}...")
    pairs = json.loads(pairs_path.read_text(encoding='utf-8'))
    print(f"  {len(pairs)} pairs loaded")

    enriched = []
    if enriched_path and enriched_path.exists():
        enriched = json.loads(enriched_path.read_text(encoding='utf-8'))
        print(f"  {len(enriched)} enriched records loaded")

    print("Building gallery...")
    html = build_html(pairs, enriched, image_base, args.title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✅ Gallery saved to {output_path} ({size_mb:.1f} MB)")
    print(f"   Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
