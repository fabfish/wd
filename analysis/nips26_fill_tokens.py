"""
Fill [[TOKEN]] placeholders in the rebuttal drafts from resolved_tokens.md.

Only replaces tokens that have a real value. PENDING tokens are left alone.
Dry-run by default; pass --write to edit the files.

    python -m analysis.nips26_fill_tokens
    python -m analysis.nips26_fill_tokens --write
"""
import argparse
import re
from pathlib import Path

from analysis.nips26_lib import REBUTTAL_DIR, TABLE_DIR


TOKEN_RE = re.compile(r'\[\[([A-Z0-9-]+)\]\]')
RESOLVED_RE = re.compile(r'`\[\[([A-Z0-9-]+)\]\]`\s*=\s*(.+)$')


def load_resolved():
    path = TABLE_DIR / 'resolved_tokens.md'
    if not path.exists():
        raise SystemExit(f"missing {path}; run python -m analysis.nips26_report first")
    out = {}
    for line in path.read_text().splitlines():
        m = RESOLVED_RE.search(line)
        if not m:
            continue
        token, value = m.group(1), m.group(2).strip()
        if value and value != 'PENDING' and not value.startswith('PENDING'):
            out[token] = value
    return out


def iter_drafts():
    for path in sorted(REBUTTAL_DIR.rglob('*.md')):
        if path.name.startswith('.') or '_data' in path.parts:
            continue
        if path.name in ('PLACEHOLDERS.md', 'resolved_tokens.md'):
            continue
        yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true',
                        help='actually edit the drafts (default: dry-run)')
    args = parser.parse_args()

    resolved = load_resolved()
    print(f"{len(resolved)} resolved tokens available")
    for token, value in sorted(resolved.items()):
        print(f"  [[{token}]] = {value}")

    total_hits = 0
    for path in iter_drafts():
        text = path.read_text()
        hits = TOKEN_RE.findall(text)
        fillable = [t for t in hits if t in resolved]
        if not fillable:
            continue
        total_hits += len(fillable)
        print(f"\n{path.relative_to(REBUTTAL_DIR)}: "
              f"{len(fillable)}/{len(hits)} tokens fillable")
        if not args.write:
            continue
        new = TOKEN_RE.sub(
            lambda m: resolved.get(m.group(1), m.group(0)), text)
        if new != text:
            path.write_text(new)
            print(f"  wrote {path}")

    if not args.write:
        print(f"\nDry-run: {total_hits} replacements available. "
              f"Re-run with --write to apply.")
    else:
        print(f"\nApplied {total_hits} replacements.")


if __name__ == '__main__':
    main()
