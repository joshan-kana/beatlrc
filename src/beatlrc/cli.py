"""One-off local batch CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import EXTENSIONS, Options, process_audio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="beatlrc",
        description="Generate synced lyrics with a per-song beat grid and quality checks.",
    )
    parser.add_argument("path", type=Path, help="an audio file or directory")
    parser.add_argument("--recursive", action="store_true", help="walk subdirectories")
    parser.add_argument(
        "--model", default="large-v3", choices=["large-v3", "large-v3-turbo", "medium"]
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--no-separate", action="store_true", help="transcribe the mix directly")
    parser.add_argument(
        "--no-compare-mix", action="store_true", help="skip second transcription pass"
    )
    parser.add_argument("--beats-per-bar", type=int, default=4)
    parser.add_argument("--bpm", type=float, help="manual constant BPM override")
    parser.add_argument("--offset", type=float, help="manual first downbeat offset in seconds")
    parser.add_argument(
        "--min-words", type=int, default=8, help="minimum reliable words for output"
    )
    parser.add_argument("--dry-run", action="store_true", help="analyze without writing lyrics")
    parser.add_argument(
        "--overwrite", action="store_true", help="back up and replace existing .lrc"
    )
    parser.add_argument(
        "--transcript-json", type=Path, help="import existing transcript for diagnostics"
    )
    parser.add_argument("--suffix", default="", help="suffix before .lrc for side-by-side tests")
    parser.add_argument("--output-dir", type=Path, help="write to separate output directory")
    args = parser.parse_args(argv)

    root = args.path.expanduser()
    if root.is_file():
        files = [root]
    elif root.is_dir():
        candidates = root.rglob("*") if args.recursive else root.iterdir()
        files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        parser.error(f"path does not exist: {root}")
    if args.transcript_json and len(files) != 1:
        parser.error("--transcript-json requires exactly one audio file")
    if args.output_dir and len(files) != 1:
        parser.error("--output-dir accepts one audio file to avoid filename collisions")

    options = Options(
        model=args.model,
        language=args.language,
        separate=not args.no_separate,
        compare_mix=not args.no_compare_mix,
        beats_per_bar=args.beats_per_bar,
        bpm=args.bpm,
        offset=args.offset,
        min_words=args.min_words,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        transcript_json=args.transcript_json,
        suffix=args.suffix,
        output_dir=args.output_dir,
    )

    outcomes = {"generated": 0, "preview": 0, "exists": 0, "no-reliable-lyrics": 0, "failed": 0}
    for index, file in enumerate(files, 1):
        print(f"[{index}/{len(files)}] {file}", flush=True)
        try:
            result = process_audio(file, options)
            status = result["status"]
            outcomes[status] += 1
            print(" ", status, result.get("output", ""), flush=True)
            if result.get("flags"):
                print(f"  {len(result['flags'])} uncertain regions; inspect review report")
        except Exception as exc:
            outcomes["failed"] += 1
            print(f"  ERROR: {exc}", file=sys.stderr, flush=True)
    print("Summary:", ", ".join(f"{k}={v}" for k, v in outcomes.items()))
    return 1 if outcomes["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
