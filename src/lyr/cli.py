"""One-off local batch CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import EXTENSIONS, Options, process_audio


def unique_audio_sources(files: list[Path]) -> tuple[list[Path], list[tuple[Path, Path]]]:
    """Only one audio format can own a basename-matched .lrc sidecar."""
    priority = {
        ".wav": 0,
        ".flac": 1,
        ".aiff": 2,
        ".aif": 3,
        ".mp3": 4,
        ".m4a": 5,
        ".ogg": 6,
        ".opus": 7,
    }
    chosen: dict[tuple[Path, str], Path] = {}
    duplicates: list[tuple[Path, Path]] = []
    for file in sorted(files, key=lambda p: (priority.get(p.suffix.lower(), 99), str(p))):
        key = (file.parent, file.stem.casefold())
        if key in chosen:
            duplicates.append((file, chosen[key]))
        else:
            chosen[key] = file
    return sorted(chosen.values()), duplicates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lyr",
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
    parser.add_argument(
        "--bar-division",
        choices=["auto", "1", "2"],
        default="auto",
        help="bars per detected bar: auto, 1 (half-time), or 2 (double-time)",
    )
    parser.add_argument(
        "--boundary-lead-ms",
        type=int,
        default=150,
        help="assign words this many milliseconds early to the following bar",
    )
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
    if not 0 <= args.boundary_lead_ms <= 400:
        parser.error("--boundary-lead-ms must be between 0 and 400")

    root = args.path.expanduser()
    if root.is_file():
        files = [root]
    elif root.is_dir():
        candidates = root.rglob("*") if args.recursive else root.iterdir()
        files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        parser.error(f"path does not exist: {root}")
    files, duplicates = unique_audio_sources(files)
    for skipped, selected in duplicates:
        print(f"Skipping alternate format: {skipped} (using {selected.name})", file=sys.stderr)
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
        bar_division=args.bar_division,
        boundary_lead_ms=args.boundary_lead_ms,
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
            action = "would write" if status == "preview" else status
            print(" ", action, result.get("output", ""), flush=True)
            if result.get("flags"):
                flags = result["flags"]
                print(f"  {len(flags)} uncertain regions")
                if args.dry_run or status == "no-reliable-lyrics":
                    for flag in flags[:12]:
                        print(f"    {flag['start']:.2f}-{flag['end']:.2f}: {flag['reason']}")
                    if len(flags) > 12:
                        print(f"    ... and {len(flags) - 12} more")
                else:
                    print("  Review details in the .lyr-review.json report")
        except Exception as exc:
            outcomes["failed"] += 1
            print(f"  ERROR: {exc}", file=sys.stderr, flush=True)
    print("Summary:", ", ".join(f"{k}={v}" for k, v in outcomes.items()))
    return 1 if outcomes["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
