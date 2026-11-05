"""Local audio inference. Heavy dependencies are imported only when invoked."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .core import Analysis, Flag, bar_grid, render_lrc, words_from_segments

EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aiff", ".aif", ".ogg", ".opus"}


@dataclass(frozen=True)
class Options:
    model: str = "large-v3"
    language: str = "en"
    separate: bool = True
    compare_mix: bool = True
    beats_per_bar: int = 4
    bpm: float | None = None
    offset: float | None = None
    min_words: int = 8
    overwrite: bool = False
    dry_run: bool = False
    transcript_json: Path | None = None
    suffix: str = ""
    output_dir: Path | None = None


def decode(path: Path, dest: Path) -> None:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-vn",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("FFmpeg could not decode audio: " + result.stderr.strip()[-400:])


def duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def separate_vocals(path: Path, workspace: Path) -> Path:
    print("  Separating vocals (Demucs)...", flush=True)
    command = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems=vocals",
        "--name",
        "htdemucs",
        "--out",
        str(workspace / "stems"),
        str(path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    output = workspace / "stems" / "htdemucs" / path.stem / "vocals.wav"
    if not output.exists():
        raise RuntimeError("Demucs did not create the expected vocal stem")
    return output


def transcribe(path: Path, *, model: str, language: str) -> list[dict]:
    try:
        import mlx_whisper
    except ImportError as exc:
        raise RuntimeError(
            "MLX Whisper is not installed. Run 'uv sync --locked --extra mac' on Apple Silicon."
        ) from exc
    known = {
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        "medium": "mlx-community/whisper-medium-mlx",
    }
    if model not in known:
        raise ValueError(f"unknown model {model!r}; choose {', '.join(known)}")
    print(f"  Transcribing {path.name} with {model} (MLX)...", flush=True)
    result = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=known[model],
        language=language,
        word_timestamps=True,
        condition_on_previous_text=False,
        hallucination_silence_threshold=1.5,
        no_speech_threshold=0.80,
        logprob_threshold=-0.85,
        compression_ratio_threshold=2.4,
        temperature=0.0,
        verbose=False,
    )
    return result["segments"]


def old_json_segments(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if "lines" in data:
        return data["lines"]
    if "segments" in data:
        return data["segments"]
    raise ValueError("JSON must have 'lines' (audio-lyrics) or 'segments' (Whisper)")


def compare_transcripts(stem: Analysis, mix: Analysis) -> Analysis:
    """Review mix disagreements and suppress uncorroborated repeating loops.

    Repetition is normal in choruses. Don't remove a repeated phrase if both
    independent transcriptions recognize it at the same time.
    """
    from difflib import SequenceMatcher

    from .core import Word, normalize

    accepted: list[Word] = []
    warnings = list(stem.flags) + list(mix.flags)
    repeated = [f for f in stem.flags if "highly repetitive" in f.reason]

    blocks = sorted(set(int(word.start // 4) for word in stem.words + mix.words))
    for block in blocks:
        start, end = 4.0 * block, 4.0 * (block + 1)
        left = [w for w in stem.words if start <= w.start < end]
        right = [w for w in mix.words if start <= w.start < end]
        suspect_repeat = any(f.start < end and start < f.end for f in repeated)
        if not left:
            if right:
                # Demucs can remove intelligible sung/rap vocals alongside
                # accompaniment. Rescue strongly recognized mix words, unless
                # that pass is itself stuck in a repetition loop.
                repeated_mix = any(
                    "highly repetitive" in f.reason and f.start < end and start < f.end
                    for f in mix.flags
                )
                if not repeated_mix and sum(w.probability for w in right) / len(right) >= 0.55:
                    accepted.extend(right)
                warnings.append(Flag(start, end, "words only in original mix; review"))
            continue
        if not right:
            if suspect_repeat:
                warnings.append(Flag(start, end, "unconfirmed repeated phrase omitted"))
            else:
                accepted.extend(left)
            continue

        a = " ".join(normalize(w.text) for w in left)
        b = " ".join(normalize(w.text) for w in right)
        similarity = SequenceMatcher(None, a, b, autojunk=False).ratio()
        if suspect_repeat:
            if similarity >= 0.56:
                accepted.extend(left)
            else:
                warnings.append(Flag(start, end, "uncorroborated repeated phrase omitted"))
        elif similarity < 0.36:
            warnings.append(
                Flag(start, end, f"isolated vocals and original mix disagree ({similarity:.0%})")
            )
            if sum(w.probability for w in left) / len(left) >= 0.55:
                accepted.extend(left)
        else:
            accepted.extend(left)
    return Analysis(accepted, warnings, stem.total_words + mix.total_words)


def process_audio(path: Path, options: Options) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    output_base = options.output_dir or path.parent
    output = output_base / f"{path.stem}{options.suffix}.lrc"
    if output.exists() and not options.overwrite and not options.dry_run:
        return {"status": "exists", "file": str(path), "output": str(output)}

    with tempfile.TemporaryDirectory(prefix="beatlrc-") as tmp:
        workspace = Path(tmp)
        decoded = workspace / "input.wav"
        decode(path, decoded)
        length = duration(decoded)

        if options.bpm is None:
            from beat_this.inference import File2Beats

            print("  Detecting beats and downbeats...", flush=True)
            tracker = File2Beats(checkpoint_path="final0", device="cpu")
            beats, downbeats = tracker(str(decoded))
        else:
            beats, downbeats = [], []
        bars, estimated_bpm, agreement = bar_grid(
            beats,
            downbeats,
            beats_per_bar=options.beats_per_bar,
            bpm=options.bpm,
            offset=options.offset,
            duration=length,
        )
        if options.bpm is None:
            print(
                f"  {estimated_bpm:.1f} BPM; {len(bars)} bars; "
                f"{agreement:.0%} downbeat-phase agreement"
            )
        else:
            print(f"  {estimated_bpm:.1f} BPM; {len(bars)} bars (manual grid)")

        if options.transcript_json:
            segments = old_json_segments(options.transcript_json)
            analysis = words_from_segments(segments, length)
        else:
            vocal_audio = separate_vocals(decoded, workspace) if options.separate else decoded
            segments = transcribe(vocal_audio, model=options.model, language=options.language)
            analysis = words_from_segments(
                segments, length, keep_repetitions=options.separate and options.compare_mix
            )
            if options.separate and options.compare_mix:
                mix_segments = transcribe(decoded, model=options.model, language=options.language)
                analysis = compare_transcripts(
                    analysis,
                    words_from_segments(mix_segments, length, keep_repetitions=True),
                )

        if len(analysis.words) < options.min_words:
            return {
                "status": "no-reliable-lyrics",
                "file": str(path),
                "bpm": estimated_bpm,
                "accepted_words": len(analysis.words),
                "flags": [f.__dict__ for f in analysis.flags],
            }

        content = render_lrc(analysis.words, bars, path.stem)
        report = {
            "file": str(path),
            "status": "preview" if options.dry_run else "generated",
            "bpm": round(estimated_bpm, 2),
            "beats_per_bar": options.beats_per_bar,
            "bar_count": len(bars),
            "downbeat_agreement": round(agreement, 3) if options.bpm is None else None,
            "accepted_words": len(analysis.words),
            "transcribed_words": analysis.total_words,
            "flags": [f.__dict__ for f in analysis.flags],
            "output": str(output),
        }
        if not options.dry_run:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and options.overwrite:
                backup = output.with_name(output.stem + ".before-beatlrc.lrc")
                if backup.exists():
                    raise FileExistsError(f"refusing to replace existing backup: {backup}")
                shutil.copy2(output, backup)
            output.write_text(content)
            if analysis.flags:
                output.with_suffix(".beatlrc-review.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
        return report
