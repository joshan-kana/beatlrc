"""Beat-grid reconstruction, transcription quality control, and LRC formatting."""

from __future__ import annotations

import math
import re
import statistics
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass(frozen=True)
class Flag:
    start: float
    end: float
    reason: str


@dataclass(frozen=True)
class Analysis:
    words: list[Word]
    flags: list[Flag]
    total_words: int


def normalize(token: str) -> str:
    return re.sub(r"[^\w]+", "", token, flags=re.UNICODE).casefold()


def _valid_time(t: object) -> bool:
    return isinstance(t, (int, float)) and math.isfinite(t) and t >= 0


def words_from_segments(
    segments: Iterable[dict], duration: float, *, keep_repetitions: bool = False
) -> Analysis:
    """Reject weak/no-speech segments, implausible word timing, and local loops.

    These are conservative heuristics, not a guarantee that ASR words are correct.
    """
    candidates: list[Word] = []
    flags: list[Flag] = []
    total = 0

    for segment in segments:
        raw = segment.get("words") or []
        total += len(raw)
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))
        confidence = segment.get("confidence")
        logprob = segment.get("avg_logprob")
        no_speech = segment.get("no_speech_prob", 0)

        reasons = []
        if confidence is not None and confidence < 0.48:
            reasons.append("low segment confidence")
        if logprob is not None and logprob < -0.90:
            reasons.append("low model log probability")
        if no_speech is not None and no_speech > 0.70 and (logprob is None or logprob < -0.70):
            reasons.append("likely non-speech")
        repetitive_output = segment.get("compression_ratio", 0) > 2.5
        if repetitive_output:
            flags.append(Flag(start, end, "repetitive model output; verify by listening"))

        if reasons:
            flags.append(Flag(start, end, ", ".join(reasons)))
            continue
        if repetitive_output and not keep_repetitions:
            continue

        invalid_timestamps = 0
        for w in raw:
            s, e = w.get("start"), w.get("end")
            if not _valid_time(s) or not _valid_time(e) or e <= s or e > duration + 0.5:
                invalid_timestamps += 1
                continue
            if e - s > 3.0:
                flags.append(Flag(s, e, "implausible word duration"))
                continue
            p = float(w.get("probability", 1.0))
            if p < 0.25:
                flags.append(Flag(s, e, "low word confidence"))
            if p < 0.10:
                continue
            t = str(w.get("word", "")).strip()
            if normalize(t):
                candidates.append(Word(float(s), float(e), t, p))
        if invalid_timestamps:
            noun = "word" if invalid_timestamps == 1 else "words"
            flags.append(Flag(start, end, f"invalid timestamps for {invalid_timestamps} {noun}"))

    candidates.sort(key=lambda w: (w.start, w.end))
    if not candidates:
        return Analysis([], flags, total)

    # Look for implausibly many identical multiword sequences in a short span.
    # This catches Whisper's common "one phrase repeated endlessly" failure.
    # Real repeated hooks can also trigger it: flagged regions require review.
    grams: dict[tuple[str, ...], list[int]] = defaultdict(list)
    tokens = [normalize(w.text) for w in candidates]
    for i in range(len(tokens) - 3):
        gram = tuple(tokens[i : i + 4])
        if all(gram):
            grams[gram].append(i)

    rejected: set[int] = set()
    for locations in grams.values():
        for k, i in enumerate(locations):
            window = [j for j in locations[k:] if candidates[j].start - candidates[i].start <= 30]
            if len(window) >= 5:
                for j in window:
                    rejected.update(range(j, j + 4))
    if rejected:
        # Flag the entire surrounding phrase. Partial filtering leaves scraps
        # of the same invented sentence behind, which is equally misleading.
        region_start = max(0.0, min(candidates[i].start for i in rejected) - 8.0)
        region_end = max(candidates[i].end for i in rejected) + 2.0
        flags.append(
            Flag(region_start, region_end, "highly repetitive transcript; verify by listening")
        )
        if not keep_repetitions:
            rejected.update(
                i for i, w in enumerate(candidates) if region_start <= w.start <= region_end
            )
        else:
            rejected.clear()
    kept = [w for i, w in enumerate(candidates) if i not in rejected]
    return Analysis(kept, flags, total)


def _repair_beats(beat_times: Iterable[float]) -> tuple[list[float], float]:
    times = sorted(float(t) for t in beat_times if _valid_time(t))
    if len(times) < 12:
        raise ValueError("too few detected beats")
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a + 0.12]
    if not gaps:
        raise ValueError("no usable beat intervals")
    # Robustly estimate quarter-note intervals, ignoring short subdivisions and skips.
    candidate_gaps = [g for g in gaps if 0.30 <= g <= 1.5]
    if len(candidate_gaps) < 8:
        raise ValueError("beat intervals are too irregular")
    pulse = statistics.median(candidate_gaps)
    if not 0.28 <= pulse <= 1.6:
        raise ValueError("detected tempo out of supported range")

    clean = [times[0]]
    for t in times[1:]:
        gap = t - clean[-1]
        if gap < pulse * 0.60:
            continue
        if pulse * 1.62 < gap < pulse * 3.5:
            missing = round(gap / pulse) - 1
            step = gap / (missing + 1)
            if missing > 0 and 0.65 * pulse < step < 1.35 * pulse:
                clean.extend(clean[-1] + step * k for k in range(1, missing + 1))
        clean.append(t)
    if len(clean) < 12:
        raise ValueError("not enough reliable beats")
    return clean, pulse


def bar_grid(
    beats: Iterable[float],
    downbeats: Iterable[float],
    *,
    beats_per_bar: int = 4,
    bpm: float | None = None,
    offset: float | None = None,
    duration: float | None = None,
) -> tuple[list[float], float, float]:
    """Return stable bar starts, tempo and downbeat-phase agreement.

    Use beat progression, not raw downbeats, because model downbeats often
    include extra half-bar impulses and subdivisions.
    """
    if beats_per_bar not in (2, 3, 4, 5, 6, 7, 8):
        raise ValueError("beats_per_bar must be in [2, 8]")
    if bpm is not None:
        if not 40 <= bpm <= 240:
            raise ValueError("manual BPM must be between 40 and 240")
        if duration is None or duration <= 0:
            raise ValueError("manual BPM requires a known audio duration")
        period = 60 / bpm * beats_per_bar
        anchor = offset if offset is not None else 0.0
        count = max(0, int((duration - anchor) / period) + 1)
        return [round(anchor + i * period, 3) for i in range(count)], bpm, 1.0

    clean, pulse = _repair_beats(beats)

    ds = [float(t) for t in downbeats if _valid_time(t)]
    votes = [0] * beats_per_bar
    for downbeat in ds:
        j = min(range(len(clean)), key=lambda i: abs(clean[i] - downbeat))
        if abs(clean[j] - downbeat) < pulse * 0.28:
            votes[j % beats_per_bar] += 1
    count_votes = sum(votes)
    confidence = max(votes) / count_votes if count_votes else 0.0
    if count_votes < 3 or confidence < 0.46:
        raise ValueError(
            f"unreliable downbeat phase ({confidence:.0%} agreement); "
            "supply --bpm and --offset or review the beat map"
        )
    phase = votes.index(max(votes))
    starts = [clean[i] for i in range(phase, len(clean), beats_per_bar)]
    if offset is not None:
        shift = offset - starts[0]
        starts = [round(t + shift, 3) for t in starts if t + shift >= 0]
    if len(starts) < 2:
        raise ValueError("too few valid musical bars")

    expected = pulse * beats_per_bar
    if any(
        (b - a) < expected * 0.55 or (b - a) > expected * 1.7 for a, b in zip(starts, starts[1:])
    ):
        raise ValueError("beat grid has large discontinuities; manually inspect")
    return starts, 60.0 / pulse, confidence


def stamp(seconds: float) -> str:
    centiseconds = round(max(0, seconds) * 100)
    return f"{centiseconds // 6000:02d}:{(centiseconds // 100) % 60:02d}.{centiseconds % 100:02d}"


def render_lrc(
    words: Iterable[Word], bars: list[float], title: str, *, clear_gaps: bool = True
) -> str:
    if not bars:
        raise ValueError("no bars")
    groups: list[list[str]] = [[] for _ in bars]
    words = sorted(words, key=lambda w: w.start)
    pickup = []
    for word in words:
        # An intro/pickup before the first detected downbeat must not
        # disappear or start playing late at the following musical bar.
        if word.start + 0.05 < bars[0]:
            pickup.append(word)
            continue
        # 50 ms tolerance for slightly early word alignments at a downbeat.
        location = bisect_right(bars, word.start + 0.05) - 1
        groups[max(0, location)].append(word.text)
    lines = [f"[ti:{title}]", "[re:beatlrc]", ""]
    if pickup:
        lines.append(f"[{stamp(pickup[0].start)}]{' '.join(w.text for w in pickup)}")
    had_lyrics = bool(pickup)
    for start, tokens in zip(bars, groups):
        if tokens:
            lines.append(f"[{stamp(start)}]{' '.join(tokens)}")
            had_lyrics = True
        elif clear_gaps and had_lyrics:
            lines.append(f"[{stamp(start)}]")
            had_lyrics = False
    return "\n".join(lines) + "\n"
