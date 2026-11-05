# beatlrc

Offline, one-shot synchronized lyrics generation for locally stored music. Transcribe vocals on Apple Silicon, detect the musical grid for each song, and write one lyric entry per bar.

## Quick start (Apple Silicon)

Requires Nix with flakes enabled. Models are downloaded to user caches on first use. Subsequent processing can stay local after the required model files have been cached.

~~~sh
git clone https://github.com/joshan-kana/beatlrc.git
cd beatlrc
nix develop -c uv run --locked --extra mac beatlrc \
  "$HOME/Dropbox/Music Production/Songs/New" --recursive
~~~

The Nix development shell supplies Python, UV, FFmpeg, and native audio libraries. The committed uv.lock pins Python dependencies. They are installed in this project's local .venv, not globally.

When a directory contains multiple formats with the same basename, the lossless source is preferred (WAV before FLAC before lossy formats) because those files share one .lrc sidecar. Existing .lrc files are skipped unless --overwrite is explicitly requested. An overwrite preserves a separately named backup.

### Test one song without replacing its existing lyrics

~~~sh
nix develop -c uv run --locked --extra mac beatlrc \
  "/path/to/song.wav" --suffix .test
~~~

This creates song.test.lrc beside song.wav. Compare it in a player before replacing any trusted lyrics. Use --output-dir to write test outputs outside the music library.

### Preview before writing

~~~sh
nix develop -c uv run --locked --extra mac beatlrc \
  "/path/to/song.wav" --dry-run
~~~

Dry runs perform inference and report quality warnings without creating sidecars. A normal run with uncertain regions also writes a .beatlrc-review.json sidecar for review.

### Options

| Option | Effect |
| --- | --- |
| --recursive | Walk a folder and its subdirectories |
| --model large-v3 | Default full Whisper model; medium and large-v3-turbo supported |
| --no-separate | Transcribe the full mix without Demucs |
| --no-compare-mix | Skip the independent full-mix transcription |
| --beats-per-bar N | Default 4; specify another meter |
| --bpm NUMBER --offset SECONDS | Override tempo and downbeat for difficult tracks |
| --min-words N | Don't generate lyrics with fewer than N reliable words |
| --overwrite | Replace an existing .lrc after creating a backup |
| --transcript-json FILE | Diagnose an existing audio-lyrics or Whisper JSON transcript for one song |
| --suffix .test | Generate song.test.lrc for side-by-side review |
| --output-dir DIR | Write generated files outside the source folder |

## Pipeline

1. FFmpeg decodes the audio into a temporary WAV.
2. [Beat This!](https://github.com/CPJKU/beat_this) detects quarter-note beats and candidate downbeats.
3. The grid is reconstructed from beat progression and downbeat agreement. Extra half-bar downbeat predictions are ignored. BPM is estimated afresh for every song.
4. [Demucs](https://github.com/facebookresearch/demucs) separates vocals, then [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) transcribes words with timestamps.
5. The original mix is independently transcribed by default. Low-confidence regions, suspicious repetitions, and significant disagreement are flagged or omitted rather than silently accepted.
6. Remaining words are grouped at bar starts and saved as .lrc entries. Blank bar entries clear lyrics in extended instrumental passages.

Instrumental detection is conservative, not infallible: tracks with too few reliable recognized words are skipped. Heavy vocal effects, loops, unusual rhythmic structures, and changing time signatures can confuse either the language model or beat tracker. Missing lyrics cannot be recovered by a beat-alignment step alone. Inspect the review reports and listen to suspect sections; editing the LRC manually may still be necessary.

Only audio source paths and generated sidecars are used. Intermediate stems and WAV files are held in a temporary directory and cleaned up automatically. The program does not upload any audio. Model weights are fetched from their upstream hosts on first use.

## Development

~~~sh
nix develop
uv sync --locked --extra mac
ruff check .
python -m unittest discover -s tests -v
nix flake check
~~~

MIT licensed.
