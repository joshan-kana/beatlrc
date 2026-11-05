import unittest

from lyr.core import Word, bar_grid, render_lrc, words_from_segments


class BarGridTests(unittest.TestCase):
    def test_discards_extra_half_bar_downbeats(self):
        beats = [i * 0.75 for i in range(64)]
        downbeats = [i * 3 for i in range(16)] + [16.5, 19.5, 22.5]
        bars, bpm, agreement = bar_grid(beats, downbeats, bar_division="1")
        self.assertEqual(bars[:5], [0, 3, 6, 9, 12])
        self.assertAlmostEqual(bpm, 80.0)
        self.assertGreater(agreement, 0.5)

    def test_auto_detects_double_time_from_extra_midbar_downbeats(self):
        beats = [i * 0.75 for i in range(72)]
        downbeats = [i * 3 for i in range(18)] + [7.5, 13.5, 19.5, 25.5, 31.5, 37.5]
        bars, bpm, agreement = bar_grid(beats, downbeats)
        self.assertEqual(bars[:6], [0.0, 1.5, 3.0, 4.5, 6.0, 7.5])
        self.assertEqual(bpm, 160)
        self.assertGreater(agreement, 0.65)

    def test_auto_stays_normal_when_no_half_bar_evidence(self):
        beats = [i * 0.75 for i in range(72)]
        downbeats = beats[::4]
        bars, bpm, _ = bar_grid(beats, downbeats)
        self.assertEqual(bars[:4], [0.0, 3.0, 6.0, 9.0])
        self.assertEqual(bpm, 80)

    def test_each_track_has_its_own_tempo(self):
        for tempo in (72, 95, 140):
            pulse = 60 / tempo
            beats = [i * pulse for i in range(80)]
            downbeats = beats[::4]
            bars, bpm, _ = bar_grid(beats, downbeats)
            self.assertAlmostEqual(bpm, tempo, delta=1)
            self.assertAlmostEqual(bars[3], 12 * pulse, delta=0.04)

    def test_rejects_missing_downbeat_agreement(self):
        beats = [i * 0.5 for i in range(60)]
        with self.assertRaisesRegex(ValueError, "downbeat phase"):
            bar_grid(beats, [])

    def test_manual_override(self):
        bars, bpm, _ = bar_grid([], [], bpm=100, offset=0.25, duration=40)
        self.assertAlmostEqual(bars[1], 2.65)
        self.assertEqual(bpm, 100)


class TranscriptTests(unittest.TestCase):
    def test_filters_low_confidence_segments(self):
        segments = [
            {
                "start": 0,
                "end": 2,
                "confidence": 0.09,
                "words": [{"word": "invented", "start": 0.2, "end": 0.8}],
            },
            {
                "start": 3,
                "end": 5,
                "confidence": 0.95,
                "words": [{"word": "hello", "start": 3.1, "end": 3.5}],
            },
        ]
        result = words_from_segments(segments, 10)
        self.assertEqual([w.text for w in result.words], ["hello"])
        self.assertTrue(result.flags)

    def test_repetition_loop_is_flagged_and_removed(self):
        words = []
        for repeat in range(7):
            for j, text in enumerate(["random", "syllables", "repeat", "now"]):
                start = repeat * 2 + j * 0.3
                words.append({"word": text, "start": start, "end": start + 0.2})
        result = words_from_segments(
            [{"start": 0, "end": 16, "words": words, "avg_logprob": -0.1}], 20
        )
        self.assertLess(len(result.words), 10)
        self.assertTrue(any("repetitive" in f.reason for f in result.flags))

    def test_invalid_word_timestamps_are_reported_once_per_segment(self):
        segment = {
            "start": 3,
            "end": 5,
            "words": [
                {"word": "a", "start": 3.1, "end": 3.1},
                {"word": "b", "start": 3.2, "end": 3.2},
                {"word": "c", "start": 3.3, "end": 3.6},
            ],
        }
        result = words_from_segments([segment], 10)
        self.assertEqual([w.text for w in result.words], ["c"])
        self.assertEqual(
            [f.reason for f in result.flags],
            ["invalid timestamps for 2 words"],
        )

    def test_early_pickup_is_displayed_before_first_downbeat(self):
        lrc = render_lrc([Word(0.25, 0.6, "Intro")], [2, 5, 8], "Example")
        self.assertIn("[00:00.25]Intro", lrc)
        self.assertIn("[00:02.00]", lrc)

    def test_repetition_flag_is_corroborated_in_two_pass_mode(self):
        segment = {
            "start": 0,
            "end": 4,
            "compression_ratio": 3.1,
            "avg_logprob": -0.2,
            "words": [{"word": "Hello", "start": 0.5, "end": 1.0}],
        }
        stem = words_from_segments([segment], 10, keep_repetitions=True)
        self.assertEqual([w.text for w in stem.words], ["Hello"])
        self.assertTrue(any("repetitive model output" in f.reason for f in stem.flags))
        self.assertEqual(words_from_segments([segment], 10).words, [])
        from lyr.pipeline import compare_transcripts

        mix = words_from_segments([segment], 10, keep_repetitions=True)
        self.assertEqual(
            [w.text for w in compare_transcripts(stem, mix, [0, 3, 6]).words],
            ["Hello"],
        )

    def test_first_beat_word_moves_to_next_bar(self):
        words = [
            Word(2.87, 3.10, "Onbeat"),
            Word(2.65, 2.80, "Previous"),
        ]
        lrc = render_lrc(words, [0, 3, 6], "Example")
        self.assertIn("[00:00.00]Previous", lrc)
        self.assertIn("[00:03.00]Onbeat", lrc)
        self.assertNotIn("Previous Onbeat", lrc)

    def test_exact_bar_changes(self):
        lyrics = [
            Word(0.1, 0.4, "First"),
            Word(3.1, 3.3, "second"),
            Word(6.2, 6.5, "Third"),
        ]
        lrc = render_lrc(lyrics, [0, 3, 6, 9], "Example")
        self.assertIn("[00:00.00]First", lrc)
        self.assertIn("[00:03.00]second", lrc)
        self.assertIn("[00:06.00]Third", lrc)
        self.assertIn("[00:09.00]", lrc)


class OutputSafetyTests(unittest.TestCase):
    def test_existing_backup_blocks_overwrite_before_processing(self):
        import tempfile
        from pathlib import Path

        from lyr.pipeline import Options, process_audio

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "demo.wav"
            audio.write_bytes(b"placeholder")
            (root / "demo.lrc").write_text("old")
            (root / "demo.before-lyr.lrc").write_text("backup")
            with self.assertRaisesRegex(FileExistsError, "existing backup"):
                process_audio(audio, Options(overwrite=True))


class SourceSelectionTests(unittest.TestCase):
    def test_prefers_lossless_with_identical_stems(self):
        from pathlib import Path

        from lyr.cli import unique_audio_sources

        sources = [
            Path("/music/new/demo.mp3"),
            Path("/music/new/demo.wav"),
            Path("/music/new/subfolder/demo.flac"),
        ]
        selected, skipped = unique_audio_sources(sources)
        self.assertEqual(
            selected,
            [Path("/music/new/demo.wav"), Path("/music/new/subfolder/demo.flac")],
        )
        self.assertEqual(skipped, [(Path("/music/new/demo.mp3"), Path("/music/new/demo.wav"))])


class VerificationTests(unittest.TestCase):
    def test_music_nospeech_score_does_not_override_high_logprob(self):
        segments = [
            {
                "start": 0,
                "end": 3,
                "avg_logprob": -0.25,
                "no_speech_prob": 0.80,
                "words": [{"word": "recognised", "start": 0.2, "end": 0.7}],
            }
        ]
        result = words_from_segments(segments, 4)
        self.assertEqual([w.text for w in result.words], ["recognised"])

    def test_mix_only_words_can_rescue_removed_vocals(self):
        from lyr.core import Analysis
        from lyr.pipeline import compare_transcripts

        stem = Analysis([], [], 0)
        mix = Analysis([Word(2, 2.5, "recognised", 0.95)], [], 1)
        output = compare_transcripts(stem, mix, [0, 3, 6])
        self.assertEqual([w.text for w in output.words], ["recognised"])
        self.assertTrue(output.flags)

    def test_repeated_hook_kept_when_both_transcriptions_agree(self):
        from lyr.pipeline import compare_transcripts

        words = []
        for repeat in range(6):
            for j, text in enumerate(["catch", "this", "hook", "again"]):
                start = repeat * 2.0 + j * 0.4
                words.append({"word": text, "start": start, "end": start + 0.3})
        segments = [{"start": 0, "end": 14, "avg_logprob": -0.1, "words": words}]
        stem = words_from_segments(segments, 20, keep_repetitions=True)
        mix = words_from_segments(segments, 20, keep_repetitions=True)
        self.assertTrue(any("highly repetitive" in f.reason for f in stem.flags))
        self.assertEqual(len(compare_transcripts(stem, mix, [0, 3, 6, 9, 12]).words), 24)
        empty = words_from_segments([], 20)
        self.assertFalse(compare_transcripts(stem, empty, [0, 3, 6, 9, 12]).words)


if __name__ == "__main__":
    unittest.main()
