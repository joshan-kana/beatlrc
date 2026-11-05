import unittest

from beatlrc.core import Word, bar_grid, render_lrc, words_from_segments


class BarGridTests(unittest.TestCase):
    def test_discards_extra_half_bar_downbeats(self):
        beats = [i * 0.75 for i in range(64)]
        downbeats = [i * 3 for i in range(16)] + [16.5, 19.5, 22.5]
        bars, bpm, agreement = bar_grid(beats, downbeats)
        self.assertEqual(bars[:5], [0, 3, 6, 9, 12])
        self.assertAlmostEqual(bpm, 80.0)
        self.assertGreater(agreement, 0.5)

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
        from beatlrc.core import Analysis
        from beatlrc.pipeline import compare_transcripts

        stem = Analysis([], [], 0)
        mix = Analysis([Word(2, 2.5, "recognised", 0.95)], [], 1)
        output = compare_transcripts(stem, mix)
        self.assertEqual([w.text for w in output.words], ["recognised"])
        self.assertTrue(output.flags)

    def test_repeated_hook_kept_when_both_transcriptions_agree(self):
        from beatlrc.pipeline import compare_transcripts

        words = []
        for repeat in range(6):
            for j, text in enumerate(["catch", "this", "hook", "again"]):
                start = repeat * 2.0 + j * 0.4
                words.append({"word": text, "start": start, "end": start + 0.3})
        segments = [{"start": 0, "end": 14, "avg_logprob": -0.1, "words": words}]
        stem = words_from_segments(segments, 20, keep_repetitions=True)
        mix = words_from_segments(segments, 20, keep_repetitions=True)
        self.assertTrue(any("highly repetitive" in f.reason for f in stem.flags))
        self.assertEqual(len(compare_transcripts(stem, mix).words), 24)
        empty = words_from_segments([], 20)
        self.assertFalse(compare_transcripts(stem, empty).words)


if __name__ == "__main__":
    unittest.main()
