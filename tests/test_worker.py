import logging
import os
from pathlib import Path
from datetime import datetime
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

import worker
import diarize


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.volume = self.root / "volume"
        self.volume.mkdir()
        self.sources = (self.root / "local", self.volume / "meetings", self.volume / "obs")
        for source in self.sources:
            source.mkdir()
        self.destination = self.volume / "transcripts"
        self.model = self.root / "model"
        self.model.mkdir()
        self.ffmpeg = self.root / "ffmpeg"
        self.ffmpeg.touch()
        patcher = patch.object(worker.config, "FFMPEG", self.ffmpeg)
        patcher.start()
        self.addCleanup(patcher.stop)
        labels_patcher = patch.object(worker.config, "SOURCE_LABELS", ("Meetings", "Meetings", "OBS"))
        labels_patcher.start()
        self.addCleanup(labels_patcher.stop)
        diarization_patcher = patch.object(worker.config, "DIARIZATION_ENABLED", False)
        diarization_patcher.start()
        self.addCleanup(diarization_patcher.stop)
        self.state_file = self.root / "state.json"
        self.state = worker.load_state(self.state_file)
        self.observations = {}
        self.now = 1000.0
        self.calls = []
        self.scan()  # Пустые источники уже приняты как baseline.

    def make_file(self, source, name="meeting.mp4", data=b"media"):
        path = source / name
        path.write_bytes(data)
        os.utime(path, (100, 100))
        return path

    def transcribe(self, path, model):
        self.calls.append(path)
        return {"segments": [{"start": 3.0, "end": 8.0, "text": " Речь. "}]}

    def scan(self, *, mounted=True, probe=True, open_paths=(), diarizer=None):
        with patch.object(Path, "is_mount", return_value=mounted), patch.object(
                worker, "is_open", side_effect=lambda path: path in open_paths), patch.object(
                worker, "has_audio", side_effect=(probe if callable(probe) else
                                                  None if probe else ValueError("bad media"))) as media:
            return worker.scan_once(
                self.state, self.observations, sources=self.sources,
                destination=self.destination, volume=self.volume,
                state_file=self.state_file, model_dir=self.model,
                transcriber=self.transcribe, diarizer=diarizer or (lambda path: []),
                clock=lambda: self.now,
                stable_seconds=worker.config.STABLE_SECONDS, retry_seconds=3600)

    def fresh_state(self):
        self.state = worker.load_state(self.root / "fresh-state.json")
        self.observations = {}

    def test_three_sources_collisions_restart_and_repeat(self):
        paths = [self.make_file(source) for source in self.sources]
        self.assertEqual(self.scan(), 0)
        self.now += 121
        self.assertEqual(self.scan(), 3)
        outputs = list(self.destination.glob("*.txt"))
        self.assertEqual(len(outputs), 3)
        self.assertEqual(len({p.name for p in outputs}), 3)
        self.assertTrue(all("[00:00:03 - 00:00:08] Речь." in p.read_text() for p in outputs))
        self.assertEqual(self.scan(), 0)
        self.state = worker.load_state(self.state_file)
        self.observations = {}
        self.assertEqual(self.scan(), 0)
        self.now += 121
        self.assertEqual(self.scan(), 0)
        self.assertEqual(self.calls, paths)

    def test_waits_for_file_that_is_still_changing(self):
        path = self.make_file(self.sources[0])
        self.scan()
        self.now += 60
        path.write_bytes(b"media more")
        os.utime(path, (self.now, self.now))
        self.assertEqual(self.scan(), 0)
        self.now += 60
        self.assertEqual(self.scan(), 0)
        self.now += 61
        self.assertEqual(self.scan(), 1)

    def test_stable_size_and_mtime_reach_90_seconds_before_probe(self):
        self.make_file(self.sources[0])
        probe = Mock()
        self.scan(probe=probe)
        for elapsed in (30, 30, 29):
            self.now += elapsed
            self.assertEqual(self.scan(probe=probe), 0)
        probe.assert_not_called()
        self.now += 1
        self.assertEqual(self.scan(probe=probe), 1)
        probe.assert_called_once()
        self.assertEqual(len(self.calls), 1)

    def test_size_change_resets_stability_timer(self):
        path = self.make_file(self.sources[0])
        probe = Mock()
        self.scan(probe=probe)
        self.now += 60
        path.write_bytes(b"media enlarged")
        os.utime(path, (100, 100))  # Same mtime, only size changes.
        self.scan(probe=probe)
        self.now += 89
        self.assertEqual(self.scan(probe=probe), 0)
        probe.assert_not_called()
        self.now += 1
        self.assertEqual(self.scan(probe=probe), 1)

    def test_mtime_change_with_same_size_resets_stability_timer(self):
        path = self.make_file(self.sources[0])
        probe = Mock()
        self.scan(probe=probe)
        self.now += 60
        os.utime(path, ns=(200_000_000_000, 200_000_000_000))
        self.scan(probe=probe)
        self.now += 89
        self.assertEqual(self.scan(probe=probe), 0)
        probe.assert_not_called()
        self.now += 1
        self.assertEqual(self.scan(probe=probe), 1)

    def test_probe_failure_waits_and_never_calls_whisper_before_success(self):
        self.make_file(self.sources[0])
        probe = Mock(side_effect=[ValueError("container unfinished"), None])
        self.scan(probe=probe)
        self.now += 90
        self.assertEqual(self.scan(probe=probe), 0)
        self.assertEqual(self.calls, [])
        self.now += 59
        self.assertEqual(self.scan(probe=probe), 0)
        self.assertEqual(probe.call_count, 1)
        self.now += 1
        self.assertEqual(self.scan(probe=probe), 1)
        self.assertEqual(probe.call_count, 2)
        self.assertEqual(len(self.calls), 1)

    def test_changed_file_clears_probe_failures(self):
        path = self.make_file(self.sources[0])
        probe = Mock(side_effect=ValueError("unfinished"))
        self.scan(probe=probe)
        self.now += 90
        self.scan(probe=probe)
        self.assertEqual(self.observations[str(path)][2], 1)
        self.now += 30
        path.write_bytes(b"new size")
        os.utime(path, (100, 100))
        self.scan(probe=probe)
        self.assertEqual(self.observations[str(path)][2], 0)
        self.now += 90
        self.scan(probe=probe)
        self.assertEqual(self.observations[str(path)][2], 1)
        self.assertEqual(self.state["files"][str(path)]["signature"],
                         worker.signature(path))
        self.assertEqual(self.state["files"][str(path)]["attempts"], 1)

    def test_three_unchanged_probe_failures_mark_failed(self):
        path = self.make_file(self.sources[0])
        probe = Mock(side_effect=ValueError("unreadable container"))
        self.scan(probe=probe)
        for delta in (90, 60, 60):
            self.now += delta
            self.scan(probe=probe)
        self.assertEqual(probe.call_count, 3)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.state["files"][str(path)]["status"], "failed")
        self.assertEqual(self.state["files"][str(path)]["reason"], "ffprobe")
        self.now += 3600
        self.scan(probe=probe)
        self.assertEqual(probe.call_count, 3)

    def test_probe_attempts_survive_worker_restart(self):
        path = self.make_file(self.sources[0])
        probe = Mock(side_effect=ValueError("unreadable"))
        self.scan(probe=probe)
        self.now += 90
        self.scan(probe=probe)
        self.assertEqual(self.state["files"][str(path)]["attempts"], 1)
        self.state = worker.load_state(self.state_file)
        self.observations = {}
        self.scan(probe=probe)
        self.now += 90
        self.scan(probe=probe)
        self.assertEqual(probe.call_count, 2)
        self.assertEqual(self.state["files"][str(path)]["attempts"], 2)

    def test_unavailable_volume_does_not_count_probe_failure(self):
        path = self.make_file(self.sources[0])
        probe = Mock(side_effect=ValueError("unreadable"))
        self.scan(probe=probe)
        self.now += 90
        self.scan(mounted=False, probe=probe)
        probe.assert_not_called()
        self.assertEqual(self.observations[str(path)][2], 0)
        self.scan(probe=probe)
        self.assertEqual(self.observations[str(path)][2], 1)

    def test_unmounted_volume_leaves_local_recording_pending(self):
        self.make_file(self.sources[0])
        self.assertEqual(self.scan(mounted=False), 0)
        self.assertEqual(list(self.destination.glob("*.txt")), [])
        self.scan()
        self.now += 121
        self.assertEqual(self.scan(), 1)

    def test_first_run_without_volume_keeps_new_local_file_pending(self):
        self.fresh_state()
        old = self.make_file(self.sources[0], "old.mp4")
        self.scan(mounted=False)
        self.assertEqual(self.state["baseline_sources"], [str(self.sources[0])])
        self.assertEqual(self.state["files"][str(old)]["status"], "baseline")
        new = self.make_file(self.sources[0], "new.wav")
        self.scan(mounted=False)
        self.now += 121
        self.assertEqual(self.scan(), 1)
        self.assertEqual(self.calls, [new])

    def test_bad_media_does_not_stop_next_file(self):
        bad = self.make_file(self.sources[0], "bad.mp4")
        good = self.make_file(self.sources[1], "good.wav")
        def probe(path, ffprobe):
            if path == bad:
                raise ValueError("bad media")
        with patch.object(Path, "is_mount", return_value=True), patch.object(
                worker, "is_open", return_value=False), patch.object(worker, "has_audio", side_effect=probe):
            kwargs = dict(sources=self.sources, destination=self.destination, volume=self.volume,
                          state_file=self.state_file, model_dir=self.model,
                          transcriber=self.transcribe, clock=lambda: self.now, stable_seconds=120)
            worker.scan_once(self.state, self.observations, **kwargs)
            self.now += 121
            with self.assertLogs(level=logging.WARNING):
                self.assertEqual(worker.scan_once(self.state, self.observations, **kwargs), 1)
            self.now += 60
            worker.scan_once(self.state, self.observations, **kwargs)
            self.now += 60
            with self.assertLogs(level=logging.ERROR):
                worker.scan_once(self.state, self.observations, **kwargs)
        self.assertEqual(self.calls, [good])
        self.assertEqual(self.state["files"][str(bad)]["status"], "failed")

    def test_open_file_is_not_transcribed(self):
        self.make_file(self.sources[0])
        self.scan()
        self.now += 121
        with patch.object(Path, "is_mount", return_value=True), patch.object(
                worker, "is_open", return_value=True):
            self.assertEqual(worker.scan_once(
                self.state, self.observations, sources=self.sources,
                destination=self.destination, volume=self.volume,
                state_file=self.state_file, model_dir=self.model,
                transcriber=self.transcribe, clock=lambda: self.now,
                stable_seconds=120), 0)
        self.assertEqual(self.calls, [])

    def test_first_run_baselines_old_files_and_new_file_runs(self):
        self.fresh_state()
        old = self.make_file(self.sources[0], "old.mp4")
        self.assertEqual(self.scan(), 0)
        self.assertEqual(self.state["files"][str(old)]["status"], "baseline")
        self.now += 121
        self.assertEqual(self.scan(), 0)
        self.state = worker.load_state(self.state_file)
        self.observations = {}
        self.assertEqual(self.scan(), 0)
        new = self.make_file(self.sources[0], "new.mp4")
        self.scan()
        self.now += 121
        self.assertEqual(self.scan(), 1)
        self.assertEqual(self.calls, [new])
        self.assertTrue((self.destination / "new [Meetings].txt").is_file())

    def test_file_changed_after_baseline_is_processed(self):
        self.fresh_state()
        old = self.make_file(self.sources[0], "changed.mp4")
        self.scan()
        old.write_bytes(b"new recording")
        os.utime(old, (100, 100))
        self.scan()
        self.now += 121
        self.assertEqual(self.scan(), 1)
        self.assertEqual(self.calls, [old])

    def test_file_open_during_baseline_is_not_lost(self):
        self.fresh_state()
        writing = self.make_file(self.sources[0], "writing.wav")
        self.scan(open_paths=(writing,))
        self.assertNotIn(str(writing), self.state["files"])
        self.scan(open_paths=(writing,))
        self.now += 121
        self.assertEqual(self.scan(), 1)
        self.assertEqual(self.calls, [writing])

    def test_language_is_passed_to_mlx(self):
        fake = types.SimpleNamespace(transcribe=Mock(return_value={"segments": []}))
        with patch.dict(sys.modules, {"mlx_whisper": fake}):
            worker.transcribe(Path("audio.wav"), self.model)
        self.assertEqual(fake.transcribe.call_args.kwargs["language"], "ru")

    def test_plain_name_and_collision_without_overwrite(self):
        first = self.make_file(self.sources[0], "same.mp4")
        self.scan()
        self.now += 121
        self.assertEqual(self.scan(), 1)
        plain = self.destination / "same [Meetings].txt"
        original = plain.read_text()
        second = self.make_file(self.sources[1], "same.mov")
        self.scan()
        self.now += 121
        self.assertEqual(self.scan(), 1)
        self.assertEqual(plain.read_text(), original)
        self.assertEqual(len(list(self.destination.glob("same*.txt"))), 2)
        self.assertNotEqual(self.state["files"][str(first)]["output"],
                            self.state["files"][str(second)]["output"])
        foreign = self.destination / "foreign [OBS].txt"
        foreign.write_text("existing result")
        self.make_file(self.sources[2], "foreign.m4a")
        self.scan()
        self.now += 121
        self.assertEqual(self.scan(), 1)
        self.assertEqual(foreign.read_text(), "existing result")
        self.assertEqual(len(list(self.destination.glob("foreign*.txt"))), 2)

    def test_local_meetings_uses_required_label_from_source_root(self):
        nested = self.sources[0] / "OBS"
        nested.mkdir()
        self.make_file(nested, "Agenda.mp4")
        self.scan()
        self.now += 90
        self.assertEqual(self.scan(), 1)
        self.assertTrue((self.destination / "Agenda [Meetings].txt").is_file())

    def test_second_meetings_source_uses_configured_label(self):
        self.make_file(self.sources[1], "Client Meeting.webm")
        self.scan()
        self.now += 90
        self.assertEqual(self.scan(), 1)
        self.assertTrue((self.destination / "Client Meeting [Meetings].txt").is_file())

    def test_obs_uses_required_label(self):
        self.make_file(self.sources[2], "Team Discussion.mkv")
        self.scan()
        self.now += 90
        self.assertEqual(self.scan(), 1)
        self.assertTrue((self.destination / "Team Discussion [OBS].txt").is_file())

    def test_same_stem_in_two_meetings_roots_never_overwrites(self):
        self.make_file(self.sources[0], "Agenda.mp4")
        self.scan()
        self.now += 90
        self.assertEqual(self.scan(), 1)
        normal = self.destination / "Agenda [Meetings].txt"
        original = normal.read_bytes()
        self.make_file(self.sources[1], "Agenda.mp4")
        self.scan()
        self.now += 90
        self.assertEqual(self.scan(), 1)
        self.assertEqual(normal.read_bytes(), original)
        names = {p.name for p in self.destination.glob("*.txt")}
        self.assertEqual(len(names), 2)
        self.assertTrue(any(name.startswith("Agenda [Meetings]__") for name in names))

    def test_same_stem_obs_and_meetings_need_no_hash(self):
        self.make_file(self.sources[0], "Agenda.mp4")
        self.make_file(self.sources[2], "Agenda.mkv")
        self.scan()
        self.now += 90
        self.assertEqual(self.scan(), 2)
        self.assertEqual({p.name for p in self.destination.glob("*.txt")},
                         {"Agenda [Meetings].txt", "Agenda [OBS].txt"})

    def test_long_meetings_name_respects_name_max(self):
        limit = os.pathconf(self.sources[0], "PC_NAME_MAX")
        stem = "t" * (limit - len(".mp4"))
        self.make_file(self.sources[0], stem + ".mp4")
        self.scan()
        self.now += 90
        self.assertEqual(self.scan(), 1)
        target = next(self.destination.glob("*.txt"))
        self.assertEqual(len(os.fsencode(target.name)), limit)
        self.assertTrue(target.name.endswith(" [Meetings].txt"))
        self.assertEqual(target.stem.removesuffix(" [Meetings]"),
                         stem[:limit - len(os.fsencode(" [Meetings].txt"))])

    def test_long_obs_name_respects_name_max(self):
        limit = os.pathconf(self.sources[2], "PC_NAME_MAX")
        stem = "o" * (limit - len(".mkv"))
        self.make_file(self.sources[2], stem + ".mkv")
        self.scan()
        self.now += 90
        self.assertEqual(self.scan(), 1)
        target = next(self.destination.glob("*.txt"))
        self.assertEqual(len(os.fsencode(target.name)), limit)
        self.assertTrue(target.name.endswith(" [OBS].txt"))
        self.assertEqual(target.stem.removesuffix(" [OBS]"),
                         stem[:limit - len(os.fsencode(" [OBS].txt"))])

    def test_long_name_keeps_full_stem(self):
        stem = "meeting_" + "a" * 80
        self.make_file(self.sources[0], stem + ".mp4")
        self.scan()
        self.now += 121
        self.assertEqual(self.scan(), 1)
        self.assertTrue((self.destination / (stem + " [Meetings].txt")).is_file())

    def test_distinct_long_names_do_not_collide(self):
        stems = ("a" * 80 + "_first", "a" * 80 + "_second")
        for stem in stems:
            self.make_file(self.sources[0], stem + ".mp4")
        self.scan()
        self.now += 121
        self.assertEqual(self.scan(), 2)
        self.assertEqual({path.name for path in self.destination.glob("*.txt")},
                         {stem + " [Meetings].txt" for stem in stems})

    def test_extreme_name_trims_only_for_collision_suffix(self):
        stem = "я" * 125  # Source filename remains within the macOS component limit.
        self.make_file(self.sources[0], stem + ".mp4")
        self.scan()
        self.now += 121
        self.assertEqual(self.scan(), 1)
        first_name = self.state["files"][str(self.sources[0] / (stem + ".mp4"))]["output"]
        self.assertTrue((self.destination / first_name).is_file())
        self.assertTrue(first_name.endswith(" [Meetings].txt"))
        self.make_file(self.sources[1], stem + ".wav")
        self.scan()
        self.now += 121
        self.assertEqual(self.scan(), 1)
        outputs = list(self.destination.glob("*.txt"))
        self.assertEqual(len(outputs), 2)
        suffixed = next(path for path in outputs if path.name != first_name)
        self.assertTrue(suffixed.is_file())
        limit = os.pathconf(self.destination, "PC_NAME_MAX")
        self.assertLessEqual(len(os.fsencode(suffixed.name)), limit)
        trimmed_with_label, suffix = suffixed.stem.split("__", 1)
        trimmed = trimmed_with_label.removesuffix(" [Meetings]")
        self.assertEqual(trimmed, stem[:len(trimmed)])
        self.assertEqual(len(suffix), 8)
        self.assertGreater(len(os.fsencode(trimmed + "я [Meetings]__" + suffix + ".txt")), limit)

    def test_maximum_ascii_name_publishes_without_temp_leftover(self):
        limit = os.pathconf(self.sources[0], "PC_NAME_MAX")
        stem = "z" * (limit - len(".mp4"))
        self.make_file(self.sources[0], stem + ".mp4")
        self.scan()
        self.now += 121
        self.assertEqual(self.scan(), 1)
        trimmed = stem[:limit - len(os.fsencode(" [Meetings].txt"))]
        target = self.destination / (trimmed + " [Meetings].txt")
        self.assertEqual(len(os.fsencode(target.name)), limit)
        self.assertTrue(target.is_file())
        self.assertEqual(list(self.destination.iterdir()), [target])

    def test_volume_disappears_after_transcription_without_failure(self):
        path = self.make_file(self.sources[0], "vanish.wav")
        self.scan()
        self.now += 121
        mounted = [True]
        def transcribe_then_unmount(p, model):
            mounted[0] = False
            return self.transcribe(p, model)
        with patch.object(Path, "is_mount", side_effect=lambda: mounted[0]), patch.object(
                worker, "is_open", return_value=False), patch.object(worker, "has_audio"):
            self.assertEqual(worker.scan_once(
                self.state, self.observations, sources=self.sources,
                destination=self.destination, volume=self.volume,
                state_file=self.state_file, model_dir=self.model,
                transcriber=transcribe_then_unmount, clock=lambda: self.now,
                stable_seconds=120), 0)
        self.assertNotEqual(self.state["files"].get(str(path), {}).get("status"), "failed")
        self.assertFalse((self.destination / "vanish [Meetings].txt").exists())
        mounted[0] = True
        self.scan()
        self.now += 121
        self.assertEqual(self.scan(), 1)

    def test_v1_failed_history_is_baselined(self):
        import json
        self.fresh_state()
        old = self.make_file(self.sources[0], "old-failed.mp4")
        legacy = {"version": 1, "files": {str(old): {"signature": worker.signature(old),
                                                    "status": "failed", "attempted_at": 900}}}
        self.state_file.write_text(json.dumps(legacy))
        self.state = worker.load_state(self.state_file)
        self.scan()
        self.assertEqual(self.state["files"][str(old)]["status"], "baseline")
        self.now += 121
        self.assertEqual(self.scan(), 0)

    def test_destination_write_error_is_pending_without_retry_penalty(self):
        path = self.make_file(self.sources[0], "waiting.wav")
        self.scan()
        self.now += 121
        real_replace = worker.os.replace
        def replace(source, target):
            if Path(target).parent == self.destination:
                raise OSError("destination unavailable")
            return real_replace(source, target)
        with patch.object(Path, "is_mount", return_value=True), patch.object(
                worker, "is_open", return_value=False), patch.object(worker, "has_audio"), patch.object(
                worker.os, "replace", side_effect=replace):
            with self.assertLogs(level=logging.ERROR):
                self.assertEqual(worker.scan_once(
                    self.state, self.observations, sources=self.sources,
                    destination=self.destination, volume=self.volume,
                    state_file=self.state_file, model_dir=self.model,
                    transcriber=self.transcribe, clock=lambda: self.now,
                    stable_seconds=120), 0)
        self.assertEqual(self.state["files"][str(path)]["status"], "pending_output")
        self.assertNotIn("attempted_at", self.state["files"][str(path)])
        self.assertEqual(list(self.destination.glob(".transcriber-*.tmp")), [])
        self.assertEqual(self.scan(), 1)
        self.assertEqual(self.state["files"][str(path)]["status"], "done")

    def test_diarization_failure_saves_plain_txt_and_done_state(self):
        path = self.make_file(self.sources[0], "fallback.wav")
        with patch.object(worker.config, "DIARIZATION_ENABLED", True):
            self.scan()
            self.now += 90
            with self.assertLogs(level=logging.WARNING) as logs:
                self.assertEqual(self.scan(diarizer=Mock(side_effect=RuntimeError("test failure"))), 1)
        self.assertEqual(self.state["files"][str(path)]["status"], "done")
        content = (self.destination / "fallback [Meetings].txt").read_text()
        self.assertIn("[00:00:03 - 00:00:08] Речь.", content)
        self.assertNotIn("SPEAKER_", content)
        self.assertTrue(any("diarization failed; transcript saved without speaker labels" in line
                            for line in logs.output))

    def test_diarization_success_saves_speaker_txt(self):
        path = self.make_file(self.sources[0], "speakers.wav")
        self.transcribe = lambda path, model: {"segments": [{"start": 1, "end": 4,
            "text": " Привет. Да.", "words": [{"start": 1, "end": 2, "word": " Привет."},
                                         {"start": 3, "end": 4, "word": " Да."}]}]}
        turns = [{"start": 0, "end": 2.5, "speaker": "first"},
                 {"start": 2.5, "end": 5, "speaker": "second"}]
        with patch.object(worker.config, "DIARIZATION_ENABLED", True):
            self.scan()
            self.now += 90
            self.assertEqual(self.scan(diarizer=Mock(return_value=turns)), 1)
        self.assertEqual(self.state["files"][str(path)]["status"], "done")
        content = (self.destination / "speakers [Meetings].txt").read_text()
        self.assertIn("SPEAKER_00: Привет.", content)
        self.assertIn("SPEAKER_01: Да.", content)

    def test_empty_diarization_result_saves_plain_txt(self):
        path = self.make_file(self.sources[0], "empty.wav")
        with patch.object(worker.config, "DIARIZATION_ENABLED", True):
            self.scan()
            self.now += 90
            with self.assertLogs(level=logging.WARNING):
                self.assertEqual(self.scan(diarizer=Mock(return_value=[])), 1)
        self.assertEqual(self.state["files"][str(path)]["status"], "done")
        self.assertNotIn("SPEAKER_", (self.destination / "empty [Meetings].txt").read_text())

    def test_disabled_diarization_does_not_run_model(self):
        self.make_file(self.sources[0], "plain.wav")
        self.scan()
        self.now += 90
        diarizer = Mock(side_effect=RuntimeError("must not run"))
        self.assertEqual(self.scan(diarizer=diarizer), 1)
        diarizer.assert_not_called()
        content = (self.destination / "plain [Meetings].txt").read_text()
        self.assertNotIn("SPEAKER_", content)

    def test_diarization_subprocess_timeout_saves_plain_txt_and_done(self):
        path = self.make_file(self.sources[0], "timeout.wav")
        fake_probe = self.root / "probe-duration"
        fake_probe.write_text("#!/bin/sh\nprintf '1.0\\n'\n")
        fake_probe.chmod(0o755)
        sleeping_diarizer = self.root / "sleeping-diarizer"
        sleeping_diarizer.write_text("#!/bin/sh\nexec sleep 30\n")
        sleeping_diarizer.chmod(0o755)
        with patch.object(worker.config, "DIARIZATION_ENABLED", True), patch.object(
                worker.config, "FFPROBE", fake_probe), patch.object(
                worker.config, "DIARIZATION_PYTHON", sleeping_diarizer), patch.object(
                worker.config, "DIARIZATION_TIMEOUT_MIN_SECONDS", 0.2), patch.object(
                worker.config, "DIARIZATION_TIMEOUT_MULTIPLIER", 0):
            self.scan()
            self.now += 90
            start = time.monotonic()
            with self.assertLogs(level=logging.WARNING) as logs:
                self.assertEqual(self.scan(diarizer=worker.diarize), 1)
            self.assertLess(time.monotonic() - start, 3)
        self.assertEqual(self.state["files"][str(path)]["status"], "done")
        content = (self.destination / "timeout [Meetings].txt").read_text()
        self.assertIn("[00:00:03 - 00:00:08] Речь.", content)
        self.assertNotIn("SPEAKER_", content)
        self.assertTrue(any("diarization failed" in line and "timed out" in line
                            for line in logs.output))


class SpeakerTests(unittest.TestCase):
    def setUp(self):
        self.path = Path("meeting.wav")
        self.now = datetime(2026, 9, 30, 12, 0)
        patcher = patch.object(worker.config, "FFMPEG", Path("/usr/bin/ffmpeg"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def result(self, words):
        return {"segments": [{"start": words[0][0], "end": words[-1][1],
                              "text": "".join(word[2] for word in words),
                              "words": [{"start": start, "end": end, "word": text}
                                        for start, end, text in words]}]}

    def turns(self, *items):
        return [{"start": start, "end": end, "speaker": label}
                for start, end, label in items]

    def text(self, words, turns):
        return worker.format_speaker_transcript(self.path, self.result(words),
                                                turns, self.now)

    def test_word_timestamps_passed_to_mlx(self):
        fake = types.SimpleNamespace(transcribe=Mock(return_value={"segments": []}))
        with patch.dict(sys.modules, {"mlx_whisper": fake}):
            worker.transcribe(self.path, Path("model"))
        self.assertIs(fake.transcribe.call_args.kwargs["word_timestamps"], True)

    def test_russian_language_passed_to_mlx(self):
        fake = types.SimpleNamespace(transcribe=Mock(return_value={"segments": []}))
        with patch.dict(sys.modules, {"mlx_whisper": fake}):
            worker.transcribe(self.path, Path("model"))
        self.assertEqual(fake.transcribe.call_args.kwargs["language"], "ru")

    def test_hallucination_threshold_passed_to_mlx(self):
        fake = types.SimpleNamespace(transcribe=Mock(return_value={"segments": []}))
        with patch.dict(sys.modules, {"mlx_whisper": fake}):
            worker.transcribe(self.path, Path("model"))
        self.assertEqual(fake.transcribe.call_args.kwargs["hallucination_silence_threshold"], 2.0)

    def test_one_speaker_groups_words_into_one_line(self):
        text = self.text([(1, 2, " Привет"), (2, 3, " всем.")],
                         self.turns((0, 4, "a")))
        self.assertEqual(text.count("SPEAKER_00:"), 1)
        self.assertIn("SPEAKER_00: Привет всем.", text)

    def test_two_speakers_split_one_whisper_segment(self):
        text = self.text([(1, 2, " Доброе"), (2, 3, " утро."),
                          (5, 6, " Да,"), (6, 7, " начнём.")],
                         self.turns((0, 4, "a"), (4.5, 8, "b")))
        self.assertIn("SPEAKER_00: Доброе утро.", text)
        self.assertIn("SPEAKER_01: Да, начнём.", text)
        self.assertEqual(text.count("SPEAKER_"), 2)

    def test_assignment_uses_maximum_overlap_not_word_start(self):
        text = self.text([(2.9, 4.0, " Переход.")],
                         self.turns((0, 3, "a"), (3, 5, "b")))
        self.assertIn("SPEAKER_01: Переход.", text)

    def test_overlapping_speech_assigns_word_once(self):
        text = self.text([(2, 3, " Одно"), (3, 4, " слово.")],
                         self.turns((0, 3.2, "a"), (2.5, 5, "b")))
        self.assertEqual(text.count("Одно"), 1)
        self.assertEqual(text.count("слово."), 1)

    def test_adjacent_words_stay_in_normal_reply(self):
        words = [(x, x + 0.2, " Слово" if x else " Первое") for x in (0, .2, .4, .6)]
        text = self.text(words, self.turns((0, 2, "a")))
        self.assertEqual(text.count("SPEAKER_00:"), 1)

    def test_long_silence_word_is_preserved_without_false_speaker(self):
        text = self.text([(1, 2, " Речь."), (10, 11, " Возможно.")],
                         self.turns((0, 3, "a")))
        self.assertIn("SPEAKER_00: Речь.", text)
        self.assertIn("[00:00:10 - 00:00:11] Возможно.", text)

    def test_empty_diarization_result_can_fall_back(self):
        with self.assertRaisesRegex(ValueError, "no speaker turns"):
            self.text([(1, 2, " Текст.")], [])

    def test_speaker_count_is_reported(self):
        text = self.text([(1, 2, " Первый."), (4, 5, " Второй.")],
                         self.turns((0, 3, "a"), (3, 6, "b")))
        self.assertIn("Спикеров обнаружено: 2", text)

    def test_exclusive_diarization_is_preferred(self):
        exclusive = [(types.SimpleNamespace(start=1, end=2), "exclusive")]
        regular = [(types.SimpleNamespace(start=3, end=4), "regular")]
        result = diarize.speaker_turns(types.SimpleNamespace(
            exclusive_speaker_diarization=exclusive, speaker_diarization=regular))
        self.assertEqual(result, [{"start": 1, "end": 2, "speaker": "exclusive"}])

    def test_regular_diarization_is_compatibility_fallback(self):
        regular = [(types.SimpleNamespace(start=3, end=4), "regular")]
        result = diarize.speaker_turns(types.SimpleNamespace(speaker_diarization=regular))
        self.assertEqual(result, [{"start": 3, "end": 4, "speaker": "regular"}])

    def test_exclusive_turns_split_one_whisper_segment(self):
        exclusive = [(types.SimpleNamespace(start=0, end=3), "a"),
                     (types.SimpleNamespace(start=3, end=6), "b")]
        regular = [(types.SimpleNamespace(start=0, end=6), "a")]
        turns = diarize.speaker_turns(types.SimpleNamespace(
            exclusive_speaker_diarization=exclusive, speaker_diarization=regular))
        text = self.text([(1, 2, " Первый."), (4, 5, " Второй.")], turns)
        self.assertIn("SPEAKER_00: Первый.", text)
        self.assertIn("SPEAKER_01: Второй.", text)

    def test_diarization_timeout_scales_with_audio_duration(self):
        with patch.object(worker.subprocess, "run", side_effect=[
                types.SimpleNamespace(stdout="1200.0\n"),
                types.SimpleNamespace(stdout="[]")]) as run:
            self.assertEqual(worker.diarize(self.path), [])
        self.assertEqual(run.call_args_list[1].kwargs["timeout"], 3600)



if __name__ == "__main__":
    unittest.main()
