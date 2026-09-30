import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import settings
import worker


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ffmpeg = self.root / "ffmpeg"
        self.ffprobe = self.root / "ffprobe"
        self.ffmpeg.touch()
        self.ffprobe.touch()

    def config(self, sources, destination, extra=""):
        path = self.root / "config.toml"
        lines = []
        for source, label in sources:
            lines.extend(("[[sources]]", f'path = "{source}"', f'label = "{label}"', ""))
        lines.extend(("[destination]", f'path = "{destination}"', "", "[tools]",
                      f'ffmpeg = "{self.ffmpeg}"', f'ffprobe = "{self.ffprobe}"', extra))
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def test_toml_loads_tilde_unicode_spaces_and_multiple_sources(self):
        path = self.config([("~/Documents/Мои встречи", "Собрание"),
                            (str(self.root / "Meetings 2"), "Zoom"),
                            (str(self.root / "OBS"), "OBS")], self.root / "Results")
        values = settings.load(path)
        self.assertEqual(len(values["sources"]), 3)
        self.assertEqual(values["sources"][0], Path.home() / "Documents/Мои встречи")
        self.assertEqual(values["labels"], ("Собрание", "Zoom", "OBS"))

    def test_custom_label_controls_filename(self):
        source = self.root / "Zoom"; source.mkdir()
        destination = self.root / "out"; destination.mkdir()
        path = self.config([(str(source), "Custom")], destination)
        values = settings.load(path)
        recording = source / "meeting.mp4"; recording.touch()
        with patch.object(settings, "SOURCE_LABELS", values["labels"]):
            result = worker.output_path(recording, destination, {"files": {}}, values["sources"])
        self.assertEqual(result.name, "meeting [Custom].txt")

    def test_internal_destination_needs_no_mount(self):
        path = self.config([(str(self.root / "source"), "One")], self.root / "out")
        values = settings.load(path)
        with patch.object(Path, "is_mount", side_effect=AssertionError("unexpected mount check")):
            self.assertTrue(settings.path_available(values["destination"]))

    def test_external_destination_requires_actual_mount(self):
        path = self.config([(str(self.root / "source"), "One")], "/Volumes/TestDisk/out")
        values = settings.load(path)
        with patch.object(Path, "is_mount", return_value=False):
            self.assertFalse(settings.path_available(values["destination"]))
        with patch.object(Path, "is_mount", return_value=True):
            self.assertTrue(settings.path_available(values["destination"]))

    def test_external_source_and_destination_are_independent(self):
        source = Path("/Volumes/SourceDisk/in")
        destination = Path("/Volumes/OtherDisk/out")
        def mounted(path):
            return str(path) == "/Volumes/SourceDisk"
        with patch.object(Path, "is_mount", mounted):
            self.assertTrue(settings.path_available(source))
            self.assertFalse(settings.path_available(destination))

    def test_missing_external_source_does_not_stop_internal_source(self):
        local = self.root / "local"; local.mkdir()
        missing = Path("/Volumes/MissingTestDisk/recordings")
        destination = self.root / "out"
        state = {"version": 2, "files": {}, "baseline_sources": []}
        with patch.object(settings, "FFMPEG", self.ffmpeg), patch.object(
                Path, "is_mount", return_value=False):
            worker.scan_once(state, {}, sources=(missing, local), destination=destination,
                             state_file=self.root / "state.json", model_dir=self.root,
                             ffprobe=self.ffprobe)
        self.assertIn(str(local), state["baseline_sources"])
        self.assertNotIn(str(missing), state["baseline_sources"])

    def test_missing_external_destination_never_creates_fake_mount(self):
        source = self.root / "local"; source.mkdir()
        destination = Path("/Volumes/MissingTestDiskForTranscriber/out")
        state = {"version": 2, "files": {}, "baseline_sources": [str(source)]}
        with patch.object(settings, "FFMPEG", self.ffmpeg), patch.object(
                Path, "is_mount", return_value=False):
            worker.scan_once(state, {}, sources=(source,), destination=destination,
                             state_file=self.root / "state.json", model_dir=self.root,
                             ffprobe=self.ffprobe)
        self.assertFalse(destination.exists())

    def test_ffmpeg_discovered_from_path(self):
        source = self.root / "source"
        path = self.config([(str(source), "One")], self.root / "out")
        content = path.read_text().split("[tools]")[0]
        path.write_text(content)
        with patch.object(settings.shutil, "which", side_effect=lambda name: str(self.root / name)):
            values = settings.load(path)
        self.assertEqual(values["ffmpeg"], self.ffmpeg)
        self.assertEqual(values["ffprobe"], self.ffprobe)

    def test_explicit_ffmpeg_override_beats_path(self):
        path = self.config([(str(self.root / "source"), "One")], self.root / "out")
        with patch.object(settings.shutil, "which", side_effect=AssertionError("PATH lookup")):
            self.assertEqual(settings.load(path)["ffmpeg"], self.ffmpeg)

    def test_invalid_config_has_clear_errors(self):
        cases = [([], self.root / "out", "at least one"),
                 ([("", "One")], self.root / "out", "non-empty"),
                 ([(str(self.root / "source"), "")], self.root / "out", "label"),
                 ([(str(self.root / "source"), "One")], self.root / "source", "destination")]
        for sources, destination, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(settings.ConfigError, message):
                    settings.load(self.config(sources, destination))

    def test_destination_nested_in_source_rejected(self):
        source = self.root / "source"
        with self.assertRaisesRegex(settings.ConfigError, "inside a monitored source"):
            settings.load(self.config([(str(source), "One")], source / "out"))

    def test_nested_source_rejected(self):
        parent = self.root / "Meetings"
        child = parent / "Zoom"
        with self.assertRaisesRegex(settings.ConfigError, "source directories must not overlap"):
            settings.load(self.config([(str(parent), "Meetings"), (str(child), "Zoom")],
                                      self.root / "out"))

    def test_reverse_nested_source_rejected(self):
        parent = self.root / "Meetings"
        child = parent / "Zoom"
        with self.assertRaisesRegex(settings.ConfigError, "source directories must not overlap"):
            settings.load(self.config([(str(child), "Zoom"), (str(parent), "Meetings")],
                                      self.root / "out"))

    def test_sibling_sources_are_allowed(self):
        first = self.root / "Meetings"
        second = self.root / "Zoom"
        values = settings.load(self.config([(str(first), "Meetings"),
                                            (str(second), "Zoom")], self.root / "out"))
        self.assertEqual(values["sources"], (first, second))

    def test_missing_destination_rejected(self):
        path = self.root / "missing-destination.toml"
        path.write_text('[[sources]]\npath = "~/Meetings"\nlabel = "Meetings"\n')
        with self.assertRaisesRegex(settings.ConfigError, "destination"):
            settings.load(path)

    def test_explicit_backfill_only_and_repeat_safe(self):
        source = self.root / "source"; source.mkdir()
        destination = self.root / "out"
        media = source / "old.wav"; media.write_bytes(b"audio")
        os.utime(media, (100, 100))
        state = {"version": 2, "files": {}, "baseline_sources": []}
        calls = []
        def transcriber(path, model):
            calls.append(path)
            return {"segments": [{"start": 0, "end": 1, "text": " Speech."}]}
        args = dict(sources=(source,), destination=destination,
                    state_file=self.root / "state.json", model_dir=self.root,
                    ffprobe=self.ffprobe, transcriber=transcriber,
                    clock=lambda: 1000, stable_seconds=90)
        with patch.object(settings, "FFMPEG", self.ffmpeg), patch.object(
                settings, "SOURCE_LABELS", ("One",)), patch.object(
                settings, "DIARIZATION_ENABLED", False), patch.object(
                worker, "is_open", return_value=False), patch.object(
                worker, "has_audio"):
            self.assertEqual(worker.scan_once(state, {}, **args), 0)
            self.assertEqual(state["files"][str(media)]["status"], "baseline")
            self.assertEqual(worker.scan_once(state, {}, **args), 0)
            self.assertEqual(worker.scan_once(state, {}, backfill=True, **args), 1)
            original = (destination / "old [One].txt").read_bytes()
            self.assertEqual(worker.scan_once(state, {}, backfill=True, **args), 0)
            self.assertEqual((destination / "old [One].txt").read_bytes(), original)
        self.assertEqual(calls, [media])

    def test_plist_template_has_no_personal_paths(self):
        template = (Path(__file__).parents[1] / "local.transcriber.worker.plist").read_text()
        self.assertIn("__APP_DIR__", template)
        self.assertNotIn(str(Path.home()) + "/", template)
        self.assertNotIn("/Volumes/", template)


if __name__ == "__main__":
    unittest.main()
