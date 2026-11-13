"""Offline regression tests for the public moderation/translation pipeline.

Run with ``python3 -m unittest discover -s tests -v``.  ComfyUI imports are
stubbed and generation is scripted; this suite never imports torch or loads a
model.  The real node entry point, query builders, and JSON parsers still run.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch


NODE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"


def load_offline_nodes():
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = types.ModuleType("comfy.model_management")
    comfy.sd = types.ModuleType("comfy.sd")
    folder_paths = types.ModuleType("folder_paths")

    def unexpected_model_access(*args, **kwargs):
        raise AssertionError("Offline pipeline tests must not resolve or load a model")

    comfy.sd.load_clip = unexpected_model_access
    folder_paths.get_full_path = unexpected_model_access
    stubs = {
        "comfy": comfy,
        "comfy.model_management": comfy.model_management,
        "comfy.sd": comfy.sd,
        "folder_paths": folder_paths,
    }
    spec = importlib.util.spec_from_file_location("offline_moderation_nodes", NODE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import node from {NODE_PATH}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", stubs):
        spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    @staticmethod
    def decode(value):
        return value


class FakeClip:
    """Record real query order and return one JSON response per stage."""

    def __init__(self, *, blocked=False, languages=None, translation="A quiet garden.",
                 language_error=None, language_responses=None, category="NSFW"):
        self.tokenizer = FakeTokenizer()
        self.calls = []
        self.queries = []
        self.events = []
        self.language_responses = (list(language_responses)
                                   if language_responses is not None else None)
        self.responses = {
            "moderation": {
                "blocked": blocked,
                "categories": [category] if blocked else [],
            },
            "language": language_error if language_error is not None else {
                "languages": ["English"] if languages is None else languages,
            },
            "translation": {"translation": translation},
        }

    @staticmethod
    def tokenize(query):
        return query

    def generate(self, tokens, **kwargs):
        system_text = tokens.split("<|im_end|>", 1)[0]
        if "strict multilingual moderation classifier" in system_text:
            stage = "moderation"
        elif "Identify the actual natural language(s)" in system_text:
            stage = "language"
        elif "Professional translation engine" in system_text:
            stage = "translation"
        else:
            raise AssertionError("Unexpected generation query in offline test")
        self.calls.append(stage)
        self.queries.append((stage, tokens))
        self.events.append(("generate", stage))
        response = self.responses[stage]
        if stage == "language" and self.language_responses is not None:
            if not self.language_responses:
                raise AssertionError("Language detection exceeded the scripted retry bound")
            response = self.language_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, str):
            return response
        return json.dumps(response, ensure_ascii=False)


class ModerationPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = load_offline_nodes()

    def setUp(self):
        self.node = self.nodes.DetectNSFWText()
        # Keep tests independent of the user's hot-reloaded keyword files.
        self.keywords = patch.object(self.nodes, "_load_file_keywords", return_value=[])
        self.keywords.start()
        self.addCleanup(self.keywords.stop)
        self.logging = patch.object(self.nodes, "_log_error")
        self.logging.start()
        self.addCleanup(self.logging.stop)

    def test_blocked_input_is_moderated_before_language_validation(self):
        clip = FakeClip(
            blocked=True,
            language_error=self.nodes.UnsupportedLanguageError("Cannot detect short input"),
        )
        with self.assertRaises(self.nodes.NSFWDetectedError):
            self.node.check("nude", clip=clip, translate_to_english=True)
        self.assertEqual(clip.calls, ["moderation"])

    def test_safe_foreign_input_is_moderated_then_detected_then_translated(self):
        clip = FakeClip(languages=["French"], translation="A quiet garden.")
        value = self.node.check("Un jardin calme.", clip=clip, translate_to_english=True)
        self.assertEqual(clip.calls, ["moderation", "language", "translation"])
        self.assertEqual(value["result"][0], "A quiet garden.")
        self.assertFalse(json.loads(value["result"][1])["blocked"])
        self.assertEqual(value["ui"]["detected_languages"], ["French"])

    def test_safe_english_input_keeps_original_and_skips_translation(self):
        clip = FakeClip(languages=["English"])
        prompt = "A quiet garden."
        value = self.node.check(prompt, clip=clip, translate_to_english=True)
        self.assertEqual(clip.calls, ["moderation", "language"])
        self.assertEqual(value["result"][0], prompt)
        self.assertEqual(value["ui"]["translation_skipped"], [True])

    def test_unsupported_known_language_rejects_only_after_moderation(self):
        clip = FakeClip(language_responses=[
            {"languages": ["Swahili"]},
            {"languages": ["English"]},
        ])
        with self.assertRaises(self.nodes.UnsupportedLanguageError):
            self.node.check("Bustani tulivu wakati wa asubuhi.", clip=clip)
        self.assertEqual(clip.calls, ["moderation", "language"])

    def test_english_only_nonenglish_rejects_only_after_moderation(self):
        clip = FakeClip(languages=["Vietnamese"])
        with self.assertRaises(self.nodes.UnsupportedLanguageError):
            self.node.check("Một khu vườn yên tĩnh.", clip=clip,
                            language_filter="english_only")
        self.assertEqual(clip.calls, ["moderation", "language"])

    def test_mixed_supported_and_unsupported_languages_are_not_retried(self):
        clip = FakeClip(language_responses=[
            {"languages": ["English", "Hindi"]},
            {"languages": ["English"]},
        ])
        with self.assertRaises(self.nodes.UnsupportedLanguageError):
            self.node.check("A garden and unsupported text.", clip=clip)
        self.assertEqual(clip.calls, ["moderation", "language"])

    def test_language_generation_runtime_failure_is_not_retried(self):
        clip = FakeClip(language_error=RuntimeError("model execution failed"))
        with self.assertRaises(RuntimeError):
            self.node.check("A quiet garden.", clip=clip)
        self.assertEqual(clip.calls, ["moderation", "language"])

    def test_safe_unknown_language_attempts_strict_translation_when_enabled(self):
        clip = FakeClip(languages=[], translation="A quiet garden.")
        value = self.node.check("unclassified short token", clip=clip,
                                translate_to_english=True)
        self.assertEqual(clip.calls, ["moderation", "language", "language", "translation"])
        self.assertEqual(value["result"][0], "A quiet garden.")
        self.assertEqual(value["ui"]["detected_languages"], ["UNKNOWN"])
        self.assertEqual(value["ui"]["language_status"], ["UNKNOWN"])
        report = json.loads(value["result"][1])
        self.assertEqual(report["detected_languages"], ["UNKNOWN"])
        self.assertEqual(report["language_status"], "UNKNOWN")

    def test_safe_unknown_language_keeps_original_when_translation_disabled(self):
        clip = FakeClip(languages=[])
        prompt = "unclassified short token"
        value = self.node.check(prompt, clip=clip, translate_to_english=False)
        self.assertEqual(clip.calls, ["moderation", "language", "language"])
        self.assertEqual(value["result"][0], prompt)
        self.assertEqual(value["ui"]["translation_skipped"], [True])
        self.assertEqual(value["ui"]["detected_languages"], ["UNKNOWN"])

    def test_english_only_unknown_language_rejects_after_moderation(self):
        clip = FakeClip(languages=[])
        with self.assertRaises(self.nodes.UnsupportedLanguageError):
            self.node.check("unclassified short token", clip=clip,
                            language_filter="english_only")
        self.assertEqual(clip.calls, ["moderation", "language", "language"])

    def test_disabling_translation_does_not_skip_the_supported_language_gate(self):
        clip = FakeClip(languages=["Swahili"])
        with self.assertRaises(self.nodes.UnsupportedLanguageError):
            self.node.check("Bustani tulivu wakati wa asubuhi.", clip=clip,
                            translate_to_english=False)
        self.assertEqual(clip.calls, ["moderation", "language"])

    def test_slovene_alias_is_canonicalized_case_insensitively(self):
        clip = FakeClip(languages=[" sLoVeNe ", "Slovenian", " slovene "],
                        translation="A quiet garden.")
        value = self.node.check("Miren vrt.", clip=clip)
        self.assertEqual(clip.calls, ["moderation", "language", "translation"])
        self.assertEqual(value["ui"]["detected_languages"], ["Slovenian"])

    def test_invalid_language_responses_fail_only_after_moderation(self):
        invalid_responses = (
            "not JSON",
            '{"languages": ["English"], "languages": ["Hindi"]}',
            {"unexpected": []},
            {"languages": None},
            {"languages": "English"},
            {"languages": [123]},
            {"languages": [""]},
            {"languages": ["English"], "extra": True},
            ["English"],
        )
        for response in invalid_responses:
            with self.subTest(response=response):
                clip = FakeClip()
                clip.responses["language"] = response
                with self.assertRaises(self.nodes.UnsupportedLanguageError):
                    self.node.check("A quiet garden.", clip=clip)
                self.assertEqual(clip.calls, ["moderation", "language", "language"])

    def test_empty_language_retry_then_english_preserves_original(self):
        clip = FakeClip(language_responses=[
            {"languages": []},
            {"languages": ["English"]},
        ])
        prompt = "A quiet garden."
        value = self.node.check(prompt, clip=clip)
        self.assertEqual(clip.calls, ["moderation", "language", "language"])
        self.assertEqual(value["result"][0], prompt)
        self.assertEqual(value["ui"]["detected_languages"], ["English"])
        self.assertEqual(value["ui"]["translation_skipped"], [True])

    def test_invalid_language_retry_then_french_translates(self):
        for first_response in ("not JSON", {"languages": "French"}, {"unexpected": []}):
            with self.subTest(first_response=first_response):
                clip = FakeClip(language_responses=[
                    first_response,
                    {"languages": ["French"]},
                ], translation="A quiet garden.")
                value = self.node.check("Un jardin calme.", clip=clip)
                self.assertEqual(clip.calls,
                                 ["moderation", "language", "language", "translation"])
                self.assertEqual(value["result"][0], "A quiet garden.")
                self.assertEqual(value["ui"]["detected_languages"], ["French"])

    def test_two_malformed_language_responses_stop_at_retry_bound(self):
        clip = FakeClip(language_responses=["not JSON", {"languages": None}])
        with self.assertRaises(self.nodes.UnsupportedLanguageError):
            self.node.check("A quiet garden.", clip=clip)
        self.assertEqual(clip.calls, ["moderation", "language", "language"])
        self.assertEqual(clip.language_responses, [])

    def test_malformed_then_empty_language_does_not_erase_parser_error(self):
        clip = FakeClip(language_responses=["not JSON", {"languages": []}])
        with self.assertRaises(self.nodes.UnsupportedLanguageError):
            self.node.check("A quiet garden.", clip=clip)
        self.assertEqual(clip.calls, ["moderation", "language", "language"])

    def test_empty_then_malformed_language_does_not_translate_untrusted_output(self):
        clip = FakeClip(language_responses=[{"languages": []}, "not JSON"])
        with self.assertRaises(self.nodes.UnsupportedLanguageError):
            self.node.check("A quiet garden.", clip=clip)
        self.assertEqual(clip.calls, ["moderation", "language", "language"])

    def test_blocked_input_stops_even_with_english_only_filter(self):
        clip = FakeClip(blocked=True, languages=["Vietnamese"])
        with self.assertRaises(self.nodes.NSFWDetectedError):
            self.node.check("khỏa thân", clip=clip, language_filter="english_only")
        self.assertEqual(clip.calls, ["moderation"])

    def test_empty_input_rejects_without_generation_or_model_loading(self):
        for prompt in ("", "   ", "\n\t"):
            with self.subTest(prompt=repr(prompt)):
                clip = FakeClip()
                with self.assertRaises(ValueError):
                    self.node.check(prompt, clip=clip)
                self.assertEqual(clip.calls, [])
                with patch.object(self.nodes, "_load_clip") as load_clip:
                    with self.assertRaises(ValueError):
                        self.node.check(prompt)
                    load_clip.assert_not_called()

    def test_whitelist_match_cannot_bypass_real_moderation(self):
        clip = FakeClip(blocked=True)
        with self.assertRaises(self.nodes.NSFWDetectedError):
            self.node.check("nude", clip=clip, whitelist="nude")
        self.assertEqual(clip.calls, ["moderation"])
        self.assertIn("nude", clip.queries[0][1])

    def test_blacklist_match_still_calls_real_check_and_injects_keywords(self):
        clip = FakeClip(blocked=True, category="BLACKLIST")
        with self.assertRaises(self.nodes.NSFWDetectedError):
            self.nodes._moderate_prompt(
                self.nodes.DEFAULT_CLIP, clip=clip, prompt="violet lantern",
                blacklist="violet lantern", whitelist="violet lantern",
            )
        self.assertEqual(clip.calls, ["moderation"])
        self.assertIn("<blacklist>", clip.queries[0][1])
        self.assertIn("- violet lantern", clip.queries[0][1])

    def test_detect_stage_moderates_then_validates_language_before_translation(self):
        clip = FakeClip(languages=["French"])

        def on_stage(stage, **values):
            clip.events.append(("stage", stage, bool(values.get("complete"))))

        self.nodes._process_prompt("Un jardin calme.", clip=clip, on_stage=on_stage)
        events = clip.events
        self.assertLess(events.index(("stage", "detect", False)),
                        events.index(("generate", "moderation")))
        self.assertLess(events.index(("generate", "moderation")),
                        events.index(("generate", "language")))
        self.assertLess(events.index(("generate", "language")),
                        events.index(("stage", "detect", True)))
        self.assertLess(events.index(("stage", "detect", True)),
                        events.index(("stage", "translate", False)))
        self.assertLess(events.index(("stage", "translate", False)),
                        events.index(("generate", "translation")))
        self.assertLess(events.index(("generate", "translation")),
                        events.index(("stage", "translate", True)))


if __name__ == "__main__":
    unittest.main()
