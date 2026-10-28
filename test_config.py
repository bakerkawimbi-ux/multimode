"""Config validation tests. unittest.TestCase so they run with `python -m pytest`
or `python -m unittest discover -s tests -t .`. No network."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from core.config import ROOT, ConfigError, load_config, load_env, provider_status, validate


def _real() -> tuple[dict, dict, str]:
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text), json.loads((ROOT / "data" / "labels.json").read_text(encoding="utf-8")), text


class Base(unittest.TestCase):
    def setUp(self):
        self.raw, self.cat, self.text = _real()

    def errs(self, raw=None, cat=None, text=""):
        e, _ = validate(raw if raw is not None else self.raw, cat if cat is not None else self.cat, text)
        return e

    def assertHas(self, errors, fragment):
        self.assertTrue(any(fragment in e for e in errors), f"{fragment!r} not in {errors}")

    def raw2(self):
        return copy.deepcopy(self.raw)

    def cat2(self):
        return copy.deepcopy(self.cat)


class ValidConfig(Base):
    def test_real_config_is_valid(self):
        self.assertEqual(self.errs(text=self.text), [])

    def test_load_config_returns_parts(self):
        raw, cat, warnings = load_config()
        self.assertEqual(raw["project"]["name"], "poem-writer")
        self.assertEqual(set(cat["axes"]), {"scene", "mood"})
        self.assertIsInstance(warnings, list)

    def test_catalog_shape(self):
        scene, mood = self.cat["axes"]["scene"], self.cat["axes"]["mood"]
        self.assertEqual(len(scene["labels"]), 19)   # 17 real + unknown + blocked
        self.assertEqual(len(mood["labels"]), 9)     # 8 real + unknown
        self.assertEqual(scene["blocked_ids"], ["blocked/unsafe_content"])
        self.assertTrue(scene["uses_audio"] and not mood["uses_audio"])

    def test_every_scene_label_has_audio_prompts(self):
        for lab in self.cat["axes"]["scene"]["labels"]:
            if lab["id"] != "blocked/unsafe_content":
                self.assertGreaterEqual(len(lab["audio_prompts"]), 2, lab["id"])

    def test_cli_exit_code_zero(self):
        proc = subprocess.run([sys.executable, "-m", "core.config"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("CONFIG OK", proc.stdout)

    def test_cli_strict_fails_on_unverified_models(self):
        proc = subprocess.run([sys.executable, "-m", "core.config", "--strict"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("not verified", proc.stdout)

    def test_paid_provider_is_disabled_by_default(self):
        self.assertIs(self.raw["providers"]["anthropic_paid"]["enabled"], False)


class ConfigErrors(Base):
    def test_unknown_top_level_key(self):
        raw = self.raw2(); raw["surprise"] = 1
        self.assertHas(self.errs(raw), "unknown top-level key")

    def test_missing_section(self):
        raw = self.raw2(); del raw["poem"]
        self.assertHas(self.errs(raw), "poem: missing section")

    def test_secret_in_config_text(self):
        self.assertHas(self.errs(text="key: AIzaSyA1234567890abcdefghijKLMNOP"), "secret-looking")

    def test_negative_weight(self):
        raw = self.raw2(); raw["fusion"]["weights"]["audio"] = -0.1
        self.assertHas(self.errs(raw), "fusion.weights.audio")

    def test_weights_need_all_sources(self):
        raw = self.raw2(); del raw["fusion"]["weights"]["audio"]
        self.assertHas(self.errs(raw), "fusion.weights")

    def test_rank_scores_length(self):
        raw = self.raw2(); raw["fusion"]["rank_scores"] = [0.6, 0.4]
        self.assertHas(self.errs(raw), "must equal theme_scoring.top_k")

    def test_uncertain_above_min_confidence(self):
        raw = self.raw2(); raw["fusion"]["uncertain_below"] = 0.9
        self.assertHas(self.errs(raw), "uncertain_below must be <= min_confidence")

    def test_audio_longer_than_video_limit(self):
        raw = self.raw2(); raw["media_prep"]["audio"]["max_seconds"] = 120
        self.assertHas(self.errs(raw), "audio.max_seconds must be <=")

    def test_bad_pooling(self):
        raw = self.raw2(); raw["media_prep"]["video"]["pooling"] = "median"
        self.assertHas(self.errs(raw), "pooling")

    def test_probe_target_below_zero_shot(self):
        raw = self.raw2(); raw["targets"]["scene_top1_probe"] = 0.5
        self.assertHas(self.errs(raw), "ladder only climbs")

    def test_missing_target(self):
        raw = self.raw2(); del raw["targets"]["p95_latency_video_s"]
        self.assertHas(self.errs(raw), "p95_latency_video_s")

    def test_score_target_out_of_range(self):
        raw = self.raw2(); raw["targets"]["human_quality_mean"] = 7
        self.assertHas(self.errs(raw), "human_quality_mean")

    def test_default_form_must_exist(self):
        raw = self.raw2(); raw["poem"]["default_form"] = "ode"
        self.assertHas(self.errs(raw), "not defined under poem.forms")

    def test_form_max_below_min(self):
        raw = self.raw2(); raw["poem"]["forms"]["free_verse"]["max_lines"] = 2
        self.assertHas(self.errs(raw), "max_lines must be >= min_lines")

    def test_syllable_pattern_length(self):
        raw = self.raw2(); raw["poem"]["forms"]["haiku"]["syllable_pattern"] = [5, 7]
        self.assertHas(self.errs(raw), "syllable_pattern: length")

    def test_unknown_form_key(self):
        raw = self.raw2(); raw["poem"]["forms"]["haiku"]["rhyme"] = "abab"
        self.assertHas(self.errs(raw), "unknown key")

    def test_copied_ngram_too_small(self):
        raw = self.raw2(); raw["poem"]["checks"]["max_copied_ngram"] = 1
        self.assertHas(self.errs(raw), "max_copied_ngram")

    def test_invalid_safety_regex(self):
        raw = self.raw2(); raw["safety"]["blocked_request_patterns"].append("([unclosed")
        self.assertHas(self.errs(raw), "invalid regex")

    def test_chain_undefined_provider(self):
        raw = self.raw2(); raw["agents"]["image"]["providers"].append("nope")
        self.assertHas(self.errs(raw), "'nope' is not defined")

    def test_chain_wrong_type(self):
        raw = self.raw2(); raw["agents"]["audio"]["providers"] = ["local_clip"]
        self.assertHas(self.errs(raw), "allowed here")

    def test_generator_chain_tier_order(self):
        raw = self.raw2(); raw["generators"]["providers"] = ["colab_poet", "gemini_poet"]
        self.assertHas(self.errs(raw), "breaks the order")

    def test_generator_requires_tier(self):
        raw = self.raw2(); del raw["providers"]["groq_poet"]["tier"]
        self.assertHas(self.errs(raw), "needs tier")

    def test_paid_provider_needs_explicit_enabled(self):
        raw = self.raw2(); del raw["providers"]["anthropic_paid"]["enabled"]
        self.assertHas(self.errs(raw), "must set 'enabled' explicitly")

    def test_generators_on_all_fail_value(self):
        raw = self.raw2(); raw["generators"]["on_all_fail"] = "raise"
        self.assertHas(self.errs(raw), "on_all_fail")

    def test_llm_provider_requires_model_and_key(self):
        raw = self.raw2()
        del raw["providers"]["gemini_poet"]["model"]
        del raw["providers"]["groq_poet"]["env_key"]
        e = self.errs(raw)
        self.assertHas(e, "requires 'model'")
        self.assertHas(e, "requires 'env_key'")

    def test_custom_http_requires_endpoint(self):
        raw = self.raw2(); raw["providers"]["colab_image"]["options"] = {}
        self.assertHas(self.errs(raw), "endpoint")

    def test_custom_http_requires_url_env(self):
        raw = self.raw2(); del raw["providers"]["colab_audio"]["base_url_env"]
        self.assertHas(self.errs(raw), "custom_http requires")

    def test_bad_env_name(self):
        raw = self.raw2(); raw["providers"]["groq_poet"]["env_key"] = "my key"
        self.assertHas(self.errs(raw), "env var name")

    def test_bad_verified_date(self):
        raw = self.raw2(); raw["providers"]["groq_poet"]["model_verified_on"] = "yesterday"
        self.assertHas(self.errs(raw), "YYYY-MM-DD")

    def test_unknown_provider_key(self):
        raw = self.raw2(); raw["providers"]["groq_poet"]["api_key"] = "x"
        self.assertHas(self.errs(raw), "unknown key")

    def test_extensions_need_all_kinds(self):
        raw = self.raw2(); del raw["limits"]["allowed_extensions"]["video"]
        self.assertHas(self.errs(raw), "allowed_extensions")

    def test_bool_is_not_a_number(self):
        raw = self.raw2(); raw["open_set"]["scene"]["reject_below"] = True
        self.assertHas(self.errs(raw), "open_set.scene.reject_below")

    def test_privacy_flag_must_be_bool(self):
        raw = self.raw2(); raw["privacy"]["allow_hosted_vision"] = "no"
        self.assertHas(self.errs(raw), "allow_hosted_vision")


class CatalogErrors(Base):
    def test_unknown_label_missing(self):
        cat = self.cat2(); ax = cat["axes"]["mood"]
        ax["labels"] = [x for x in ax["labels"] if x["id"] != "other/unknown"]
        self.assertHas(self.errs(cat=cat), "not in the label list")

    def test_blocked_id_must_exist(self):
        cat = self.cat2(); cat["axes"]["mood"]["blocked_ids"] = ["blocked/nope"]
        self.assertHas(self.errs(cat=cat), "blocked id")

    def test_duplicate_id(self):
        cat = self.cat2(); ax = cat["axes"]["scene"]; ax["labels"].append(copy.deepcopy(ax["labels"][0]))
        self.assertHas(self.errs(cat=cat), "duplicate id")

    def test_too_few_prompts(self):
        cat = self.cat2(); lab = cat["axes"]["scene"]["labels"][0]; lab["prompts"] = lab["prompts"][:2]
        self.assertHas(self.errs(cat=cat), "at least 3")

    def test_duplicate_prompts(self):
        cat = self.cat2(); p = cat["axes"]["mood"]["labels"][0]["prompts"]; p[1] = p[0]
        self.assertHas(self.errs(cat=cat), "duplicate prompts")

    def test_bad_id_format(self):
        cat = self.cat2(); cat["axes"]["scene"]["labels"][0]["id"] = "Forest"
        self.assertHas(self.errs(cat=cat), "family/name")

    def test_family_must_match_prefix(self):
        cat = self.cat2(); cat["axes"]["scene"]["labels"][0]["family"] = "wrong"
        self.assertHas(self.errs(cat=cat), "family")

    def test_too_few_anchors(self):
        cat = self.cat2(); cat["axes"]["scene"]["labels"][0]["anchors"] = ["tree"]
        self.assertHas(self.errs(cat=cat), "at least 4 anchor")

    def test_anchor_must_be_single_lowercase_word(self):
        cat = self.cat2(); cat["axes"]["scene"]["labels"][0]["anchors"][0] = "Tall Tree"
        self.assertHas(self.errs(cat=cat), "lowercase single words")

    def test_unknown_must_not_have_anchors(self):
        cat = self.cat2()
        for lab in cat["axes"]["mood"]["labels"]:
            if lab["id"] == "other/unknown":
                lab["anchors"] = ["none"]
        self.assertHas(self.errs(cat=cat), "no anchors")

    def test_audio_prompts_required_on_audio_axis(self):
        cat = self.cat2(); del cat["axes"]["scene"]["labels"][0]["audio_prompts"]
        self.assertHas(self.errs(cat=cat), "audio_prompts")

    def test_audio_prompts_forbidden_on_non_audio_axis(self):
        cat = self.cat2(); cat["axes"]["mood"]["labels"][0]["audio_prompts"] = ["calm music", "soft hum"]
        self.assertHas(self.errs(cat=cat), "uses_audio=false")

    def test_axes_must_be_scene_and_mood(self):
        cat = self.cat2(); del cat["axes"]["mood"]
        self.assertHas(self.errs(cat=cat), "labels.axes")

    def test_uncertain_outcome_required(self):
        cat = self.cat2(); cat["uncertain_outcome"] = "maybe"
        self.assertHas(self.errs(cat=cat), "uncertain_outcome")

    def test_need_enough_real_labels(self):
        cat = self.cat2(); ax = cat["axes"]["mood"]; ax["labels"] = ax["labels"][:3] + ax["labels"][-1:]
        self.assertHas(self.errs(cat=cat), "at least 6 real labels")


class EnvAndStatus(Base):
    def test_load_env_parses_comments_and_blanks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text('# c\nGROQ_API_KEY="abc"\nEMPTY=\nNOEQ\nANTHROPIC_API_KEY=   # optional\nA=1\n', encoding="utf-8")
            self.assertEqual(load_env(p), {"GROQ_API_KEY": "abc", "A": "1"})

    def test_load_env_missing_file(self):
        self.assertEqual(load_env(Path("/nonexistent/.env")), {})

    def test_status_without_keys(self):
        st = provider_status(self.raw, {})
        self.assertEqual(st["local_parsers"], "ready")
        self.assertEqual(st["anthropic_paid"], "disabled")
        self.assertIn("allow_hosted_vision", st["vision_llm_gemini"])
        self.assertIn("GROQ_API_KEY", st["groq_poet"])
        self.assertTrue(st["colab_poet"].startswith("skipped"))

    def test_status_ready_with_keys(self):
        st = provider_status(self.raw, {"GROQ_API_KEY": "x", "COLAB_TOKEN": "t", "COLAB_SERVER_URL": "http://h"})
        self.assertEqual(st["groq_poet"], "ready")
        self.assertEqual(st["colab_poet"], "ready")

    def test_vision_llm_ready_only_when_allowed_and_keyed(self):
        raw = self.raw2(); raw["privacy"]["allow_hosted_vision"] = True
        self.assertEqual(provider_status(raw, {"GEMINI_API_KEY": "x"})["vision_llm_gemini"], "ready")


class Warnings(Base):
    def test_unverified_models_warn(self):
        _, w = validate(self.raw, self.cat, "")
        self.assertTrue(any("gemini_poet" in x for x in w))

    def test_disabled_provider_does_not_warn(self):
        _, w = validate(self.raw, self.cat, "")
        self.assertFalse(any("anthropic_paid" in x for x in w))

    def test_verified_model_does_not_warn(self):
        raw = self.raw2(); raw["providers"]["groq_poet"]["model_verified_on"] = "2026-10-28"
        _, w = validate(raw, self.cat, "")
        self.assertFalse(any("providers.groq_poet" in x for x in w))

    def test_enabled_paid_provider_warns_about_cost(self):
        raw = self.raw2(); raw["providers"]["anthropic_paid"]["enabled"] = True
        _, w = validate(raw, self.cat, "")
        self.assertTrue(any("$0 run" in x for x in w))


class LoadErrors(Base):
    def test_load_config_raises_with_all_errors(self):
        with tempfile.TemporaryDirectory() as d:
            raw = self.raw2()
            raw["fusion"]["weights"]["image"] = -1
            raw["paths"]["labels"] = "data/missing.json"
            p = Path(d) / "config.yaml"
            p.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(ConfigError) as ctx:
                load_config(p)
            joined = "\n".join(ctx.exception.errors)
            self.assertIn("file not found", joined)
            self.assertIn("fusion.weights.image", joined)


if __name__ == "__main__":
    unittest.main()
