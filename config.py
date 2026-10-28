"""Config and label-catalog loading and validation.

Run:  python -m core.config [--config PATH] [--env PATH] [--strict]

Depends only on PyYAML and the standard library, so validation works anywhere.
Exit code 0 = valid, 1 = errors (or warnings with --strict).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ("image", "video", "audio")           # evidence sources for fusion
INPUT_KINDS = ("image", "video", "document")    # uploaded file kinds
AXES = ("scene", "mood")
VERDICTS = ("MATCH", "PARTIAL", "CONFLICT", "SINGLE_MODALITY", "UNCERTAIN")
TIERS = {"free_hosted": 0, "self_hosted": 1, "paid": 2}
PROVIDER_TYPES = {
    "local_parser", "passthrough", "custom_http", "local_clip", "local_clap", "vision_llm", "llm_gen",
}
CHAIN_ALLOWED_TYPES = {
    "agents.image": {"custom_http", "local_clip", "vision_llm"},
    "agents.video": {"custom_http", "local_clip"},
    "agents.audio": {"custom_http", "local_clap"},
    "agents.document": {"local_parser"},
    "agents.text": {"passthrough"},
    "generators": {"llm_gen", "custom_http"},
}
PROVIDER_KEYS = {
    "type", "tier", "enabled", "model", "model_verified_on", "env_key", "base_url_env",
    "timeout_s", "cooldown_s", "rpm", "daily_soft_limit", "options",
}
TOP_LEVEL_KEYS = {
    "project", "paths", "limits", "media_prep", "theme_scoring", "open_set", "fusion", "poem",
    "safety", "targets", "cache", "retry", "providers", "agents", "generators", "api", "privacy",
}
TARGET_KINDS = {  # key -> ("ratio" | "score" | "seconds")
    "scene_top1_zero_shot": "ratio", "scene_top1_probe": "ratio", "video_scene_top1": "ratio",
    "audio_ambience_top1": "ratio", "mood_top1_probe": "ratio", "unknown_rejection": "ratio",
    "hallucination_max": "ratio", "mismatch_conflict": "ratio", "form_compliance": "ratio",
    "anchor_grounded_rate": "ratio", "clip_pair_win_rate": "ratio", "blocked_prompt_refusal": "ratio",
    "human_relevance_mean": "score", "human_quality_mean": "score",
    "p95_latency_image_s": "seconds", "p95_latency_video_s": "seconds", "p95_cache_hit_s": "seconds",
}
SECRET_PATTERNS = [
    re.compile(p)
    for p in (
        r"AIza[0-9A-Za-z_\-]{20,}",
        r"\bsk-[A-Za-z0-9\-]{16,}",
        r"\bgsk_[A-Za-z0-9]{16,}",
        r"\bhf_[A-Za-z0-9]{16,}",
        r"Bearer\s+[A-Za-z0-9._\-]{16,}",
    )
]
LABEL_ID = re.compile(r"^[a-z_]+/[a-z0-9_]+$")
ANCHOR = re.compile(r"^[a-z]+$")
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FORM_KEYS = {"min_lines", "max_lines", "syllable_pattern", "syllable_tolerance", "lines_multiple_of"}


class ConfigError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("invalid configuration:\n  - " + "\n  - ".join(errors))
        self.errors = errors


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


class _Validator:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def section(self, raw: Any, key: str, allowed: set[str] | None = None) -> dict:
        s = raw.get(key) if isinstance(raw, dict) else None
        if not isinstance(s, dict):
            self.err(f"{key}: missing or not a mapping")
            return {}
        if allowed is not None:
            for k in s:
                if k not in allowed:
                    self.err(f"{key}.{k}: unknown key")
        return s

    def num(self, d: dict, key: str, path: str, lo=None, hi=None, integer=False,
            lo_open=False, required=True):
        if key not in d or d[key] is None:
            if required:
                self.err(f"{path}.{key}: missing")
            return None
        v = d[key]
        if not (_is_int(v) if integer else _is_num(v)):
            self.err(f"{path}.{key}: expected {'an integer' if integer else 'a number'}, got {v!r}")
            return None
        if lo is not None and (v <= lo if lo_open else v < lo):
            self.err(f"{path}.{key}: must be {'>' if lo_open else '>='} {lo}, got {v}")
        if hi is not None and v > hi:
            self.err(f"{path}.{key}: must be <= {hi}, got {v}")
        return v

    def string(self, d: dict, key: str, path: str, required=True, choices=None):
        if key not in d or d[key] is None:
            if required:
                self.err(f"{path}.{key}: missing")
            return None
        v = d[key]
        if not isinstance(v, str) or not v.strip():
            self.err(f"{path}.{key}: expected a non-empty string, got {v!r}")
            return None
        if choices is not None and v not in choices:
            self.err(f"{path}.{key}: must be one of {sorted(choices)}, got {v!r}")
        return v

    def boolean(self, d: dict, key: str, path: str):
        if key not in d:
            self.err(f"{path}.{key}: missing")
            return None
        if not isinstance(d[key], bool):
            self.err(f"{path}.{key}: expected true/false, got {d[key]!r}")
            return None
        return d[key]


def _check_chain(v: _Validator, name: str, chain: Any, providers: dict, tiered: bool = False) -> list[str]:
    names = chain.get("providers") if isinstance(chain, dict) else None
    if not isinstance(names, list) or not names:
        v.err(f"{name}.providers: must be a non-empty list")
        return []
    if len(set(names)) != len(names):
        v.err(f"{name}.providers: duplicate entries")
    allowed = CHAIN_ALLOWED_TYPES[name]
    last_tier = -1
    for p in names:
        if p not in providers or not isinstance(providers[p], dict):
            v.err(f"{name}.providers: '{p}' is not defined under providers")
            continue
        cfg = providers[p]
        if cfg.get("type") not in allowed:
            v.err(f"{name}.providers: '{p}' has type {cfg.get('type')!r}; allowed here: {sorted(allowed)}")
        if tiered:
            tier = cfg.get("tier")
            if tier not in TIERS:
                v.err(f"{name}.providers: '{p}' needs tier in {list(TIERS)}")
            else:
                if TIERS[tier] < last_tier:
                    v.err(f"{name}.providers: '{p}' breaks the order free_hosted -> self_hosted -> paid")
                last_tier = max(last_tier, TIERS[tier])
                if tier == "paid" and not isinstance(cfg.get("enabled"), bool):
                    v.err(f"providers.{p}: a paid provider must set 'enabled' explicitly (opt-in spend)")
    return names


def validate(raw: dict, catalog: dict | None, secret_text: str = "") -> tuple[list[str], list[str]]:
    """Validate a parsed config (and optional parsed label catalog). Returns (errors, warnings)."""
    v = _Validator()
    if not isinstance(raw, dict):
        return ["config root must be a mapping"], []
    for k in raw:
        if k not in TOP_LEVEL_KEYS:
            v.err(f"{k}: unknown top-level key")
    for k in sorted(TOP_LEVEL_KEYS - set(raw)):
        v.err(f"{k}: missing section")

    for pat in SECRET_PATTERNS:
        if pat.search(secret_text):
            v.err("secret-looking value found in config; keep keys in .env only")
            break

    pj = v.section(raw, "project", {"name", "language", "version"})
    for k in ("name", "language", "version"):
        v.string(pj, k, "project")
    pa = v.section(raw, "paths", {"labels", "cache_dir", "upload_dir", "eval_results", "golden_set", "log_dir"})
    for k in ("labels", "cache_dir", "upload_dir", "eval_results", "golden_set", "log_dir"):
        v.string(pa, k, "paths")

    li = v.section(raw, "limits", {"max_image_mb", "max_video_mb", "max_document_mb", "max_video_seconds",
                                   "max_user_note_chars", "max_document_chars", "allowed_extensions"})
    for k in ("max_image_mb", "max_video_mb", "max_document_mb", "max_video_seconds",
              "max_user_note_chars", "max_document_chars"):
        v.num(li, k, "limits", lo=1, integer=True)
    ext = li.get("allowed_extensions")
    if not isinstance(ext, dict) or set(ext) != set(INPUT_KINDS):
        v.err(f"limits.allowed_extensions: must have exactly the keys {list(INPUT_KINDS)}")
    else:
        for m, lst in ext.items():
            if not (isinstance(lst, list) and lst and all(isinstance(e, str) and e.startswith(".") for e in lst)):
                v.err(f"limits.allowed_extensions.{m}: must be a non-empty list of strings like '.png'")

    mp = v.section(raw, "media_prep", {"image", "video", "audio", "document"})
    im = v.section(mp, "image", {"downscale_max_side", "min_side", "blur_laplacian_min", "dark_mean_luma_min"})
    big = v.num(im, "downscale_max_side", "media_prep.image", lo=256, integer=True)
    small = v.num(im, "min_side", "media_prep.image", lo=32, integer=True)
    v.num(im, "blur_laplacian_min", "media_prep.image", lo=0)
    v.num(im, "dark_mean_luma_min", "media_prep.image", lo=0, hi=255)
    if big is not None and small is not None and small >= big:
        v.err("media_prep.image: min_side must be smaller than downscale_max_side")
    vi = v.section(mp, "video", {"sample_frames", "shaky_flow_max", "dark_mean_luma_min", "extract_audio",
                                 "segment_change_threshold", "pooling"})
    v.num(vi, "sample_frames", "media_prep.video", lo=2, hi=64, integer=True)
    v.num(vi, "shaky_flow_max", "media_prep.video", lo=0, lo_open=True)
    v.num(vi, "dark_mean_luma_min", "media_prep.video", lo=0, hi=255)
    v.boolean(vi, "extract_audio", "media_prep.video")
    v.num(vi, "segment_change_threshold", "media_prep.video", lo=0, hi=1)
    v.string(vi, "pooling", "media_prep.video", choices={"mean", "max"})
    au = v.section(mp, "audio", {"sample_rate", "max_seconds", "silence_rms_min", "noisy_snr_below_db"})
    v.num(au, "sample_rate", "media_prep.audio", lo=8000, integer=True)
    a_sec = v.num(au, "max_seconds", "media_prep.audio", lo=1, integer=True)
    v.num(au, "silence_rms_min", "media_prep.audio", lo=0, hi=1)
    v.num(au, "noisy_snr_below_db", "media_prep.audio")
    v_sec = li.get("max_video_seconds")
    if a_sec is not None and _is_int(v_sec) and a_sec > v_sec:
        v.err("media_prep.audio.max_seconds must be <= limits.max_video_seconds (audio comes from the video)")
    dc = v.section(mp, "document", {"min_chars"})
    v.num(dc, "min_chars", "media_prep.document", lo=0, integer=True)

    ts = v.section(raw, "theme_scoring", {"top_k", "softmax_temperature"})
    top_k = v.num(ts, "top_k", "theme_scoring", lo=1, hi=10, integer=True)
    v.num(ts, "softmax_temperature", "theme_scoring", lo=0, lo_open=True)

    op = v.section(raw, "open_set", set(AXES))
    for axis in AXES:
        ax = v.section(op, axis, {"reject_below", "margin_min"})
        v.num(ax, "reject_below", f"open_set.{axis}", lo=0, hi=1)
        v.num(ax, "margin_min", f"open_set.{axis}", lo=0, hi=1)

    fu = v.section(raw, "fusion", {"min_confidence", "uncertain_below", "weights", "agreement_bonus",
                                   "rank_scores", "secondary_min"})
    minc = v.num(fu, "min_confidence", "fusion", lo=0, hi=1)
    unc = v.num(fu, "uncertain_below", "fusion", lo=0, hi=1)
    if minc is not None and unc is not None and unc > minc:
        v.err("fusion: uncertain_below must be <= min_confidence")
    v.num(fu, "secondary_min", "fusion", lo=0, hi=1)
    w = fu.get("weights")
    if not isinstance(w, dict) or set(w) != set(SOURCES):
        v.err(f"fusion.weights: must have exactly the keys {list(SOURCES)}")
    else:
        for m, x in w.items():
            if not _is_num(x) or x < 0:
                v.err(f"fusion.weights.{m}: must be a number >= 0, got {x!r}")
        if all(_is_num(x) for x in w.values()) and sum(w.values()) <= 0:
            v.err("fusion.weights: at least one weight must be > 0")
    b = fu.get("agreement_bonus")
    if not isinstance(b, dict) or set(b) != set(VERDICTS):
        v.err(f"fusion.agreement_bonus: must have exactly the keys {list(VERDICTS)}")
    else:
        for k, x in b.items():
            if not _is_num(x) or not 0 <= x <= 1:
                v.err(f"fusion.agreement_bonus.{k}: must be a number in [0, 1], got {x!r}")
    rs = fu.get("rank_scores")
    if not isinstance(rs, list) or not rs or not all(_is_num(x) and 0 < x <= 1 for x in rs):
        v.err("fusion.rank_scores: must be a non-empty list of numbers in (0, 1]")
    else:
        if any(a < c for a, c in zip(rs, rs[1:])):
            v.err("fusion.rank_scores: must be non-increasing")
        if sum(rs) > 1.0 + 1e-9:
            v.err("fusion.rank_scores: must sum to <= 1")
        if top_k is not None and len(rs) != top_k:
            v.err(f"fusion.rank_scores: length {len(rs)} must equal theme_scoring.top_k ({top_k})")

    po = v.section(raw, "poem", {"default_form", "seed", "max_attempts", "forms", "checks", "ai_label"})
    forms = po.get("forms")
    if not isinstance(forms, dict) or not forms:
        v.err("poem.forms: must be a non-empty mapping")
        forms = {}
    for fname, f in forms.items():
        fp = f"poem.forms.{fname}"
        if not isinstance(f, dict):
            v.err(f"{fp}: must be a mapping")
            continue
        for k in f:
            if k not in FORM_KEYS:
                v.err(f"{fp}.{k}: unknown key")
        lo_l = v.num(f, "min_lines", fp, lo=1, integer=True)
        hi_l = v.num(f, "max_lines", fp, lo=1, integer=True)
        if lo_l is not None and hi_l is not None and hi_l < lo_l:
            v.err(f"{fp}: max_lines must be >= min_lines")
        v.num(f, "syllable_tolerance", fp, lo=0, hi=3, integer=True, required=False)
        v.num(f, "lines_multiple_of", fp, lo=1, integer=True, required=False)
        pat = f.get("syllable_pattern")
        if pat is not None:
            if not (isinstance(pat, list) and pat and all(_is_int(x) and x >= 1 for x in pat)):
                v.err(f"{fp}.syllable_pattern: must be a list of positive integers")
            elif not (lo_l == hi_l == len(pat)):
                v.err(f"{fp}.syllable_pattern: length must equal min_lines and max_lines")
    df = v.string(po, "default_form", "poem")
    if df is not None and forms and df not in forms:
        v.err(f"poem.default_form: '{df}' is not defined under poem.forms")
    v.num(po, "seed", "poem", integer=True)
    v.num(po, "max_attempts", "poem", lo=1, hi=5, integer=True)
    v.string(po, "ai_label", "poem")
    ck = v.section(po, "checks", {"anchor_min_hits", "max_copied_ngram"})
    v.num(ck, "anchor_min_hits", "poem.checks", lo=0, integer=True)
    v.num(ck, "max_copied_ngram", "poem.checks", lo=3, integer=True)

    sa = v.section(raw, "safety", {"refuse_real_person_naming", "blocked_request_patterns"})
    v.boolean(sa, "refuse_real_person_naming", "safety")
    pats = sa.get("blocked_request_patterns")
    if not isinstance(pats, list) or not pats or not all(isinstance(p, str) for p in pats):
        v.err("safety.blocked_request_patterns: must be a non-empty list of regex strings")
    else:
        for p in pats:
            try:
                re.compile(p, re.IGNORECASE)
            except re.error as exc:
                v.err(f"safety.blocked_request_patterns: invalid regex {p!r}: {exc}")

    tg = v.section(raw, "targets", set(TARGET_KINDS))
    for k, kind in TARGET_KINDS.items():
        if kind == "ratio":
            v.num(tg, k, "targets", lo=0, hi=1)
        elif kind == "score":
            v.num(tg, k, "targets", lo=1, hi=5)
        else:
            v.num(tg, k, "targets", lo=0, lo_open=True)
    z, p_ = tg.get("scene_top1_zero_shot"), tg.get("scene_top1_probe")
    if _is_num(z) and _is_num(p_) and p_ < z:
        v.err("targets: scene_top1_probe must be >= scene_top1_zero_shot (the ladder only climbs if it helps)")

    ca = v.section(raw, "cache", {"enabled", "ttl_days", "store_inputs"})
    v.boolean(ca, "enabled", "cache")
    v.num(ca, "ttl_days", "cache", lo=1, integer=True)
    v.boolean(ca, "store_inputs", "cache")

    rt = v.section(raw, "retry", {"max_retries", "base_delay_s", "max_delay_s", "jitter"})
    v.num(rt, "max_retries", "retry", lo=0, integer=True)
    base = v.num(rt, "base_delay_s", "retry", lo=0, lo_open=True)
    mx = v.num(rt, "max_delay_s", "retry", lo=0, lo_open=True)
    v.boolean(rt, "jitter", "retry")
    if base is not None and mx is not None and mx < base:
        v.err("retry: max_delay_s must be >= base_delay_s")

    providers = raw.get("providers")
    if not isinstance(providers, dict) or not providers:
        v.err("providers: missing or empty")
        providers = {}
    for name, p in providers.items():
        path = f"providers.{name}"
        if not isinstance(p, dict):
            v.err(f"{path}: must be a mapping")
            continue
        for k in p:
            if k not in PROVIDER_KEYS:
                v.err(f"{path}.{k}: unknown key")
        ptype = v.string(p, "type", path, choices=PROVIDER_TYPES)
        v.num(p, "timeout_s", path, lo=0, lo_open=True)
        v.num(p, "cooldown_s", path, lo=0, required=False)
        v.num(p, "rpm", path, lo=1, integer=True, required=False)
        v.num(p, "daily_soft_limit", path, lo=1, integer=True, required=False)
        v.string(p, "tier", path, required=False, choices=set(TIERS))
        if "enabled" in p and not isinstance(p["enabled"], bool):
            v.err(f"{path}.enabled: expected true/false")
        for k in ("env_key", "base_url_env"):
            val = v.string(p, k, path, required=False)
            if val is not None and not ENV_NAME.match(val):
                v.err(f"{path}.{k}: must look like an env var name (A-Z, 0-9, _), got {val!r}")
        vd = p.get("model_verified_on")
        if vd is not None and not (isinstance(vd, str) and DATE.match(vd)):
            v.err(f"{path}.model_verified_on: must be null or 'YYYY-MM-DD'")
        opts = p.get("options")
        if "options" in p and not isinstance(opts, dict):
            v.err(f"{path}.options: must be a mapping")
        if ptype in {"vision_llm", "llm_gen", "local_clip", "local_clap"} and not p.get("model"):
            v.err(f"{path}: type {ptype} requires 'model'")
        if ptype in {"vision_llm", "llm_gen"} and not p.get("env_key"):
            v.err(f"{path}: type {ptype} requires 'env_key'")
        if ptype == "custom_http":
            if not (p.get("env_key") and p.get("base_url_env")):
                v.err(f"{path}: type custom_http requires 'env_key' and 'base_url_env'")
            ep = opts.get("endpoint") if isinstance(opts, dict) else None
            if not (isinstance(ep, str) and ep.startswith("/")):
                v.err(f"{path}.options.endpoint: custom_http needs an endpoint starting with '/'")
        if p.get("model") and not p.get("model_verified_on") and p.get("enabled", True):
            v.warnings.append(f"{path}: model '{p['model']}' is not verified (model_verified_on is null); "
                              "confirm the id exists before relying on it")

    ag = v.section(raw, "agents", {"image", "video", "audio", "document", "text"})
    for m in ("image", "video", "audio", "document", "text"):
        _check_chain(v, f"agents.{m}", ag.get(m), providers)

    ge = v.section(raw, "generators", {"providers", "on_all_fail"})
    chain = _check_chain(v, "generators", ge, providers, tiered=True)
    v.string(ge, "on_all_fail", "generators", choices={"unavailable"})
    for name in chain:
        p = providers.get(name)
        if isinstance(p, dict) and p.get("tier") == "paid" and p.get("enabled") is True:
            v.warnings.append(f"providers.{name}: paid provider is enabled; a $0 run is no longer guaranteed")

    api = v.section(raw, "api", {"host", "port", "cors_origins", "mock_mode"})
    v.string(api, "host", "api")
    v.num(api, "port", "api", lo=1, hi=65535, integer=True)
    co = api.get("cors_origins")
    if not (isinstance(co, list) and all(isinstance(x, str) for x in co)):
        v.err("api.cors_origins: must be a list of strings")
    v.boolean(api, "mock_mode", "api")

    pr = v.section(raw, "privacy", {"hosted_llm_notice", "send_context_to_hosted_llm", "allow_hosted_vision"})
    for k in ("hosted_llm_notice", "send_context_to_hosted_llm", "allow_hosted_vision"):
        v.boolean(pr, k, "privacy")

    if catalog is not None:
        _validate_catalog(v, catalog)
    return v.errors, v.warnings


def _validate_catalog(v: _Validator, cat: Any) -> None:
    if not isinstance(cat, dict):
        v.err("labels: root must be a mapping")
        return
    if cat.get("uncertain_outcome") != "UNCERTAIN":
        v.err("labels.uncertain_outcome: must be 'UNCERTAIN'")
    axes = cat.get("axes")
    if not isinstance(axes, dict) or set(axes) != set(AXES):
        v.err(f"labels.axes: must have exactly the keys {list(AXES)}")
        return
    seen_global: set[tuple[str, str]] = set()
    for axis, ax in axes.items():
        ap = f"labels.axes.{axis}"
        if not isinstance(ax, dict):
            v.err(f"{ap}: must be a mapping")
            continue
        unknown_id = ax.get("unknown_id")
        blocked = ax.get("blocked_ids")
        uses_audio = ax.get("uses_audio")
        labels = ax.get("labels")
        if not isinstance(unknown_id, str) or not unknown_id:
            v.err(f"{ap}.unknown_id: missing")
        if not (isinstance(blocked, list) and all(isinstance(b, str) for b in blocked)):
            v.err(f"{ap}.blocked_ids: must be a list of strings")
            blocked = []
        if not isinstance(uses_audio, bool):
            v.err(f"{ap}.uses_audio: must be true/false")
        if not isinstance(labels, list) or not labels:
            v.err(f"{ap}.labels: must be a non-empty list")
            continue
        seen: set[str] = set()
        for i, lab in enumerate(labels):
            lp = f"{ap}.labels[{i}]"
            if not isinstance(lab, dict):
                v.err(f"{lp}: must be a mapping")
                continue
            lid = lab.get("id")
            if not isinstance(lid, str) or not LABEL_ID.match(lid):
                v.err(f"{lp}.id: must look like 'family/name', got {lid!r}")
                continue
            if lid in seen:
                v.err(f"{ap}: duplicate id {lid!r}")
            seen.add(lid)
            seen_global.add((axis, lid))
            for k in ("display_name", "family", "reference_title"):
                if not isinstance(lab.get(k), str) or not lab[k].strip():
                    v.err(f"{axis}:{lid}.{k}: must be a non-empty string")
            if isinstance(lab.get("family"), str) and lid.split("/")[0] != lab["family"]:
                v.err(f"{axis}:{lid}.family: must equal the id prefix {lid.split('/')[0]!r}")
            pr = lab.get("prompts")
            if not (isinstance(pr, list) and len(pr) >= 3 and all(isinstance(p, str) and p.strip() for p in pr)):
                v.err(f"{axis}:{lid}.prompts: need at least 3 non-empty strings")
            elif len(set(pr)) != len(pr):
                v.err(f"{axis}:{lid}.prompts: duplicate prompts")
            an = lab.get("anchors")
            is_special = lid == unknown_id or lid in blocked
            if not (isinstance(an, list) and all(isinstance(a, str) and ANCHOR.match(a) for a in an)):
                v.err(f"{axis}:{lid}.anchors: must be a list of lowercase single words")
            elif not is_special and len(an) < 4:
                v.err(f"{axis}:{lid}.anchors: need at least 4 anchor words")
            elif is_special and an:
                v.err(f"{axis}:{lid}.anchors: unknown/blocked labels must have no anchors")
            ap_ = lab.get("audio_prompts")
            if uses_audio is True and lid not in blocked:
                if not (isinstance(ap_, list) and len(ap_) >= 2 and all(isinstance(a, str) and a.strip() for a in ap_)):
                    v.err(f"{axis}:{lid}.audio_prompts: need at least 2 strings on an audio-using axis")
            elif uses_audio is False and ap_:
                v.err(f"{axis}:{lid}.audio_prompts: not allowed on an axis with uses_audio=false")
        if unknown_id and unknown_id not in seen:
            v.err(f"{ap}: unknown_id {unknown_id!r} is not in the label list")
        for b in blocked:
            if b not in seen:
                v.err(f"{ap}: blocked id {b!r} is not in the label list")
            if b == unknown_id:
                v.err(f"{ap}: a label cannot be both unknown and blocked")
        real = seen - {unknown_id} - set(blocked)
        if len(real) < 6:
            v.err(f"{ap}: need at least 6 real labels, found {len(real)}")


def load_env(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file (inline '#' comments allowed). Missing file => {}. Never logs values."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, val = line.split("=", 1)
        val = re.split(r"(?:^|\s)#", val.strip(), maxsplit=1)[0].strip().strip('"').strip("'")
        if k.strip() and val:
            out[k.strip()] = val
    return out


def provider_status(raw: dict, env: dict[str, str]) -> dict[str, str]:
    """'ready', 'disabled', or 'skipped (reason)'. A provider with no key is skipped instantly at runtime."""
    allow_vision = raw.get("privacy", {}).get("allow_hosted_vision", False)
    status: dict[str, str] = {}
    for name, p in raw["providers"].items():
        if p.get("enabled") is False:
            status[name] = "disabled"
        elif p.get("type") == "vision_llm" and not allow_vision:
            status[name] = "skipped (privacy.allow_hosted_vision=false)"
        else:
            missing = [k for k in (p.get("env_key"), p.get("base_url_env")) if k and not env.get(k)]
            status[name] = "ready" if not missing else "skipped (missing " + ", ".join(missing) + ")"
    return status


def load_config(path: str | Path | None = None) -> tuple[dict, dict, list[str]]:
    """Return (config, label_catalog, warnings). Raises ConfigError listing every problem."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    text = cfg_path.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError([f"YAML parse error: {exc}"]) from exc
    catalog = None
    pre_errors: list[str] = []
    paths = raw.get("paths") if isinstance(raw, dict) else None
    labels_rel = paths.get("labels") if isinstance(paths, dict) else None
    if isinstance(labels_rel, str):
        lp = cfg_path.parent / labels_rel
        try:
            catalog = json.loads(lp.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pre_errors.append(f"paths.labels: file not found: {lp}")
        except json.JSONDecodeError as exc:
            pre_errors.append(f"labels: invalid JSON in {lp}: {exc}")
    errors, warnings = validate(raw, catalog, secret_text=text)
    errors = pre_errors + errors
    if errors:
        raise ConfigError(errors)
    return raw, catalog, warnings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m core.config", description="Validate config.yaml and the label catalog")
    ap.add_argument("--config", default=None)
    ap.add_argument("--env", default=None, help="path to .env (default: .env next to config)")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = ap.parse_args(argv)
    cfg_path = Path(args.config) if args.config else ROOT / "config.yaml"
    try:
        raw, catalog, warnings = load_config(cfg_path)
    except ConfigError as exc:
        print("CONFIG INVALID")
        for e in exc.errors:
            print(f"  - {e}")
        return 1
    except FileNotFoundError as exc:
        print(f"CONFIG INVALID\n  - {exc}")
        return 1

    env_path = Path(args.env) if args.env else cfg_path.parent / ".env"
    env = {**load_env(env_path), **os.environ}
    print(f"CONFIG OK  ({raw['project']['name']} v{raw['project']['version']})")
    for axis, ax in catalog["axes"].items():
        special = {ax["unknown_id"], *ax["blocked_ids"]}
        n = sum(1 for x in ax["labels"] if x["id"] not in special)
        extra = f", blocked: {', '.join(ax['blocked_ids'])}" if ax["blocked_ids"] else ""
        print(f"  {axis} labels: {n} real + unknown ({ax['unknown_id']}){extra}")
    print(f"  uncertain outcome: {catalog['uncertain_outcome']}")
    print("  provider status (no key => skipped instantly at runtime):")
    for name, st in provider_status(raw, env).items():
        print(f"    {name:<20} {st}")
    print("  chains:")
    for name in ("agents.image", "agents.video", "agents.audio", "agents.document", "agents.text", "generators"):
        node = raw
        for part in name.split("."):
            node = node[part]
        print(f"    {name:<16} " + " -> ".join(node["providers"]))
    if warnings:
        print(f"  warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    ! {w}")
    return 1 if (warnings and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
