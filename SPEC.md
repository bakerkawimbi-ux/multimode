# SPEC: Poem Writer (from an image or a video)

Version 0.1 · Checkpoint 1. Every number below is a **target, not a measurement**. Nothing has been trained or evaluated yet. The same targets live in `config.yaml` under `targets:` so the evaluation can compare against them.

## 1. What the system does

The user uploads a **photo or a short video** (optionally with a short text instruction and a context document). The system works out what the media shows and what mood it carries, then writes an **original poem** about it. It says "I can't write this" instead of guessing when the media is unclear, contradictory, out of scope, or unsafe.

## 2. Modalities and what each contributes

| Input | Accepted | What it contributes | Role |
|---|---|---|---|
| Image | `.png` `.jpg` `.jpeg` `.webp` | The scene (forest, sea, city at night…) and the mood | Evidence |
| Video | `.mp4` `.mov` `.webm` `.mkv`, ≤ 30 s | Over a still: change over time (sunrise progressing, rain starting), scene segments, motion | Evidence (8 sampled frames, pooled) |
| Audio track of the video | extracted with ffmpeg | Ambience that frames can't show (waves, rain, crowd). Scene axis only | Evidence |
| Text note | ≤ 500 chars, typed | Form (haiku, sonnet…), tone, a subject hint | **Instruction, never evidence** |
| Document | `.pdf` `.docx` `.txt` `.md`, ≤ 4000 chars used | Optional background the poem may draw on (a diary line, a story) | **Context, never evidence** |

At least one of image or video is required. A text-only request is out of scope, and the system asks for an image or video. Text and document content is treated as **data, not commands**: instructions hidden inside a document are ignored.

## 3. Task type and justification

**Generate**, preceded by **recognize** (Step 4 and Step 5 rule 3: recognize first, then generate from facts).

1. Recognition agents classify each source on two axes, **scene** and **mood**, with an explicit unknown and uncertain outcome.
2. Deterministic fusion (per axis) decides the verdict and the final labels.
3. A gate stops generation on `CONFLICT`, `UNCERTAIN`, `other/unknown`, or a blocked label, and asks for another input.
4. The poet (an LLM chain) receives **only the structured facts**: scene, secondary scenes, mood, form, quality flags, optional note and document text. It never receives raw pixels or the raw video, and never decides what the media shows.
5. A deterministic verifier checks the poem. If it fails, regenerate (at most 2 attempts), else return "generation unavailable".

```
 image ──────────► ImageAgent ──┐
 video ─(frames)─► VideoAgent ──┼─► fusion per axis ─► gate ─► Poet (LLM chain) ─► verifier ─► Poem + AI label
 video ─(audio)──► AudioAgent ──┘    (scene, mood)       │            ▲                │ fail ≤2x
 note + document ─► context (data only) ───────────────────────────────┘                ▼
                                                         CONFLICT / UNCERTAIN / blocked ► ask for another input
```

## 3a. Poem forms and verification

Forms (from `config.yaml`): `free_verse` (4 to 16 lines), `haiku` (3 lines, 5-7-5 syllables, tolerance 1), `rhymed_quatrains` (4 to 12 lines, multiple of 4), `sonnet` (14 lines). Default is `free_verse`; the text note may pick another form.

The verifier is deterministic and checks:

1. **Form:** line count, line multiples, and a heuristic syllable count for haiku. The syllable counter is approximate, which is why a tolerance exists.
2. **Anchor grounding:** the poem uses at least 1 anchor word of the fused scene label (or a secondary scene). This is a cheap grounding heuristic, not proof the poem is good.
3. **No copying:** no run of 8 or more consecutive words copied from the context document or the reference poems.
4. **Safety:** no real person is named as the subject of the media; no refused content.

## 4. Label catalog (`data/labels.json`)

Two axes. Each label has zero-shot prompts, one public-domain reference poem, and anchor words. Scene labels also have audio prompts for CLAP.

**Scene** (17 real + `other/unknown` + `blocked/unsafe_content`): forest, mountains, flowers and gardens, ocean and shore, river and lake, sunrise and sunset, night sky and stars, storm clouds, rain, snow and winter, autumn leaves, city street, city lights at night, cozy interior, gathering and celebration, wildlife and pets, road and path.

**Mood** (8 real + `other/unknown`): serene and calm, joyful and bright, melancholy and wistful, dramatic and intense, nostalgic and warm, lonely and quiet, hopeful and renewing, mysterious and eerie.

Outcomes beyond labels: `UNCERTAIN` (fusion, no label named) and **blocked** (generation refused: explicit or sexual content, graphic violence, a person being hurt).

If the scene is clear but the mood is `UNCERTAIN`, the poem is still written with a neutral tone (or the tone from the user's note). If the scene is not clear, no poem is written.

## 5. Success criteria (targets)

Evaluation set (to be assembled; licences recorded per file): about 68 images (4 per scene), about 30 short videos, about 20 out-of-scope items (screenshots, documents, memes, blank frames, silent clips), and programmatically built mismatched pairs.

| Metric | Target |
|---|---|
| Scene top-1, images, zero-shot | ≥ 0.70 |
| Scene top-1, images, linear probe (hash-split held-out) | ≥ 0.85 |
| Scene top-1, videos (pooled frames) | ≥ 0.75 |
| Scene top-1, audio ambience (clips with a distinctive sound) | ≥ 0.60 |
| Mood top-1, linear probe (subjective; ceiling set by annotator agreement) | ≥ 0.55 |
| Unknown rejection on out-of-scope inputs | ≥ 0.85 |
| Hallucination rate (confident scene on an out-of-scope input) | ≤ 0.10 |
| Mismatched pair (image of scene A + video of scene B) gives `CONFLICT` | ≥ 0.90 |
| Accepted poems that pass the form check | 100% (fails closed) |
| Accepted poems with ≥ 1 anchor word | ≥ 0.95 |
| CLIP score: poem matches its own image better than a mismatched image (40 pairs) | ≥ 0.85 of pairs |
| Red-team prompts refused (real-person identification, sexual, self-harm; 20 prompts) | 100% |
| Human rating of 20 poems, relevance to the media (1 to 5) | mean ≥ 3.8 |
| Human rating of 20 poems, overall quality (1 to 5) | mean ≥ 3.5 |
| p95 latency, image to poem, cache cold, free tier | ≤ 20 s |
| p95 latency, video (≤ 30 s) to poem | ≤ 45 s |
| p95 latency, cache hit | ≤ 1 s |
| Cost of a full run on the free path | $0 |

Technique ladder: zero-shot first, linear probe on frozen CLIP/CLAP embeddings only if zero-shot misses its target, fine-tuning only if both miss, with the reason written down. Open-set rejection comes from an explicit `other/unknown` class plus a tuned `reject_below` threshold per axis.

## 6. Provider chains (all in `config.yaml`)

A provider without its key is skipped instantly; an agent that exhausts its chain returns a failure result and never raises, so the other sources can still answer.

| Stage | Chain |
|---|---|
| Image | Colab server (CLIP) → local CLIP → hosted vision LLM (rank-based scores, **off by default** for privacy) |
| Video | Colab server (CLIP on frames) → local CLIP |
| Audio | Colab server (CLAP) → local CLAP |
| Document | local parsers (pypdf, python-docx, text) |
| Poet | Gemini free → Groq free → OpenRouter `:free` → Colab open model → paid Claude (**disabled by default**) |

A 429 on a daily cap means "move to the next provider", never a retry loop. If every generator fails the result is "generation unavailable", never an exception. Free-tier limits change, so check each provider's console.

## 7. Likely failure modes

| Failure | Handling |
|---|---|
| Blurry, dark, or tiny image | Quality flags (`blurry`, `dark`, `low_resolution`); below `min_side` rejected with a clear message |
| Shaky or dark video | Flags `shaky`, `dark`; lower confidence, still usable |
| Video longer than 30 s or too large | Rejected or trimmed per `limits` |
| Video has no audio track, or it is silent | Audio source dropped (flag `silent` or `no_audio`); fusion runs on frames only |
| Video changes scene midway | Segments detected from frame-embedding distance; the poet gets the dominant scene plus secondary scenes |
| Image and video disagree (or sound disagrees with picture) | `CONFLICT`: no poem; ask for one clear input |
| Screenshot, document, meme, blank image | `other/unknown` wins; no poem |
| No clear scene, weak margin | `UNCERTAIN`; no poem |
| Clear scene, unclear mood | Poem written with neutral tone or the user's tone |
| Explicit, violent, or harmful media | `blocked`; generation refused |
| People in the media | Described generically ("a crowd", "a figure"); never identified or named. "Who is this?" requests are refused |
| Prompt injection in note or document | Treated as data; the poet's system prompt forbids following embedded instructions |
| Poem copies a document or a known poem | Verifier rejects at 8 or more consecutive copied words |
| Poem ignores the scene | Anchor check fails; regenerate, then "generation unavailable" |
| All generators fail or lack keys | "Generation unavailable" plus the recognized facts, so the user still gets value |
| Non-English request | English only in v0.1 |

## 8. Data and licences

- Golden-set media: only items with a permissive licence (CC0, CC BY, public domain) recorded per file; user uploads are never added to the golden set.
- Audio ambience data: ESC-50 (CC BY-NC) for teaching demos; verify each dataset's licence on its page before using it for anything else.
- Reference poems are public domain. Generated poems are labelled "AI-generated poem" (visible mark plus metadata); prompt, model, seed, and provider are logged.

## 9. Privacy

- Uploaded media is never cached (`cache.store_inputs: false`).
- The poet gets structured facts, not pixels.
- Hosted free tiers may use prompts for training. By default the note and document go **only** to self-hosted generators (`privacy.send_context_to_hosted_llm: false`), and image pixels never go to a hosted vision model (`privacy.allow_hosted_vision: false`). Turning either on is an explicit choice, and the UI shows a notice.

## 10. Decisions and deviations from the template

- The inputs list in your template (document, text, image) did not match the idea (picture, video). I treated **image and video as the evidence inputs** and **text and document as optional instructions and context**.
- `foundations/` follows Step 2 as written, with text, image, audio and video tokenizers. Audio is needed because a video carries a sound track.
- Speech in a video's audio track (Whisper) is out of scope for v0.1; only ambience is used.
- Image generation, text to speech, and music are not part of this system. The only generated output is the poem.
- Model ids in `config.yaml` have `model_verified_on: null` and must be confirmed against Hugging Face and each provider's console before first use.
- This is a separate project from `poem_lesson/` (which derives a moral from a poem). The two could be chained later: write a poem, then derive its lesson.
