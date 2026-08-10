# Nahual presentation — design system

The single source of truth for the deck's visual language and the workflow for
building slides. Read this before adding or editing a slide. The mechanical
rules here are also enforced by `tools/check_style.py` (run automatically after
edits via a `PostToolUse` hook); the subjective rules are not, so they live on
the checklist at the bottom and depend on discipline.

Tokens live in `css/nahual-theme.css` `:root`. Always use the CSS variables, not
raw hex.

## Palette and color roles

| Token | Hex | Role |
|---|---|---|
| `--orange` | `#fd5108` | Core accent. Text **only** at display size, and only on white or `--bg-peach-100`. Elsewhere structural: rules, bars, fills. Never body copy. |
| `--viz-1/2/3` | `#fd5108 #fe7c39 #ffaa72` | **Sequential** ramp (one hue lightening). Magnitude only, cap ~3 steps, always direct-label. Never categorical. |
| `--bg-white` | `#ffffff` | Background. |
| `--bg-peach-300` | `#ffcda8` | Background (deepest tint, sparingly). |
| `--bg-peach-200` | `#ffe8d4` | Media-frame fill. |
| `--bg-peach-100` | `#fff5ed` | Default content-slide background. |
| `--text-black` | `#000000` | Body text. The only text color that passes contrast on every background. |
| `--text-muted` | `#595959` | De-emphasized text (7:1 on white). |
| `--line-grey` | `#a1a8b3` | **Non-text only**: borders, hairlines, gridlines, tree branches, markers. Fails as text (2.4:1). |
| `--amber` | `#ff9f00` | Tertiary / illustration only (e.g. square list markers). Never text (2.06:1). |
| `--status-good/warn/bad` | `#059669 #e9b01f #dc2626` | Functional state only, always with an icon or label, never a decorative or series color. |

Contrast rule of thumb: black on any background is safe. Orange as text is
large-display-only, on white or `#fff5ed`. Muted `#595959` is the de-emphasized
text color. `#a1a8b3` and `#ff9f00` never carry text.

Categorical charts (if ever needed): orange + black (+ grey), never shades of
orange to distinguish series.

## Typography

Two typefaces only. No third family; the trailing `serif`/`sans-serif` are
generic category keywords, not fonts.

- `--font-display` = `Georgia, serif` — headlines, leads, quotes, big display.
- `--font-text` = `Arial, sans-serif` — sub-heads, body, intros, labels, large
  data numbers.

Rules:
- Regular or bold weights only. **No italics anywhere.** `em`/`i`/`blockquote`/`q`
  are neutralized to `font-style: normal`; emphasis is carried by color
  (`.accent`), never slant.
- Monospace is reserved for literal `<code>`/`<pre>` only.

## Card tiers

Two coherent tiers. Keep them distinct.

- **Content card** (holds text): orange top-rule `5px` only, **no surrounding
  border**, white background. → `.card`, `.feature__card`, `.split-card`,
  `.forest__result`.
- **Media frame / tile** (holds image, SVG, or video): `1px solid --line-grey`
  border, optionally plus an orange top-rule, optionally `border-radius: 6px`. →
  `.collage__item`, `.origin__media`, `.split-card__clip`, `.detect__panel`,
  `.forest__tree`.

## Shared elements

- `.eyebrow` — orange uppercase Arial label (`0.4em`, letter-spacing `0.18em`).
  Used as the section label at the top of a slide. (Small orange technically
  misses 4.5:1; accepted for this bold uppercase label only. Never use orange
  for small **body** text.)
- `.accent` — orange span inside a Georgia headline. Color only, never italic.
- Amber square markers for bullet lists (`::before` squares).

## Backgrounds

- `--bg-peach-100` is the default for "how it works" content slides.
- White is fine for variety (intro, origin); not treated as an inconsistency.
- `--bg-peach-200` fills media frames so they sit above a peach-100 slide.
- The cover uses a peach gradient with an orange left border.

## Layout catalog (built slides)

Reuse or extend these before inventing a new layout.

| Slide | Class | Shape |
|---|---|---|
| Cover | `.cover` | Eyebrow, Georgia title w/ orange accent word, orange underline, meta line. |
| Who Am I | `.intro` + `.collage` | Text column + photo collage (1 large square + 3 small, all 1:1). |
| Accessibility | `.statement` | Text-forward "why": eyebrow, big Georgia hero w/ orange accent, muted sub. |
| How Nahual Was Created | `.origin` (+ `.origin--reverse`) | Two-column story + media, as a vertical pair. Georgia lead / Arial body / orange punch. Reverse swaps image side. |
| Static and Dynamic | `.split` | Two complementary cards (static/dynamic): label, desc, video clip slot, letter chips. |
| MediaPipe | `.detect` | Pipeline: camera-frame placeholder → landmark SVG, with captions + a muted note. |
| Random Forest | `.forest` | Text left (lead, why-list, note) + voting diagram right (trees → majority result). |

## Copy rules (Jorge's preferences)

- No em dashes. State the gap once, echo briefly at the close.
- No implementation jargon in covers/eyebrows/non-technical copy; naming tools
  (MediaPipe, Random Forest) is fine in the **body** of technical sections.
- Specific "built with Claude" credit, not vague "AI assistance."
- **Inform over minimal**: slides communicate the right info; the full spoken
  script goes into `<aside class="notes">` (paragraphs as `<p>`).
- **Propose wording as plain text in chat first**; only edit files after Jorge
  picks. Use his exact provided text when he gives it; silently fix obvious
  typos.

## Build workflow (per slide)

1. Pull the segment's text from `docs/SCRIPT.md` (the approved content).
2. **Compose, don't transcribe.** Distill prose into slide lines (lead +
   support); don't dump paragraphs. Propose wording first when it's a choice.
3. Build with the system: reuse/extend a layout, Georgia lead, one orange
   accent moment, Arial body/muted, correct card tier.
4. Put the full verbatim script into `<aside class="notes">`.
5. Run the consistency checklist below before calling it done.
6. Serve-check (`npm run start`) that it renders and assets resolve.

## Consistency checklist

Mechanical (also enforced by `tools/check_style.py`):
- [ ] Palette colors only (via CSS vars); no stray hex.
- [ ] No italics; no `<em>`/`<i>`.
- [ ] No em dashes in visible copy.
- [ ] Offline-safe: no external URLs / CDN / webfonts.

Subjective (not automatable — check by eye against this doc):
- [ ] Eyebrow label present.
- [ ] Georgia lead at an established size; Arial body.
- [ ] Exactly one orange accent moment (display-size text on white/peach-100, or
      structural).
- [ ] Composed lines, not prose paragraphs.
- [ ] Card tiers correct (content = top-rule only; media = 1px border).
- [ ] Speaker notes carry the full spoken script.
- [ ] Background is intentional (peach-100 default, or deliberate variety).
