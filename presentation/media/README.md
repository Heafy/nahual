# Deck media

Every image and video the slide deck loads. One folder per slide section.

The deck must stay **offline-safe**: nothing here may be fetched from a CDN or a
remote URL, and `tools/check_style.py` fails the build if an external URL shows
up in `index.html` or the stylesheet. Assets live in this folder and ship with
the repo.

```
media/
├── README.md
├── _src/          originals, gitignored (see below)
├── intro/         photo-1.jpg  photo-2.jpg  photo-3.jpg  photo-4.jpg
├── origin/        lsm-alphabet.jpg  nahual-01.jpg  nahual-02.jpg
├── gestures/      static.mp4  dynamic.mp4
└── detect/        camera-frame.jpg
```

Ten assets total. Filenames are fixed: use these exact names so the paths in
this doc, the markup comments, and the wiring steps below all agree. Landing a
file under the right name is not enough on its own to make it appear: some
slots are already wired into `index.html`, others are still placeholder tags,
so check the slide markup itself for the current state, and see "Wiring an
asset in" at the bottom for what wiring a new slot involves.

## What each slot needs

Slot sizes are the rendered size on Reveal's 960x700 logical canvas, measured
from the current stylesheet. They are approximate and will shift slightly if a
layout changes. Export sizes are roughly 2.5x the slot so the assets hold up on
a retina laptop and a 1080p projector without bloating the deck.

**No cropping.** Every media fill rule in `css/nahual-theme.css` defaults to
`object-fit: contain`: whatever aspect ratio a photo or clip is supplied in is
the aspect ratio it renders at, full frame, letterboxed against the slot's
peach fill if it does not match the slot's own shape. Do not pre-crop
supplied media to force a fit; let the frame letterbox instead.

| File | Slide | Slot class | Slot size | Aspect | Export |
|---|---|---|---|---|---|
| `intro/photo-1.jpg` | Who Am I | `.collage__top .collage__item` | 366 x 366 | 1:1 | 1100 x 1100 |
| `intro/photo-2.jpg` | Who Am I | `.collage__row .collage__item` | 210 x 210 | 1:1 | 640 x 640 |
| `intro/photo-3.jpg` | Who Am I | `.collage__row .collage__item` | 210 x 210 | 1:1 | 640 x 640 |
| `intro/photo-4.jpg` | Who Am I | `.collage__row .collage__item` | 210 x 210 | 1:1 | 640 x 640 |
| `origin/lsm-alphabet.jpg` | How Nahual Was Created, down 1 | `.origin__media.origin__media--contain` | 398 x 622 | own aspect, letterboxed | long edge 1560 |
| `origin/nahual-01.jpg` | How Nahual Was Created, down 2 | `.origin__media-stack .origin__media-tile` (top) | 398 x 302 | own aspect, letterboxed | as supplied |
| `origin/nahual-02.jpg` | How Nahual Was Created, down 2 | `.origin__media-stack .origin__media-tile--contain` (bottom) | 398 x 302 | own aspect, letterboxed | as supplied |
| `gestures/static.mp4` | Static and Dynamic | `.split-card__clip` | 357 x 260 | own aspect, letterboxed | as supplied |
| `gestures/dynamic.mp4` | Static and Dynamic | `.split-card__clip` | 357 x 260 | own aspect, letterboxed | as supplied |
| `detect/camera-frame.jpg` | Hand Detection | `.detect__panel--frame` | 380 x 445 | 4:5 tall | 800 x 1000 |

The landmarks panel on the Hand Detection slide is an inline SVG already drawn
in `index.html`. It is not an asset and needs no file.

### Notes per slot

**`intro/` photos.** Photo 1 is the large anchor on top and carries the orange
top rule; photos 2 to 4 are the small row beneath. The frames are square, but
per the no-cropping rule above, a non-square supply photo letterboxes inside
the square rather than getting cropped to fit — square source photos are the
only way to fill the frame edge to edge.

**`origin/lsm-alphabet.jpg`.** The slot is a tall 2:3 column; the chart itself
is a 6-column grid at roughly 0.87:1, close to square. A `cover` crop down to
2:3 would slice off the leftmost and rightmost letter columns, so this slot
carries the `.origin__media--contain` modifier instead: it letterboxes against
the frame's peach fill rather than cropping, which keeps every letter
readable. JPEG, not PNG, since the content is photographs of hands, not flat
illustration. See the `.origin__media--contain` rule in
`css/nahual-theme.css` if a future replacement chart needs the same
treatment, or drop the modifier if a replacement is already 2:3 and can bleed
edge to edge like the other frames.

**`origin/nahual-01.jpg`, `origin/nahual-02.jpg`.** This slot ended up holding
two images stacked in the same reserved column instead of one, via
`.origin__media-stack` wrapping two `.origin__media-tile` frames (equal
height, gap between them). Only the top tile keeps the orange top-rule; the
bottom tile is a plain 1px border so the accent does not repeat.

`nahual-01.jpg` is landscape, close to the tile's own aspect, so it nearly
fills the frame. `nahual-02.jpg` is a portrait photo of a carving (tall ears,
long beard) in the same landscape tile, so it letterboxes with visible peach
bars left and right. Both show at their own aspect ratio, uncropped.

An auto-advancing carousel was considered for this slot and deliberately
dropped: this deck is presented live and an auto-timer has no way to know how
long the presenter is talking, so an image could flip mid-sentence. Plain
stacked images is the sturdier choice for a spoken talk. If a third image
shows up later, `.origin__media-stack` can take a third `.origin__media-tile`
child without further changes, flex handles the resize.

**`gestures/*.mp4`.** Short silent loops of a hand signing. Both supplied
clips are 16:9 (1920x1080) against a ~4:3 slot, so this frame uses
`object-fit: contain` rather than `cover`: the full recording shows,
letterboxed with peach bars top and bottom, instead of cropping into the
hand. Target for future clips:

- Loop cleanly (end pose close to the start pose)
- H.264 in an `.mp4` container, no audio track at all
- under 2 MB each so the deck stays easy to move around

`static.mp4` shows a held pose, `dynamic.mp4` shows one of the six letters
that need motion (J, K, Q, X, Z, N with tilde). Both are wired with `autoplay
loop muted playsinline data-autoplay`. The supplied clips run well over the
under 2 MB guidance above (static.mp4 is ~28.8s, 41.7 MB; dynamic.mp4 is
~10.1s, 11.3 MB) -- still valid H.264 and playable, just heavier than the
deck wants to carry around. Worth trimming and re-exporting later; ffmpeg
isn't installed in this environment to do that compression on the spot.

**`detect/camera-frame.jpg`.** A single still of what the webcam sees, raw, with
no landmarks drawn on it. The slide's whole point is the before and after, so
this is the "before". Its 4:5 shape is deliberate: it nearly matches the
landmark SVG's 5:6 viewBox next to it, so the two panels read as a pair.

## `_src/` originals

Raw phone photos, uncropped screenshots, and full-length screen recordings go in
`_src/`. It is gitignored, so nothing in it is committed and nothing in it is
referenced by the deck. The point is to be able to re-export a slot later
without hunting for the source again.

## Wiring an asset in

Dropping a file into the folder is not enough on its own. `.collage__item`,
`.origin__media`, `.origin__media-tile`, and `.split-card__clip` already have
fill rules in `css/nahual-theme.css`, all `object-fit: contain` per the
no-cropping rule above. `.detect__panel` has no equivalent rule yet; when the
first real asset lands there, add a `contain` fill rule to match.

Per slot, wiring means:

1. Replace the placeholder tag (`.collage__tag`, `.origin__media-tag`,
   `.split-card__clip-tag`, `.detect__panel-tag`) with the real element.
2. Add the fill rule for that slot's class if it does not exist yet.
3. Give every `img` a real `alt`. Decorative frames take `alt=""`.
4. For video, use `<video src="..." autoplay loop muted playsinline
   data-autoplay></video>`. Reveal's `data-autoplay` starts it when the slide
   becomes active; `muted` is what lets it autoplay at all.
5. Re-run the deck with `npm run start` and confirm the asset resolves.

The two video slots already carry commented-out markup in `index.html` with the
correct paths, so those two are close to a straight uncomment.
