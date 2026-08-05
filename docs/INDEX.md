# Nahual: Presentation Index

> **Step 2 of 4: index.** This is now the single source of truth for the
> presentation: topic content, order, acts, time budgets, and what each
> segment needs to land. `TOPICS.md` has been merged in and removed.

## Target run time

- **30 min content (goal), 45 min hard ceiling** including Q&A.
- Q&A is open-ended and absorbs whatever time remains under the ceiling
  (roughly 13 min if content lands on budget).

## At a glance

| # | Topic | Act | Time |
|---|-------|-----|------|
| 1 | Intro | Open | 2 min |
| 2 | Accessibility | Why | 3 min |
| 3 | How Nahual Was Created | Why | 2 min |
| 4 | Static vs. Dynamic Gestures | How It Works | 2 min |
| 5 | MediaPipe and Hand Detection | How It Works | 2 min |
| 6 | Machine Learning and Random Forest | How It Works | 2 min |
| 7 | Heuristics: Turning Landmarks Into Features | How It Works | 2 min |
| 8 | Capture, Training, and Live Demo | Proof | 7 min |
| 9 | Challenges Along the Way | The Real Story | 3 min |
| 10 | Built with Claude | The Real Story | 1 min |
| 11 | Current Limitations and Roadmap | What's Next | 2 min |
| 12 | Call to Action | What's Next | 1.5 min |
| 13 | Q&A | Close | ~13 min (flexible) |

Content total: ~29.5 min + transitions ≈ **31 min**, leaving headroom to 45.

## Segment detail

### 1. Intro — Open — 2 min
- **Goal:** Establish who's speaking and give the audience a reason to
  listen, before the pitch itself starts.
- **Needs:** None technical, just you.
- **Bridge:** From who you are to the problem you decided to work on.
- **Description:** A brief self-introduction, name, role, and whatever
  context the room should have before the accessibility pitch begins. This
  is not the origin story (that's #3), just enough for people to place who's
  talking. Kept short on purpose.

### 2. Accessibility — Why — 3 min
- **Goal:** Land the gap (mainstream tech doesn't understand LSM) so the
  audience feels the "why" before any technology appears.
- **Needs:** No diagrams or code yet, keep this human, not technical.
- **Bridge:** From the abstract gap to a personal reason someone closed it.
- **Description:** Millions of people in México communicate daily in LSM
  (Lengua de Señas Mexicana), yet the technology around them (voice
  assistants, captioning, dictation) was built to understand speech and
  text, not signs. This topic opens the presentation by naming that gap
  directly: sign language is a complete, independent language that
  mainstream tech has largely ignored, and closing that gap is the reason
  Nahual exists.

### 3. How Nahual Was Created — Why — 2 min
- **Goal:** Personal motivation and the name story, make the project feel
  human before it turns technical.
- **Needs:** None technical; a personal beat.
- **Bridge:** From motivation to the first real design fact about LSM
  itself.
- **Description:** The origin story of the project: what motivated starting
  it, and why it is named Nahual. A nahual (or nagual) is a figure from
  Mesoamerican folklore able to shift between forms, a fitting name for a
  tool whose whole job is translating one form of expression (hand shape
  and motion) into another (letters and text). Covers the path from first
  idea to a working prototype.

### 4. Static vs. Dynamic Gestures — How It Works — 2 min
- **Goal:** Establish that LSM letters split into held poses vs. movement,
  this is the reason two models exist later.
- **Needs:** A simple visual grouping the alphabet into static vs. dynamic
  letters.
- **Bridge:** From why two models exist to how a hand gets seen at all.
- **Description:** Not every letter in the LSM alphabet behaves the same
  way. Most are a held pose, but a handful (j, k, q, x, z, ñ) only make
  sense as movement. This topic explains that split as a deliberate design
  decision, two datasets and two models instead of one classifier forced to
  solve both problems, and the trade-offs that came with it.

### 5. MediaPipe and Hand Detection — How It Works — 2 min
- **Goal:** Explain how a hand becomes 21 tracked points, and why detection
  quality matters.
- **Needs:** A landmark-overlay visual (screenshot or short clip).
- **Bridge:** From raw points to what actually classifies them.
- **Description:** Before any gesture can be classified, the system needs
  to see a hand at all. This topic covers MediaPipe's HandLandmarker model,
  which extracts 21 three-dimensional landmark points per hand from every
  camera frame, and why the quality of that detection (tuned tracking
  confidence, settings shared identically between data collection and live
  use) is the foundation everything else depends on.

### 6. Machine Learning and Random Forest — How It Works — 2 min
- **Goal:** Name the algorithm, explain why Random Forest, tie back to the
  static/dynamic split already introduced.
- **Needs:** Optional simple diagram of trees voting.
- **Bridge:** From naming the algorithm to what data it's actually fed.
- **Description:** Introduces the classification algorithm behind Nahual
  and why Random Forest was chosen: reliable accuracy without needing a
  large dataset or a GPU, and straightforward to retrain as the project
  grows. Covers that there are two separate Random Forest models at play,
  one for static poses and one for dynamic motion, matching the split
  introduced earlier.

### 7. Heuristics: Turning Landmarks Into Features — How It Works — 2 min
- **Goal:** Explain the feature-extraction step that turns landmarks into
  model input.
- **Needs:** A before/after diagram: raw landmarks → feature vector.
- **Bridge:** From theory to proof, seeing it actually work.
- **Description:** The bridge between raw hand landmarks and something a
  classifier can actually learn from. Covers how 21 raw points become
  meaningful features: normalized distances and angles between fingers for
  static poses, and speed, direction, and trajectory shape for dynamic
  motion. Also covers why the same feature extraction code is shared
  between data collection and live recognition, so training and real world
  use never see a different shape of data.

### 8. Capture, Training, and Live Demo — Proof — 7 min
- **Goal:** Show the real workflow end to end and prove it works live,
  twice (desktop, then the public website), then recap how the pieces fit
  together.
- **Needs:** Working desktop app, working internet connection for the
  website, camera and lighting tested in the actual room beforehand.
- **Bridge:** From the polished result to what it took to get there.
- **Description:** The practical workflow from start to finish: collecting
  labeled gesture samples letter by letter with the interactive collector,
  then training the static and dynamic classifiers from that dataset.
  Includes a live demonstration of the result, shown first running locally
  through the desktop app, and then again through the public website, so
  the audience sees the same recognition working in two very different
  environments. Closes with a short architecture recap, how the entry
  scripts (`main.py`, `collect.py`, `train.py`, `inspect_data.py`) and the
  shared `nahual/` package fit together, tying the whole workflow back into
  one picture before moving on.

### 9. Challenges Along the Way — The Real Story — 3 min
- **Goal:** Show the engineering rigor behind the smooth demo just shown.
- **Needs:** A concrete before/after metric or clip if one's available
  (e.g. dropped-frame counts).
- **Bridge:** From the hard problems to who helped solve them.
- **Description:** The behind the scenes engineering story: hands getting
  lost mid gesture during fast motion, tuning MediaPipe's tracking
  sensitivity to recover without becoming unstable, and filling small
  tracking gaps with interpolation so motion data stays clean. Later on,
  diagnosing why the live web demo felt worse on a hosted server than
  locally, a network latency problem that quietly eroded the classifier's
  confidence below its threshold. Shows the project as a series of real
  problems reasoned through, not a model that simply worked on the first
  try.

### 10. Built with Claude — The Real Story — 1 min
- **Goal:** Specific, credited acknowledgment of what Claude contributed.
- **Needs:** None, a quick text beat.
- **Bridge:** From credit to honesty about current scope.
- **Description:** A direct, specific acknowledgment of what Claude
  contributed: help with the machine learning design and the real time
  engineering, rather than a vague mention of "AI assistance." Frames
  Nahual itself as an example of what is achievable when building alongside
  an AI collaborator.

### 11. Current Limitations and Roadmap — What's Next — 2 min
- **Goal:** Set realistic expectations before the ask.
- **Needs:** None, a plain talking point.
- **Bridge:** From limitations to a concrete ask.
- **Description:** An honest account of scope: Nahual recognizes individual
  letters (fingerspelling), not full words, phrases, or continuous natural
  signing, and the dataset so far comes from a single signer. Names what is
  next: broadening the dataset, validating accuracy with native LSM signers
  or Deaf community members, and any features being considered beyond the
  alphabet.

### 12. Call to Action — What's Next — 1.5 min
- **Goal:** Give the audience something concrete to do right now.
- **Needs:** QR code or URL for the public web demo, visible on screen.
- **Bridge:** From the ask to open floor.
- **Description:** Closes the presentation with what is being asked of the
  audience: try the live public web demo yourself (no install, just a
  link), share feedback on where recognition felt wrong, and help connect
  the project with LSM speakers, interpreters, or Deaf community
  organizations who could validate or expand it. This is also where the
  project's move from a local desktop tool to a public website gets its
  closing moment. Ends on what people can do next, not only on what has
  already been built.

### 13. Q&A — Close — ~13 min (flexible)
- **Goal:** Real audience questions, plus the pre-loaded ones already
  answered so the closing never stalls.
- **Needs:** Keep the confirmed question list handy (below).
- **Description:** Open floor for the audience to ask their own questions,
  but also a chance to answer a handful of predictable ones even if nobody
  raises them, so the closing does not stall on the spot. Confirmed
  questions so far:
  - Is this approach reusable for other sign languages, such as American
    Sign Language (ASL)?
  - Is my camera feed private, does video ever leave my device?
  - Why Random Forest instead of a neural network?
  - Is the model overfitted to my hand and gestures specifically?
  - What happens if the hand is no longer detected mid-gesture?

  More questions to be folded in here as they come back to mind.
