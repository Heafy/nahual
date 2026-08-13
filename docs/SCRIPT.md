# Nahual: Presentation Script (Draft)

> **Step 3 of 4: script.** Hybrid format: the opener (#2) and the closing ask
> (#12) are written verbatim, meant to be said close to as-is since exact
> phrasing matters most there. Everything else is talking points, cues and
> facts to hit, delivered naturally rather than read. Time tags are loose
> references from `INDEX.md`, not hard limits, some topics (MediaPipe,
> Random Forest) may reasonably run longer than tagged, that's fine.
>
> First draft, react to it and edit freely.

---

### 1. Intro (~2 min) — TALKING POINTS

- `[JORGE: introduce yourself, name, role, and whatever background you want
  the room to have before the pitch starts]`
- Keep it short, a name and a reason to listen, not the origin story, that's
  coming up next in topic #3.

---

### 2. Accessibility (~3 min) — VERBATIM

Right now, in México, millions of people communicate every day in LSM,
Lengua de Señas Mexicana. It isn't a translation of Spanish. It's its own
language, with its own grammar, used daily by the Deaf community across the
country.

And yet, look at the technology around us. Voice assistants that listen.
Captions that transcribe. Dictation that types what we say. All of it was
built to understand speech, or text. None of it understands a sign.

That gap has been sitting there the whole time. This is Nahual: a tool that
reads a hand the way a microphone reads a voice.

---

### 3. How Nahual Was Created (~2 min) — TALKING POINTS

- Started from a 1:1 with my manager, Pau Rico. She was learning LSM, and
  I'd always been curious about it too, somewhere in that conversation the
  idea clicked: could a machine actually recognize the gestures. Took it
  personally from there.
- The name: a nahual (or nagual), a figure from Mesoamerican folklore able
  to shift between forms. Fitting, since the whole job of this tool is
  translating one form of expression (hand shape and motion) into another
  (letters and text).
- Quick path from first idea to a working prototype, no need to dwell here,
  the "why" already landed in the opener.

---

### 4. Static vs. Dynamic Gestures (~2 min) — TALKING POINTS

- Not every letter in the LSM alphabet behaves the same way.
- Most letters: a held pose. Show the hand, hold it, that's the letter.
- A handful only make sense as movement: j, k, q, x, z, ñ.
- `[VISUAL: alphabet split into two columns, static vs. dynamic]`
- This split is why there end up being two separate models later, not one
  classifier stretched to cover both.

---

### 5. MediaPipe and Hand Detection (~2 min) — TALKING POINTS

- Before anything gets classified, the system has to see a hand at all.
- MediaPipe's HandLandmarker: 21 points per hand, in 3D, every frame.
- `[VISUAL/DEMO: landmark overlay on a live or recorded hand]`
- Detection quality is tuned deliberately (tracking confidence) and the
  exact same settings are used both when collecting training data and when
  recognizing live, so the model is never surprised by a different kind of
  input.
- This is the foundation everything downstream depends on, worth taking the
  time this needs even if it runs past the tag.

---

### 6. Machine Learning and Random Forest (~2 min) — TALKING POINTS

A decision tree is a supervised learning algorithm used for classification and regression tasks.
Consists of a root node, branches and leaf nodes. 

A decision tree split the dataset based on feature values to create pure subsets of the items in
a group that belongs to the same class. Asking different questions. 
For example:
1. Root node: Is the index fingertip higher than this line?
    Yes: Pass to internal node
    No: Go to a different question
2. Internal node:  Is the thumb touching the middle finger?
   Yes: It's letter b

The three builds itself during training, every questions thries thousand of possibilities and 
keeps the ones that best separates the letters that still in play until it ends on a single answer and 
it's how you would describe a hand sign: Thumb up, index and middle extended, that fingers are together. 
A tree learns the same kind or rule just measured instead of described.

But a single tree only memorize, give it 200 photos and it learn your hand, then fails on someone else.
Fix it using 200 decision trees and make every one of them use random features on purpose. Each tree doesn't look all the features at once. It picks a few random on how to split the data. So the trees stay different from each other.

Now each three making his own predition based on what it learned from its part of the data. And each 
single vote create confidencence. The forest answer independently and the system averages their answers.
If 190 trees saying "B" is high confidence around 95% of them. That confidence show how much the trees agree.I set the confidence threshold at 65% agreement for static letters, and 40% for the moving ones — motion is harder to pin down frame by frame, so I gave it more room.

Why this model, because it trains on seconds on a Macbook, no GPU, no cloud resources, it can run fast
enough for live video in with low device requirements and it's inspectable so when two letters get confused I can see exactly which pair and why.

---

### 7. Heuristics: Turning Landmarks Into Features (~2 min) — TALKING POINTS

- Random Forest doesn't take raw landmark points directly, it needs
  meaningful features.
- This is the translation step: 21 raw points become normalized distances
  and angles between fingers for static poses, and speed, direction, and
  trajectory shape for dynamic motion.
- `[VISUAL: before/after diagram, raw landmarks → feature vector]`
- Same feature-extraction code runs during data collection and during live
  use, so training and the real world are always speaking the same
  language.

---

### 8. Capture, Training, and Live Demo (~7 min) — TALKING POINTS

- Collecting labeled samples letter by letter with the interactive
  collector.
- Training the static and dynamic classifiers from that dataset.
- `[DEMO: switch to the desktop app]` — show a few static letters, then a
  dynamic one.
- `[DEMO: switch to the public website]` — same recognition, now running
  from a link, no install.
- Close with a short architecture recap: `main.py`, `collect.py`,
  `train.py`, `inspect_data.py`, all sharing the same `nahual/` package
  underneath, tying the whole workflow back into one picture.

---

### 9. Challenges Along the Way (~3 min) — TALKING POINTS

- Getting here wasn't a straight line.
- Hands getting lost mid-gesture during fast motion.
- Tuning MediaPipe's tracking sensitivity cut dropped frames during a quick
  gesture from about 9 down to about 3.
- Remaining gaps filled with interpolation so the motion data stays clean.
- Separately: the web demo felt worse hosted than local, diagnosed as
  network latency starving the model of frames and quietly eroding
  confidence below its threshold, not a compute problem.
- First fix (throttling, batching) helped, but the real fix was more
  fundamental: move recognition into the browser entirely, so there's no
  network round trip per frame anymore. Verified it predicts bit-identical
  to the desktop app.
- Point: real problems, reasoned through, sometimes solved by rethinking the
  architecture rather than patching the symptom.

---

### 10. Built with Claude (~1 min) — TALKING POINTS

- Specific credit: Claude helped with the machine learning design and the
  real-time engineering work throughout this project, not just "AI
  assistance" in the abstract.
- Nahual itself is an example of what's achievable building alongside an AI
  collaborator.

---

### 11. Current Limitations and Roadmap (~2 min) — TALKING POINTS

- Honest about scope: individual letters today (fingerspelling), not full
  words, phrases, or continuous natural signing yet.
- Dataset collected from a single signer so far.
- What's next: broaden the dataset, validate accuracy with native LSM
  signers or Deaf community members, consider what's beyond the alphabet.

---

### 12. Call to Action (~1.5 min) — VERBATIM

So here's what I'd ask. Open your phone right now, scan this, and try it
yourself. No install, nothing to set up, just a link.

Tell me where it got a letter wrong. And if you know an LSM speaker, an
interpreter, or anyone connected to the Deaf community here in México,
connect me with them. This gets better with more hands and more signers
than one person collecting data alone can provide.

We started because the tools around us never learned to see a sign. Now you
can carry one in your pocket.

---

### 13. Q&A (~13 min, flexible) — ANSWER KEY

- **Reusable for ASL?** The pipeline (landmarks → features → classifier)
  generalizes to any sign language. The labels and training data are
  LSM-specific, so ASL would need its own dataset and a retrain, not a code
  rewrite.
- **Is my camera feed private?** The browser demo runs MediaPipe and both
  classifiers client-side. No video or landmarks are sent to a server.
- **Why Random Forest, not a neural network?** Solid accuracy with a modest
  dataset, no GPU needed, fast to retrain as data grows. A neural net would
  want far more data than currently collected.
- **Overfitted to my hand specifically?** A fair concern, but closed testing
  with a handful of different users showed the model still recognized
  gestures confidently across different hand sizes and shapes.
- **What if the hand is lost mid-gesture?** A grace period tolerates a few
  consecutive missing frames before giving up, and any gap gets filled with
  interpolation so a brief dropout doesn't kill the recognition.
