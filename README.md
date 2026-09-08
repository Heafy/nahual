# Nahual

Nahual is a **real-time hand tracking recognition system** and an initial-scope ML model built in Python. The main purpose of Nahual is to learn and identify sign languages, currently LSM (Lengua de Señas Mexicana) and ASL (American Sign Language), selected with the `-lsm` / `-asl` flag. It works:

* Collecting labeled hand gesture sequences from video input
* Training a lightweight ML classifier with those gestures
* Running a demo that recognizes gestures as characters of the selected sign language in real time

## Usage

Install the dependencies once, then run any script through [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

| Script            | Purpose                                             |
| ----------------- | --------------------------------------------------- |
| `main.py`         | Real-time recognition demo (webcam + OpenCV window) |
| `collect.py`      | Capture labeled gesture samples                     |
| `train.py`        | Train the static and dynamic classifiers            |
| `inspect_data.py` | Print a table of sample counts per label            |

### Choosing a sign language

Every script takes the same optional language flag. **LSM is the default**, so
`-lsm` can be omitted; pass `-asl` to use the ASL dataset and models instead:

```bash
uv run python main.py              # LSM (default)
uv run python main.py -lsm         # LSM (explicit)
uv run python main.py -asl         # ASL

uv run python collect.py -asl
uv run python train.py -asl
uv run python inspect_data.py -asl
```

### Typical workflow for a new language

```bash
uv run python collect.py -asl       # capture samples, one label at a time
uv run python inspect_data.py -asl  # check the counts per label
uv run python train.py -asl         # train the static + dynamic classifiers
uv run python main.py -asl          # try it live
```

### Keyboard controls

Both webcam windows are driven from the keyboard:

| Key | `main.py`                               | `collect.py`                                              |
| --- | --------------------------------------- | --------------------------------------------------------- |
| `l` | —                                       | Enter a gesture label (typed in the terminal)             |
| `s` | —                                       | Capture one static sample                                 |
| `d` | Start / stop a manual dynamic recording | Start / stop dynamic capture (auto-stops after 2 seconds) |
| `m` | Toggle the motion-debug readout         | —                                                         |
| `q` | Quit                                    | Quit                                                      |
