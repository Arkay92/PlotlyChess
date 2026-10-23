# Plotly Chess — change breakdown and handover

Prepared: 23 October 2026.

This document describes the changes made during this rework, how to run them,
what was verified, and the remaining limitations. It describes the local
workspace; it does not imply that the changes have been committed, pushed or
successfully deployed to Plotly Cloud.

## 1. Overview

The original application was a single large `main.py` containing the Dash UI,
chess rules, a shared board, language-model loading and unfinished socket-based
multiplayer. It has been replaced with a modular application, an interactive
board, an optional trainable chess AI, tests and deployment preparation tools.

| Area | Original behaviour | Current behaviour |
| --- | --- | --- |
| Structure | Rules, UI, AI and networking in one file | Separate modules with a small entry point |
| Game state | Global mutable board shared across visitors | Move history and selection stored per browser session |
| Startup | Language-model download/loading and socket listener | Explicit app setup; no model downloads or socket listener on import |
| Opponent | Random choice favouring captures | Two-ply material search, optionally supplemented by routed NPC predictions |
| Advice | Language-model question box with move extraction | A legal move hint from the actual move selector |
| Multiplayer | Incomplete socket lobby | Working two-player mode on one device |
| Board | Plotly traces and Unicode piece characters | Responsive HTML squares and bundled SVG pieces |
| Interaction | Click-based movement | Click, mouse/pen/touch dragging and keyboard controls |
| Training | No chess training pipeline | PGN ingestion, circuit experts, neural gate and evaluation |
| Deployment | Factory-created app in `main.py` | Explicit `app = Dash(...)` in root `app.py` plus upload preparation |
| Verification | No project test suite | 15 automated tests and a separate real-browser check |

## 2. Project structure

```text
app.py                         Plotly Cloud entry point; explicit Dash instance
main.py                        Local entry point and compatibility WSGI export
requirements.txt               Runtime dependencies
requirements-dev.txt           Runtime dependencies plus pytest
README.md                      Setup, gameplay, training and deployment instructions
PROJECT_CHANGES.md             This handover
.gitignore                     Generated/local files excluded from Git
.github/workflows/tests.yml    Automated test workflow

plotly_chess/
  __init__.py                  Package marker
  app.py                       Layout, UI helpers, application factory and callback
  game.py                      Game creation, replay, move validation and click transitions
  board.py                     Interactive board, SVG pieces and optional Plotly renderer
  ai.py                        Concepts, probabilistic circuits, neural routing and search
  train.py                     PGN loading, progress reporting, fitting and evaluation

assets/
  styles.css                   Responsive application and board styling
  board.js                     Pointer gestures and keyboard interaction
  logo.png                     Existing asset retained

data/pgn/
  README.md                    Dataset sources and preparation instructions
  prepare.py                   Validation, mainline deduplication and combined PGN export
  manifest.json                Dataset counts, source links and hashes
  Carlsen.zip / Carlsen.pgn    Downloaded source files, ignored by Git
  Kasparov.zip / Kasparov.pgn  Downloaded source files, ignored by Git
  Anand.zip / Anand.pgn        Downloaded source files, ignored by Git
  training.pgn                 Prepared training collection, ignored by Git

scripts/
  prepare_cloud.py             Creates the Plotly Cloud upload folder

tests/
  test_game.py                 Rule handling and independent game states
  test_ai.py                   Circuit fitting, probabilities, persistence and AI integration
  test_train.py                Training pipeline, progress and recorded-game preview
  test_app.py                  HTTP endpoints and callback actions
  test_board.py                Board orientation, pieces, legality and highlights
  test_cloud.py                Cloud entry point and asset serving
  browser_check.py             Optional actual-browser interaction checks
```

The obsolete commented-out `assets/script.js` was removed and replaced by the
working `assets/board.js` interaction code.

## 3. Rules and game-state refactor

`plotly_chess/game.py` contains the game operations independently of Dash.

- A game is represented by its UCI move history and selected square.
- Replaying that history reconstructs the board, including castling rights,
  en passant state and repetition history.
- Move transitions return new state instead of mutating a global board.
- Both SAN, such as `Nf3`, and UCI, such as `e2e4`, are accepted.
- Legal moves are checked with python-chess.
- Promotion supports queen, rook, bishop and knight.
- Moves after game completion are rejected.
- Status text identifies the next player, check and completed-game outcomes.

The UI stores game data in a session-backed `dcc.Store`. Play mode also persists
within the browser session. Refreshing the page restores the game and mode;
changing mode deliberately starts a new game.

Draw claims are automatically accepted when python-chess permits them. There is
no separate manual draw-claim button.

The server validates submitted moves, but game history is supplied by the
browser. There is no authoritative shared game database, account system or
tamper-resistant competitive match service.

## 4. AI and the supplied NPC algorithm

### What was adapted

The supplied `neural_probabilistic_circuits_v18_conditional_expert_routing.py`
is a research notebook export for network-intrusion classification. Its central
idea—conditionally weighting specialist probabilistic experts from their
posterior evidence—was adapted for chess.

The notebook was treated as reference material. It is not imported or executed
by the app. Its runtime package installation, intrusion dataset downloads and
intrusion-classification weights are not used.

### Chess-specific implementation

`plotly_chess/ai.py` provides three bootstrapped categorical probabilistic
circuit experts. Each estimates the outcome for the side to move:

1. Loss.
2. Draw.
3. Win.

The board concepts are coarse categorical versions of:

- Material balance.
- Legal-move availability.
- Whether the side to move is in check.
- Game phase, approximated by the number of remaining pieces.
- Capture availability.

Each expert is a class-prior sum with factorized categorical likelihood
products. A small neural gate uses a tanh hidden layer and softmax routing
weights to combine the experts' normalized posteriors.

The gate receives expert entropy, top-class margins, win/loss posterior mass
and KL disagreement with the ensemble. It is fitted on games separate from
those used to estimate the experts.

This is a compact chess adaptation, not a complete port of the notebook's
joint hierarchical architecture, specialist concepts or experimental suite.
The circuit experts use count-based estimation; the routing gate is the
neural component trained by gradient updates.

### Move selection

The default opponent searches the candidate move and the opponent's replies,
using material and terminal results. With a configured trained model, a bounded
routed outcome term supplements that evaluation. Predictions for the opponent
to move are interpreted from the correct perspective.

Terminal outcomes take priority over the small learned term. The AI only
chooses legal moves, and the same selector powers hints. In AI mode the human
plays White; local two-player mode allows both sides to be played on one device.

### Model storage

Models are stored in compressed `.npz` files. Loading uses
`numpy.load(..., allow_pickle=False)`.

Set `PLOTLY_CHESS_MODEL` to the artifact path to enable routing. Without it,
the app uses material search and identifies that mode in the UI. A configured
model that cannot be loaded causes a startup error rather than silently
pretending that routing is active.

No playing-strength improvement or superiority over uniform averaging has
been established. The work performed here did not include a full training run
on the downloaded corpus or an engine-strength benchmark.

## 5. Downloaded training games

Three public collections were downloaded from
[PGN Mentor](https://www.pgnmentor.com/files.html). The counts below are from
the actual downloaded files, rather than the website's listing.

| Collection | Games read | Retained in combined file | Duplicates removed | Other exclusions |
| --- | ---: | ---: | ---: | ---: |
| [Carlsen](https://www.pgnmentor.com/players/Carlsen.zip) | 7,818 | 7,818 | 0 | 0 |
| [Kasparov](https://www.pgnmentor.com/players/Kasparov.zip) | 2,128 | 2,124 | 3 | 1 |
| [Anand](https://www.pgnmentor.com/players/Anand.zip) | 4,311 | 4,083 | 227 | 1 |
| **Total** | **14,257** | **14,025** | **230** | **2** |

The combined `data/pgn/training.pgn` contains:

| Outcome | Games |
| --- | ---: |
| White win | 5,189 |
| Black win | 3,585 |
| Draw | 5,251 |

`data/pgn/prepare.py` parses the source files, excludes errors, incomplete
results, empty games and nonstandard starting positions, and deduplicates
identical mainline UCI move sequences across the collections. Comments and
variations are stripped from the combined export.

Completed outcomes are taken from the source result headers. A resignation or
agreed draw does not need to end in a terminal board position to be included.

`manifest.json` records source URLs, source PGN SHA-256 hashes, exact counts and
the combined-file hash. Source archives and PGNs remain local and are ignored
by Git; they are not included in the deployment upload.

This collection is a convenient master-game dataset, not a balanced sample of
all chess play. Whole-game splitting avoids copying the same game across
partitions, but different games can still contain the same opening positions.

## 6. Training pipeline and progress

The training command reads completed PGN games and splits whole games into:

- 60% for fitting the circuit experts.
- 20% for fitting the neural gate.
- 20% for held-out evaluation.

At least ten usable games are required. The default random seed is `42`, and
the gate runs for 300 epochs unless overridden.

Console reporting includes:

- PGN loading progress measured by file bytes.
- Loaded games, extracted positions and skipped games.
- Elapsed time and approximate ETA.
- Expert-fitting progress.
- Gate epoch progress and fitting negative log likelihood (NLL).
- Held-out gated, uniform-average and oracle NLL.
- Routing regret relative to the oracle expert ceiling.

Lower NLL is better. Fitting NLL is a training metric; it is not the held-out
evaluation result. The oracle chooses the best expert using the known true
label and is a diagnostic ceiling, not a deployable routing strategy.

`--show-games` prints sampled ASCII boards while recorded PGN moves are
processed. Previews are throttled to roughly two-second intervals without
adding sleeps. They show recorded-game replay, not self-play, and do not
continue during numerical circuit/gate fitting.

Training currently runs in memory with a full-batch gate. There is no resume
checkpoint, early stopping, graphical training dashboard or automatic model
promotion to the running app.

## 7. Interface redesign

The UI now uses a warm neutral background, muted green board colours, a compact
header and a desktop layout with the board beside a control panel.

The redesign includes:

- Crisp SVG pieces from the existing python-chess package, embedded locally.
- Player bars identifying White, Black and whose turn it is.
- A computer/local two-player mode selector.
- Separate turn status and move feedback.
- Hint and new-game buttons.
- Move history grouped into numbered White/Black rows.
- A collapsible section for notation entry and promotion choice.
- A label showing whether material search or NPC routing is active.
- A mobile layout with controls below the board.

The visible playing surface is now an HTML/CSS grid rather than a Plotly
scatter plot. The optional Plotly renderer remains in `board.py`.

## 8. Board interaction

`assets/board.js` bridges pointer and keyboard events to a Dash store. The
server then validates the requested action and returns the updated board.

Supported interactions and feedback:

- Click a piece, then click its destination.
- Drag with a mouse, pen or touch.
- A piece preview follows the pointer while dragging.
- Legal destinations appear during selection and dragging.
- Capturable destinations use a ring; empty destinations use a dot.
- The previous move and a checked king are highlighted.
- Invalid drops leave the move history unchanged and display feedback.
- Dropping outside the board or pressing Escape during a drag cancels it.
- Tab or arrow keys move focus; Enter/Space selects squares.

Squares have accessible labels and focus indicators. Feedback uses a live
status region, and reduced-motion preferences disable transitions.

The interaction code suppresses the synthetic click after a pointer gesture
so a drag does not also select a square. It also coordinates actions while
Dash updates the board, including queuing a subsequent click.

## 9. Plotly Cloud preparation

The original refactored entry point used `app = create_app()`. That is valid
locally, but was a likely cause of the Cloud upload scanner reporting no Dash
app. A root `app.py` now explicitly declares `app = Dash(...)` and passes that
instance into the package's `create_app(app=None)` setup function.

`main.py` imports the configured app and server, preserving the local command
and `main:server` compatibility. `app:server` is also available.

`scripts/prepare_cloud.py` creates `deploy/plotly-cloud/` containing:

- Root `app.py` and `main.py`.
- `requirements.txt`.
- The `plotly_chess/` Python modules.
- The `assets/` files.
- Optionally, a trained model copied as `chess-npc.npz`.

The upload excludes training data, tests and the virtual environment. The
generated deployment folder is ignored by Git. Re-run preparation after code
changes; the folder is a copy, not a live link to the source.

Upload its contents while preserving the subfolders. Choose `app.py` as the
main file and Python 3.11. Uploading just `app.py` is insufficient because it
imports the package. See [Plotly's publishing guide](https://dash.plotly.com/plotly-cloud/publish).

The prepared folder was imported and exercised independently, and its layout,
dependencies and assets returned successful HTTP responses. Actual Cloud
detection, building and deployment have not been verified in the hosted service.

## 10. Dependencies and startup cleanup

Runtime requirements are now bounded to:

```text
dash>=3.0,<4
plotly>=6.0,<7
chess>=1.11,<2
numpy>=1.26,<3
```

`requirements-dev.txt` adds `pytest>=8,<10`. These ranges are not an exact
dependency lockfile.

Removed runtime requirements and behaviour include:

- Transformers and PyTorch model loading.
- The hardcoded Hugging Face cache path.
- The unused pandas import.
- Automatic chess language-model downloads.
- The socket listener and background networking threads.
- The free-form language-model question and move-extraction flow.

A workspace-local `.venv` was created for running and testing the app.
There is no new production requirement for Playwright; it is used only by the
optional browser check.

## 11. Tests and verification

The latest automated suite completed with **15 tests passing**.

Coverage includes independent game states, legal/illegal moves, click
transitions, repetition history, checkmate, mating move selection, castling,
en passant, underpromotion, normalized model outputs, model persistence,
training progress, PGN preview behaviour, HTTP endpoints, callback moves and
resets, drag-action validation, board highlights and the Cloud entry point.

The separate Playwright/Microsoft Edge browser check passed for actual clicks,
mouse dragging, illegal and off-board drops, keyboard play, refresh persistence,
knight promotion, AI replies and touch dragging. Desktop and mobile screenshots
were generated in ignored `artifacts/` and visually reviewed.

JavaScript syntax and `git diff --check` also passed. Windows line-ending
normalization notices were present but did not indicate whitespace failures.

GitHub Actions is configured to install dependencies and run pytest on pushes
and pull requests using Python 3.11. A hosted Actions run has not been verified;
the browser check is separate from that workflow.

## 12. Commands

Run these from the project root in PowerShell.

### Install and run

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python main.py
```

Open `http://127.0.0.1:8050`.

### Rebuild the combined dataset

```powershell
.\.venv\Scripts\python data/pgn/prepare.py
```

The three extracted source PGNs must already be in `data/pgn/`.

### Train with progress and board previews

```powershell
.\.venv\Scripts\python -m plotly_chess.train data/pgn/training.pgn --output chess-npc.npz --show-games
```

Optional overrides include `--epochs 100` and `--seed 42`.

### Run the trained opponent

```powershell
$env:PLOTLY_CHESS_MODEL = "$PWD\chess-npc.npz"
.\.venv\Scripts\python main.py
```

Restart the app after training or changing the artifact; the model is loaded
at application setup, not on every move.

### Run automated tests

```powershell
.\.venv\Scripts\python -m pytest -q
node --check assets/board.js
```

Node is needed for that optional syntax-check command, not to run the app.

### Run the optional browser check

Start a test server in one terminal:

```powershell
.\.venv\Scripts\python -c "from plotly_chess.app import create_app; create_app().run(port=8060, debug=False)"
```

In another terminal, using a Python environment with Playwright and installed
Microsoft Edge:

```powershell
python tests/browser_check.py
```

### Prepare the Cloud upload

Without a trained model:

```powershell
.\.venv\Scripts\python scripts/prepare_cloud.py
```

With a trained model:

```powershell
.\.venv\Scripts\python scripts/prepare_cloud.py --model chess-npc.npz
```

For the latter, set the Cloud environment variable
`PLOTLY_CHESS_MODEL=chess-npc.npz`.

## 13. Remaining scope

- Remote multiplayer, accounts and shared game storage are not implemented.
- The NPC implementation is a small adaptation with coarse concepts; it has
  not been shown to improve chess strength.
- Full-corpus training, benchmark matches and hosted Cloud deployment were
  not completed as part of this work.
- AI evaluation runs synchronously and uses a shallow search.
- Training does not resume from checkpoints or automatically deploy artifacts.
- The board stays oriented with White at the bottom; there is no flip control.
- Game clocks, undo, game export and a manual draw-claim flow are not included.
- Generated datasets, artifacts, screenshots and deployment copies are local
  files excluded from Git and need separate handling if moving machines.
