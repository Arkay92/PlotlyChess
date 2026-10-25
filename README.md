# Plotly Chess

A Dash/Plotly chess game with independent browser sessions, legal move validation,
SAN/UCI input, promotion selection, hints, local two-player play and live
WebSocket multiplayer rooms.

The responsive playing surface uses interactive HTML squares and bundled SVG
pieces. Drag with a mouse, pen or touch, or click a piece and its destination.
Legal destinations, the previous move and a checked king are highlighted. Drop
outside the board or press Escape during a drag to cancel. Keyboard users can
Tab between squares (or use arrow keys) and press Enter/Space to select and move.
Every submitted move is checked by python-chess on the server. Use the side
panel for hints, new games, move history and notation/promotion options. Game
state and play mode survive a refresh within the browser session.

## Run

Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python main.py
```

Open http://127.0.0.1:8050. Run tests with
`.\.venv\Scripts\python -m pytest`.

## Plotly Cloud upload

`app.py` is the Cloud entry point. It declares `app = Dash(...)` at module scope
so upload scanners can identify it; the UI and callbacks remain modular.
`main.py` remains available for local development.

Prepare a small upload folder:

```powershell
.\.venv\Scripts\python scripts/prepare_cloud.py
```

In Plotly Cloud, remove the previous upload and upload the **contents** of
`deploy/plotly-cloud/`, keeping `assets/` and `plotly_chess/` as folders.
Select **app.py** as the main file and **Python 3.11**. The folder includes
`requirements.txt` and excludes the virtual environment, datasets and tests.
Uploading only the entry point will fail because it imports the package.

To deploy a trained model, prepare with `--model chess-npc.npz` and set the Cloud
environment variable `PLOTLY_CHESS_MODEL` to `chess-npc.npz`. Without a model,
material search remains available. See the
[official publishing guide](https://dash.plotly.com/plotly-cloud/publish).

## Neural probabilistic circuits

`plotly_chess/ai.py` adapts the supplied v18 conditional expert routing idea to
chess. It does not import or execute the notebook, install packages at runtime,
download its intrusion dataset, or use intrusion-classification weights.

Each expert is a normalized categorical sum/product circuit: a sum over three
outcome classes (loss/draw/win) with a product of categorical concept likelihoods.
Three bootstrap experts learn material balance, mobility, check, game phase and
capture availability from completed chess games. A tanh neural gate combines
their posteriors using entropy, margin, win/loss mass and KL disagreement.
This is a smaller chess-specific adaptation, not the notebook's full joint
hierarchical circuit architecture or pretrained model.

Train from your own PGN collection:

A downloaded, validated starter collection is available locally at
`data/pgn/training.pgn`. See [dataset sources and preparation](data/pgn/README.md).

```powershell
.\.venv\Scripts\python -m plotly_chess.train games.pgn --output chess-npc.npz
$env:PLOTLY_CHESS_MODEL = 'chess-npc.npz'
.\.venv\Scripts\python main.py
```

Training prints loading percentage, elapsed time, approximate ETA, expert fitting
progress and neural-gate fitting NLL. Add `--show-games` for terminal board previews:

```powershell
.\.venv\Scripts\python -m plotly_chess.train data/pgn/training.pgn --output chess-npc.npz --show-games
```

Previews sample the recorded game being processed every two seconds without
delaying training. They are PGN replay, not live self-play. Board previews appear
during data loading; expert and gate fitting show numerical progress. Use
`--epochs 100` to change the number of gate training epochs (default 300).

Training splits whole games 60/20/20 for experts, gate fitting and held-out
evaluation. It reports gated/uniform/oracle negative log likelihood and routing
regret. At least ten games are required; use a large, varied collection including
wins for both colours and draws. Duplicate games should be removed before training.
The reported held-out result is necessary evidence: routing is not guaranteed to
improve over uniform averaging or the fallback. No trained chess artifact ships.

The AI searches two plies using material and terminal results. When a trained
model is configured, its routed outcome prediction adds a bounded evaluation
term; opponent-to-move posteriors are inverted correctly. Without an artifact,
the UI explicitly identifies material search. Neither mode claims engine strength.

## Structure and scope

- `game.py`: immutable transitions and move-history replay, including repetition.
- `board.py`: SVG pieces, interactive squares and highlights; optional Plotly figure renderer.
- `multiplayer.py`: server-authoritative multiplayer room registry and RFC 6455 WebSocket transport (`/ws`).
- `assets/board.js`: pointer gestures, drag previews, keyboard navigation and live WebSocket synchronization.
- `ai.py`: search, categorical circuits, neural gating and safe NPZ persistence.
- `train.py`: PGN ingestion and evaluation.
- `app.py`: application factory and one callback coordinating game state.
- `tests/`: rule, session, multiplayer, AI, persistence and HTTP smoke checks.

Selecting **Multiplayer** lets a player choose **Create game** (generating a
6-character room code to share with a friend) or **Join game** (entering a
friend's room code). Once both players are connected, moves are validated on the
server and pushed in real time over WebSockets (`/ws`) via
`window.dash_clientside.set_props` so White and Black take turns on the board.
There is no language-model question box: hints now come from the actual legal
move selector. Model loading is explicit through the environment variable, with
errors surfaced at startup.

Draw claims are automatically accepted whenever python-chess permits a claim.
Changing mode starts a new game. The WSGI entry point is `main:server`; importing
it does not bind sockets or download models. AI search runs synchronously, suitable
for this small search depth rather than heavy concurrent workloads.

An optional browser check covers actual mouse/touch moves, rejected drops,
keyboard controls, session persistence, underpromotion and AI replies. With
Playwright and Microsoft Edge installed, start the app on port 8060 and run
`python tests/browser_check.py`. Screenshots are written to ignored `artifacts/`.
