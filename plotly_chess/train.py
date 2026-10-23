"""Train chess experts and gate from completed PGN games, split by game."""
import argparse
from pathlib import Path
import time
import chess.pgn
import numpy as np
from .ai import CircuitRouter, concepts

class Progress:
    """Flushed, throttled terminal output; no extra dependencies or sleeps."""
    def __init__(self, show_games=False, interval=2.0):
        self.show_games = show_games
        self.interval = interval
        self.last = 0
        self.stage = None
        self.started = time.monotonic()

    def update(self, stage, current, total, loss=None, detail='', board=None, force=False):
        now = time.monotonic()
        if stage != self.stage:
            self.started, self.stage, self.last = now, stage, 0
        if not force and current != total and now-self.last < self.interval:
            return
        self.last = now
        elapsed = now-self.started
        percent = 100*current/max(total,1)
        eta = elapsed*(total-current)/current if current else 0
        metric = f' | fitting NLL {loss:.4f}' if loss is not None else ''
        print(f'{stage}: {percent:5.1f}% ({current:,}/{total:,}) | {elapsed:.0f}s elapsed | ETA {eta:.0f}s{metric} {detail}', flush=True)
        if self.show_games and board is not None:
            print(str(board) + '\n  a b c d e f g h', flush=True)


def load_games(path, progress=None):
    games = []
    total = Path(path).stat().st_size
    read, skipped, positions = 0, 0, 0
    with open(path, encoding="utf-8-sig") as stream:
        while (game := chess.pgn.read_game(stream)) is not None:
            read += 1
            result = game.headers.get('Result')
            if result not in ('1-0','0-1','1/2-1/2') or game.errors:
                skipped += 1
                continue
            rows, labels = [], []
            board = game.board()
            title = f"{game.headers.get('White','?')} vs {game.headers.get('Black','?')}"
            for move in game.mainline_moves():
                san = board.san(move) if progress and progress.show_games else ''
                board.push(move)
                if progress and progress.show_games:
                    progress.update('Loading PGN', min(stream.tell(),max(total-1,0)), total,
                                    detail=f'Game {read:,}: {title} | ply {board.ply()} {san}', board=board)
                if board.is_game_over(claim_draw=True):
                    continue
                rows.append(concepts(board))
                labels.append(1 if result == '1/2-1/2' else (2 if (result == '1-0') == board.turn else 0))
            if rows:
                games.append((rows,labels))
                positions += len(rows)
            else:
                skipped += 1
            if progress:
                progress.update('Loading PGN', min(stream.tell(),max(total-1,0)), total,
                                detail=f'games {len(games):,} | positions {positions:,} | skipped {skipped:,}')
    if progress:
        progress.update('Loading PGN',total,total,detail=f'games {len(games):,} | positions {positions:,} | skipped {skipped:,}',force=True)
    return games

def flatten(games):
    return np.concatenate([np.asarray(g[0]) for g in games]), np.concatenate([np.asarray(g[1]) for g in games])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pgn')
    parser.add_argument('--output', default='chess-npc.npz')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=300)
    parser.add_argument('--show-games',action='store_true',help='Show sampled boards as recorded PGN games are replayed (not self-play).')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    progress = Progress(show_games=args.show_games)
    print(f'Loading {args.pgn}. Progress uses file bytes; ETA is approximate.',flush=True)
    if args.show_games:
        print('Preview: replaying recorded games; boards sampled every two seconds without slowing training.',flush=True)
    games = load_games(args.pgn,progress)
    if len(games) < 10:
        parser.error('Provide at least 10 completed, valid games; many more are recommended.')
    np.random.default_rng(args.seed).shuffle(games)
    train_end,gate_end = int(.6*len(games)),int(.8*len(games))
    x,y = flatten(games[:train_end])
    gx,gy = flatten(games[train_end:gate_end])
    tx,ty = flatten(games[gate_end:])
    print(f'Split: {train_end:,} expert / {gate_end-train_end:,} gate / {len(games)-gate_end:,} test games.',flush=True)
    model = CircuitRouter.fit(x,y,gx,gy,seed=args.seed,epochs=args.epochs,progress=progress.update)
    print('Evaluating held-out positions...',flush=True)
    p,_ = model.predict(tx)
    experts = model.experts(tx)
    true = experts[np.arange(len(ty)),:,ty]
    nll = lambda prob: float(-np.log(np.maximum(prob,1e-12)).mean())
    loss = nll(p[np.arange(len(ty)),ty])
    oracle = nll(true.max(1))
    print(f'Held-out positions: {len(ty)}; NLL: {loss:.4f}; uniform NLL: {nll(true.mean(1)):.4f}; oracle NLL: {oracle:.4f}; routing regret: {loss-oracle:.4f}')
    model.save(args.output)
    print(f'Saved {args.output}')

if __name__ == '__main__':
    main()
