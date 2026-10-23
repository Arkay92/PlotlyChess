"""Validate and deduplicate the downloaded standard-chess PGNs for training.

Run from the repository root: .venv/Scripts/python data/pgn/prepare.py
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import chess
import chess.pgn


def main():
    root = Path(__file__).resolve().parent
    seen = set()
    report = {"sources": {}, "results": Counter()}
    destination = root / "training.pgn"
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for name in ("Carlsen", "Kasparov", "Anand"):
            source = root / f"{name}.pgn"
            counts = Counter()
            with source.open(encoding="utf-8-sig", errors="replace") as stream:
                while (game := chess.pgn.read_game(stream)) is not None:
                    counts["read"] += 1
                    result = game.headers.get("Result")
                    if (game.errors or game.board().fen() != chess.STARTING_FEN
                            or result not in ("1-0", "0-1", "1/2-1/2")):
                        counts["excluded"] += 1
                        continue
                    moves = " ".join(move.uci() for move in game.mainline_moves())
                    if not moves:
                        counts["excluded"] += 1
                        continue
                    key = hashlib.sha256(moves.encode("ascii")).hexdigest()
                    if key in seen:
                        counts["duplicates"] += 1
                        continue
                    seen.add(key)
                    counts["kept"] += 1
                    report["results"][result] += 1
                    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
                    output.write(game.accept(exporter) + "\n\n")
            report["sources"][name] = {
                "url": f"https://www.pgnmentor.com/players/{name}.zip",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                **counts,
            }
    report["total_games"] = len(seen)
    report["training_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
