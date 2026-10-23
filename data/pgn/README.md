# Downloaded chess games

Downloaded on 23 October 2026 from [PGN Mentor](https://www.pgnmentor.com/files.html):

- [Magnus Carlsen archive](https://www.pgnmentor.com/players/Carlsen.zip)
- [Garry Kasparov archive](https://www.pgnmentor.com/players/Kasparov.zip)
- [Viswanathan Anand archive](https://www.pgnmentor.com/players/Anand.zip)

The original ZIPs and extracted PGNs are retained locally. `training.pgn` combines
the three collections, keeps completed standard games from the initial position,
excludes parser errors and empty games, removes duplicate mainline move sequences,
and strips comments and variations. Result labels come from the source PGN headers;
resignations and agreed draws need not end in a terminal board position.

`manifest.json` records exact counts, source URLs and SHA-256 hashes.
The downloaded collections and generated PGN are ignored by Git.

Train from the repository root:

```powershell
.\.venv\Scripts\python -m plotly_chess.train data/pgn/training.pgn --output chess-npc.npz
$env:PLOTLY_CHESS_MODEL = "$PWD\chess-npc.npz"
.\.venv\Scripts\python main.py
```

To reproduce the combined file after downloading and extracting those archives:

```powershell
.\.venv\Scripts\python data/pgn/prepare.py
```

This is a convenient initial experiment with master games. It is not a balanced
sample of all chess play or evidence that the model improves playing strength.
Whole-game splitting prevents the same game crossing partitions; common opening
positions can still occur in different games. PGN Mentor offers free downloads;
refer to the source for any redistribution terms.

For a larger subsequent dataset, [Lichess's open database](https://database.lichess.org/)
provides standard rated games under CC0. Its monthly archives are compressed `.pgn.zst`
files and must be decompressed and sampled before using this in-memory trainer.
