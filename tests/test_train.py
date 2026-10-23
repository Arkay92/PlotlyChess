import sys
import numpy as np
from plotly_chess.train import main, Progress, load_games
from plotly_chess.ai import CircuitRouter, concepts
import chess

def test_pgn_training_end_to_end(tmp_path, monkeypatch, capsys):
    pgn = tmp_path/'games.pgn'
    # Tiny synthetic corpus checks plumbing only, not predictive quality.
    pgn.write_text(('''[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1

''')*10,encoding='utf-8')
    artifact = tmp_path/'model.npz'
    monkeypatch.setattr(sys,'argv',['train',str(pgn),'--output',str(artifact)])
    main()
    output = capsys.readouterr().out
    for stage in ['Loading PGN:','Experts:','Neural gate:','fitting NLL','routing regret']:
        assert stage in output
    p,_ = CircuitRouter.load(artifact).predict(np.array([concepts(chess.Board())]))
    np.testing.assert_allclose(p.sum(1),1)

def test_recorded_game_preview_does_not_change_training_data(tmp_path, capsys):
    path = tmp_path/'preview.pgn'
    path.write_text('[White "Alice"]\n[Black "Bob"]\n[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1\n',encoding='utf-8')
    expected = load_games(path)
    actual = load_games(path,Progress(show_games=True,interval=0))
    assert len(expected) == len(actual) == 1
    np.testing.assert_array_equal(expected[0][0],actual[0][0])
    assert expected[0][1] == actual[0][1]
    output = capsys.readouterr().out
    assert 'Alice vs Bob' in output
    assert 'ply 4 Qh4#' in output
    assert 'a b c d e f g h' in output
