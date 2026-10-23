import numpy as np
import chess
from plotly_chess.ai import ChessAI
from plotly_chess.ai import CircuitRouter, CARDINALITIES

def test_circuit_training_normalization_and_roundtrip(tmp_path):
    rng = np.random.default_rng(12)
    x = np.column_stack([rng.integers(k,size=120) for k in CARDINALITIES])
    y = (x[:,0] % 3).astype(int)
    model = CircuitRouter.fit(x[:80],y[:80],x[80:100],y[80:100],epochs=20)
    p,w = model.predict(x[100:])
    assert np.isfinite(p).all()
    assert (p>=0).all() and (w>=0).all()
    np.testing.assert_allclose(p.sum(1),1)
    np.testing.assert_allclose(w.sum(1),1)
    path = tmp_path/'model.npz'
    model.save(path)
    np.testing.assert_allclose(CircuitRouter.load(path).predict(x[100:])[0],p)
    ai = ChessAI(path)
    board = chess.Board()
    move, explanation = ai.choose(board)
    assert move in board.legal_moves
    assert 'Expert weights' in explanation
