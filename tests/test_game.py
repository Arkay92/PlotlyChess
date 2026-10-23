import chess
import pytest
from plotly_chess import game
from plotly_chess.ai import ChessAI

def test_sessions_and_illegal_moves():
    first, second = game.new_game(),game.new_game()
    updated,_ = game.play(first,'e4')
    assert game.restore(second).fen() == chess.STARTING_FEN
    assert first['moves'] == []
    assert game.restore(updated).turn == chess.BLACK
    with pytest.raises(ValueError):
        game.play(first,'e5')

def test_click_selection_and_move():
    state,msg,moved = game.click(game.new_game(),chess.E2)
    assert not moved and state['selected'] == chess.E2
    state,msg,moved = game.click(state,chess.E4)
    assert moved and state['moves'] == ['e2e4']

def test_repetition_survives_serialization():
    state = game.new_game()
    for move in (['Nf3','Nf6','Ng1','Ng8']*2)[:7]:
        state,_ = game.play(state,move)
    assert game.restore(state).can_claim_threefold_repetition()
    assert 'Game over' in game.status(game.restore(state))

def test_checkmate_and_ai_no_move():
    state = game.new_game()
    for move in ['f3','e5','g4','Qh4#']:
        state,_ = game.play(state,move)
    with pytest.raises(ValueError):
        ChessAI().choose(game.restore(state))
    with pytest.raises(ValueError):
        game.play(state,'a3')

def test_ai_takes_mate_without_mutation():
    board = chess.Board('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1')
    original = board.fen()
    move,_ = ChessAI().choose(board)
    assert board.fen() == original
    board.push(move)
    assert board.is_checkmate()

def test_castling_and_en_passant():
    state = game.new_game()
    for move in ['e4','a6','e5','d5','exd6']:
        state,_ = game.play(state,move)
    assert game.restore(state).piece_at(chess.D5) is None
    state = game.new_game()
    for move in ['e4','e5','Nf3','Nc6','Bc4','Nf6','O-O']:
        state,_ = game.play(state,move)
    assert game.restore(state).piece_at(chess.F1).piece_type == chess.ROOK

def test_underpromotion_click():
    state = game.new_game()
    for move in ['a4','h5','a5','h4','a6','h3','axb7','hxg2']:
        state,_ = game.play(state,move)
    state,_,_ = game.click(state,chess.B7)
    state,_,moved = game.click(state,chess.A8,'n')
    assert moved
    assert game.restore(state).piece_at(chess.A8) == chess.Piece(chess.KNIGHT,chess.WHITE)
