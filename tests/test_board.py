import chess
from plotly_chess.board import interactive_board


def test_board_orientation_legality_and_svg_pieces():
    grid = interactive_board(chess.Board(),chess.E2)
    squares = {int(s.to_plotly_json()['props']['data-square']):s.to_plotly_json()['props'] for s in grid.children}
    assert len(squares) == 64
    assert grid.children[0].id == 'square-a8'
    assert grid.children[-1].id == 'square-h1'
    assert squares[chess.E2]['data-legal'] == '20,28'
    assert squares[chess.E2]['data-movable'] == 'true'
    assert squares[chess.E7]['data-movable'] == 'false'
    assert 'legal-target' in squares[chess.E4]['className']
    assert 'selected' in squares[chess.E2]['className']
    image = squares[chess.E2]['children'][-1]
    assert image.src.startswith('data:image/svg+xml')


def test_last_move_and_check_highlights():
    board = chess.Board()
    for san in ('f3','e5','g4','Qh4#'):
        board.push_san(san)
    squares = {s.id:s.to_plotly_json()['props'] for s in interactive_board(board).children}
    assert 'in-check' in squares['square-e1']['className']
    assert 'last-move' in squares['square-h4']['className']
    assert all(s['data-movable'] == 'false' for s in squares.values())
