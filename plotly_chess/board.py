"""Plotly rendering independent of callbacks."""
import chess
import chess.svg
from functools import lru_cache
from urllib.parse import quote
from dash import html
import plotly.graph_objects as go


@lru_cache(maxsize=12)
def piece_image(symbol):
    """Use python-chess's bundled vector pieces, with no external asset requests."""
    svg = chess.svg.piece(chess.Piece.from_symbol(symbol), size=96)
    return 'data:image/svg+xml;charset=utf-8,' + quote(svg)


def interactive_board(board, selected=None, allowed_color=None):
    """A semantic, responsive board. Pointer events are bridged by board.js."""
    ended = board.is_game_over(claim_draw=True)
    can_act = not ended and (allowed_color is None or board.turn == allowed_color)
    moves = list(board.legal_moves) if can_act else []
    destinations = {m.to_square for m in moves if m.from_square == selected}
    last = board.peek() if board.move_stack else None
    squares = []
    for rank in range(7, -1, -1):
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            name = chess.square_name(square)
            classes = ['square', 'light' if (rank+file) % 2 else 'dark']
            if last and square in (last.from_square, last.to_square):
                classes.append('last-move')
            if square == selected:
                classes.append('selected')
            if square in destinations:
                classes.append('legal-capture' if piece else 'legal-target')
            if piece and piece.piece_type == chess.KING and piece.color == board.turn and board.is_check():
                classes.append('in-check')
            children = []
            if file == 0:
                children.append(html.Span(str(rank+1), className='rank-coordinate', **{'aria-hidden':'true'}))
            if rank == 0:
                children.append(html.Span(chess.FILE_NAMES[file], className='file-coordinate', **{'aria-hidden':'true'}))
            if piece:
                children.append(html.Img(src=piece_image(piece.symbol()), className='piece', alt='', draggable='false'))
            label = f'{name}, ' + (f'{"white" if piece.color else "black"} {chess.piece_name(piece.piece_type)}' if piece else 'empty')
            squares.append(html.Button(children, id=f'square-{name}', className=' '.join(classes),
                title=label, type='button', **{
                    'aria-label':label, 'aria-pressed':str(square == selected).lower(),
                    'data-square':str(square),
                    'data-movable':str(bool(piece and piece.color == board.turn and can_act)).lower(),
                    'data-legal':','.join(str(m.to_square) for m in moves if m.from_square == square)}))
    return html.Div(squares, className='chess-grid', **{'aria-label':'Chessboard. White at the bottom.', 'data-fen':board.fen()})

def figure(board, selected=None):
    legal = {m.to_square for m in board.legal_moves if m.from_square == selected}
    fig = go.Figure()
    for square in chess.SQUARES:
        x, y = chess.square_file(square), chess.square_rank(square)
        color = "#d8b65c" if square == selected else ("#a8bd8a" if square in legal else ("#ede1c9" if (x+y) % 2 else "#69826b"))
        fig.add_shape(type="rect", x0=x, y0=y, x1=x+1, y1=y+1, fillcolor=color, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=[chess.square_file(s)+.5 for s in chess.SQUARES],
        y=[chess.square_rank(s)+.5 for s in chess.SQUARES],
        text=[board.piece_at(s).unicode_symbol() if board.piece_at(s) else "" for s in chess.SQUARES],
        customdata=list(chess.SQUARES), mode="markers+text", marker={"size":48,"opacity":0},
        textfont={"size":38,"color":"#16201c"},
        hovertext=[chess.square_name(s) for s in chess.SQUARES], hoverinfo="text"))
    fig.update_layout(height=580, margin=dict(l=25,r=15,t=15,b=25), dragmode=False, showlegend=False)
    fig.update_xaxes(range=[0,8], tickvals=[i+.5 for i in range(8)], ticktext=list("abcdefgh"), fixedrange=True, showgrid=False, zeroline=False)
    fig.update_yaxes(range=[0,8], tickvals=[i+.5 for i in range(8)], ticktext=list("12345678"), fixedrange=True, showgrid=False, zeroline=False, scaleanchor="x")
    return fig
