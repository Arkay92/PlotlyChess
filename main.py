import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import chess
import random

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
board = chess.Board()
selected_square = None
reset_timestamp = 0

def generate_board_figure(selected=None):
    fig = go.Figure()
    base_colors = ['#cfbf90', '#7b5e09']
    highlight_colors = ['#decba4', '#9c7c2a']
    square_size = 70
    piece_size = 50
    piece_font_size = 30
    chess_symbols = {
        chess.PAWN: {'white': '♙', 'black': '♟'},
        chess.ROOK: {'white': '♖', 'black': '♜'},
        chess.KNIGHT: {'white': '♘', 'black': '♞'},
        chess.BISHOP: {'white': '♗', 'black': '♝'},
        chess.QUEEN: {'white': '♕', 'black': '♛'},
        chess.KING: {'white': '♔', 'black': '♚'}
    }

    for rank in range(8):
        for file in range(8):
            flipped_rank = rank
            square_index = chess.square(file, flipped_rank)
            is_selected = selected is not None and selected == square_index
            cell_color = highlight_colors[(rank + file) % 2] if is_selected else base_colors[(rank + file) % 2]
            fig.add_trace(go.Scatter(
                x=[file + 0.5], y=[rank + 0.5],
                marker=dict(color=cell_color, size=square_size),
                mode='markers',
                marker_symbol='square',
                line=dict(width=0)
            ))

            piece = board.piece_at(square_index)
            if piece:
                color = 'white' if piece.color == chess.WHITE else 'black'
                symbol = chess_symbols[piece.piece_type][color]
                fig.add_trace(go.Scatter(
                    x=[file + 0.5], y=[rank + 0.5],
                    text=[symbol],
                    mode='text',
                    textfont=dict(color=color, size=piece_font_size),
                    textposition='middle center',
                    showlegend=False,
                    marker=dict(size=piece_size),
                    name=symbol
                ))

    fig.update_layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        height=square_size * 8, width=square_size * 8,
        dragmode=False, showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 8]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 8])
    )
    return fig

def choose_computer_move(board):
    capture_moves = [move for move in board.legal_moves if board.is_capture(move)]
    
    if capture_moves:
        return random.choice(capture_moves)
    return random.choice(list(board.legal_moves))

app.layout = html.Div([
    dcc.Graph(
        id='chessboard',
        config={'staticPlot': False, 'scrollZoom': False, 'editable': False, 'displayModeBar': False}
    ),
    html.Div(id='feedback-message', style={'textAlign': 'center', 'fontSize': 20, 'color': 'red'}),
    dcc.Store(id='selected-pos'),
    html.Div(id='selected-piece'),
    html.Button('Reset Game', id='reset-button', n_clicks_timestamp=0, style={'marginTop': '20px'})
])

@app.callback(
    [Output('chessboard', 'figure'), Output('feedback-message', 'children')],
    [Input('chessboard', 'clickData'), State('selected-pos', 'data')],
    [Input('reset-button', 'n_clicks_timestamp')]
)
def update_chessboard(click_data, selected_pos, reset_clicks_timestamp):
    global selected_square, board, reset_timestamp
    message = ''

    # Handle reset button
    if reset_clicks_timestamp > reset_timestamp:
        reset_timestamp = reset_clicks_timestamp
        board.reset()
        selected_square = None
        return generate_board_figure(), 'Game has been reset.'

    # Handle player's move
    if click_data:
        point = click_data['points'][0]
        clicked_rank = point['y'] - 0.5
        clicked_file = point['x'] - 0.5
        clicked_square = chess.square(int(clicked_file), int(clicked_rank))

        if selected_square is None:
            selected_square = clicked_square
        else:
            move = chess.Move(selected_square, clicked_square)
            if move in board.legal_moves:
                board.push(move)
                message = 'Your move was successful!'

                # Computer's response
                if not board.is_game_over():
                    computer_move = choose_computer_move(board)
                    board.push(computer_move)
                    message += ' Computer has moved.'
                else:
                    result = board.result()
                    message += f' Game over! Result: {result}'
            else:
                message = 'Invalid move! Try again.'

            selected_square = None
    else:
        selected_square = selected_pos

    return generate_board_figure(selected_square), message

@app.callback(
    Output('selected-pos', 'data'),
    [Input('chessboard', 'clickData')]
)
def update_selected_position(click_data):
    if click_data:
        point = click_data['points'][0]
        clicked_rank = point['y'] - 0.5
        clicked_file = point['x'] - 0.5
        return chess.square(int(clicked_file), int(clicked_rank))
    return None

if __name__ == '__main__':
    app.run_server(debug=True)
