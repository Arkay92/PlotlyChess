from dash import dcc, html, Input, Output, State
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess import Move

import plotly.graph_objects as go
import chess, random, dash, torch, os, re, logging

logging.basicConfig(level=logging.DEBUG)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
board = chess.Board()
selected_square = None
start_square = None
reset_timestamp = 0

# Load the tokenizer and model for chesspythia-70m
tokenizer = AutoTokenizer.from_pretrained("mlabonne/chesspythia-70m")
model = AutoModelForCausalLM.from_pretrained("mlabonne/chesspythia-70m").to('cuda' if torch.cuda.is_available() else 'cpu')

# Define app layout
app.layout = html.Div([
    dcc.Graph(
        id='chessboard',
        config={'staticPlot': False, 'scrollZoom': False, 'editable': False, 'displayModeBar': False}
    ),
    html.Div(id='feedback-message', style={'textAlign': 'center', 'fontSize': 20, 'color': 'red'}),
    dcc.Store(id='selected-pos'),
    dcc.Store(id='start-square'),
    html.Div([
        dcc.Input(id='llm-input', type='text', placeholder='Ask a question'),
        html.Button('Submit', id='submit-llm-input', n_clicks=0)
    ], style={'textAlign': 'center', 'marginTop': '20px'}),
    html.Button('Reset Game', id='reset-button', n_clicks_timestamp=0, style={'marginTop': '20px'})
])

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
            square_index = chess.square(file, rank)
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

def get_piece_name_from_letter(letter):
    """ Returns the full name of the chess piece from its letter notation. """
    return {
        'N': 'Knight',
        'B': 'Bishop',
        'R': 'Rook',
        'Q': 'Queen',
        'K': 'King',
        '': 'Pawn'  # Pawns are not represented by a letter in algebraic notation.
    }.get(letter, '')

def get_piece_name(piece_type):
    """ Returns the name of the chess piece by its type. """
    return {
        chess.PAWN: 'Pawn',
        chess.ROOK: 'Rook',
        chess.KNIGHT: 'Knight',
        chess.BISHOP: 'Bishop',
        chess.QUEEN: 'Queen',
        chess.KING: 'King'
    }.get(piece_type, 'Piece')

def uci_to_human(board, uci_move):
    move = chess.Move.from_uci(uci_move)
    piece = board.piece_at(move.from_square)
    if not piece:
        return "Invalid move: No piece at the start square."

    # Get the basic move description
    piece_name = get_piece_name(piece.piece_type)  # Using existing function get_piece_name()
    start_square = chess.SQUARE_NAMES[move.from_square]
    end_square = chess.SQUARE_NAMES[move.to_square]
    move_description = f"{piece_name} from {start_square.upper()} to {end_square.upper()}"

    # Check for capture
    if board.is_capture(move):
        move_description += " capturing a piece"

    # Check for check or checkmate
    board.push(move)  # Make the move on a copy of the board to check for check/checkmate
    if board.is_checkmate():
        move_description += " delivering checkmate"
    elif board.is_check():
        move_description += " delivering check"
    board.pop()  # Undo the move

    return move_description

def format_move(move):
    """ Formats the chess move into a more human-readable format. """
    piece = board.piece_at(move.from_square)
    piece_name = get_piece_name(piece.piece_type) if piece else 'Piece'
    to_square = chess.square_name(move.to_square).capitalize()
    return f'{piece_name} to {to_square}'

def choose_computer_move(board):
    """ Choose a move for the computer and format it. """
    capture_moves = [move for move in board.legal_moves if board.is_capture(move)]
    selected_move = random.choice(capture_moves if capture_moves else list(board.legal_moves))
    formatted_move = format_move(selected_move)
    board.push(selected_move)
    return f'Computer moved {formatted_move}.'

def generate_prompt(fen_position, question):
    return f"We're playing a game of chess. I'm playing white. The current chessboard position is '{fen_position}'. {question}"

def extract_and_format_moves(text):
    """Extracts and formats chess moves from the LLM-generated text."""
    import re
    # Define a regular expression to find common chess notation patterns like Nf3, e4, etc.
    pattern = r'\b([NBRQK]?[a-h]?[1-8]?[x-]?[a-h][1-8](=[NBRQ])?|O-O(-O)?)\b'
    matches = re.findall(pattern, text)
    
    # Convert moves to a readable format or fallback to original text if none found
    formatted_moves = ' '.join([match[0] for match in matches]) if matches else text
    return formatted_moves

def clean_up_repetitive_moves(output_str):
    import re
    moves = re.findall(r'\b([NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](=[NBRQ])?)(?=\s|\b)', output_str)
    cleaned_moves = []
    last_move = None

    for move in moves:
        if move != last_move:
            cleaned_moves.append(move)
        last_move = move

    return ' '.join(cleaned_moves)

def clean_model_output(output_str):
    """Clean up unnecessary annotations from model output."""
    cleaned_output = re.sub(r"\{[^}]*\}", "", output_str)  # Removes anything inside curly braces
    return cleaned_output

def get_llm_response(question, fen_position):
    """Get a response from the language model regarding the current game state."""
    prompt = generate_prompt(fen_position, question)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50)
    output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_output = clean_model_output(output_str)
    logging.debug(f"Model output: {output_str}")

    # Extract meaningful moves from the model's output
    moves = extract_moves(cleaned_output)
    logging.debug(f"Validating moves on board with FEN: {board.fen()}")
    valid_move = filter_to_one_valid_move(moves, board)
    if valid_move:
        move = board.san(chess.Move.from_uci(valid_move))
        return f'Best move according to the model: {move}'
    else:
        logging.debug(f"None of the suggested moves were valid: {moves}")
        return f"Model suggested invalid moves: {', '.join(moves)}"

def extract_moves(text):
    """Extract chess moves from a given text using regex patterns."""
    pattern = r'\b([NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](=[NBRQ])?|O-O(-O)?)\b'
    matches = re.findall(pattern, text)
    moves = [match[0] for match in matches]
    unique_moves = []

    # Adding filtering for repetitive moves
    for move in moves:
        if move not in unique_moves:
            unique_moves.append(move)
    return unique_moves

def filter_to_one_valid_move(moves, board):
    """Find the first valid move that is legal on the board or report back."""
    last_error = ""
    for move_notation in moves:
        try:
            move = board.parse_san(move_notation)
            if move in board.legal_moves:
                return move.uci()
        except ValueError as e:
            last_error = str(e)  # Save the last error for logging or feedback
    logging.debug(f"All moves were invalid. Last parsing error: {last_error}")
    return None

def parse_move(notation):
    """ Parse a single chess move notation into a chess.Move object, if valid. """
    try:
        return Move.from_uci(notation)
    except:
        # Attempt to translate SAN to UCI if provided move is in SAN format
        try:
            return board.parse_san(notation)
        except:
            return None

@app.callback(
    [Output('chessboard', 'figure'), Output('feedback-message', 'children')],
    [Input('chessboard', 'clickData'),
     Input('submit-llm-input', 'n_clicks')],
    [State('selected-pos', 'data'), State('start-square', 'data'),
     State('llm-input', 'value'),
     State('reset-button', 'n_clicks_timestamp')]
)
def update_chessboard_and_get_response(click_data, llm_n_clicks, selected_pos, start_pos, question, reset_clicks_timestamp):
    global selected_square, start_square, board, reset_timestamp
    message = ''
    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if reset_clicks_timestamp > reset_timestamp:
        reset_timestamp = reset_clicks_timestamp
        board.reset()
        selected_square = None
        start_square = None
        return generate_board_figure(), 'Game has been reset.'

    if click_data and input_id == 'chessboard':
        point = click_data['points'][0]
        clicked_rank = int(point['y'] - 0.5)
        clicked_file = int(point['x'] - 0.5)
        clicked_square = chess.square(clicked_file, clicked_rank)

        if selected_square is None:
            selected_square = clicked_square
            start_square = clicked_square
        else:
            move = chess.Move(start_square, clicked_square)
            if move in board.legal_moves:
                board.push(move)
                formatted_move = format_move(move)
                message = f'Your move was successful! {formatted_move}.'

                if not board.is_game_over():
                    computer_move = choose_computer_move(board)
                    message += f' {computer_move}'
                else:
                    result = board.result()
                    message += f' Game over! Result: {result}'
            else:
                message = 'Invalid move! Try again.'

            selected_square = None
            start_square = None

    if llm_n_clicks > 0 and question and input_id == 'submit-llm-input':
        llm_response = get_llm_response(question, board.fen())
        message += '\n' + llm_response

    return generate_board_figure(selected_square), message

@app.callback(
    Output('chessboard', 'figure', allow_duplicate=True),
    [Input('reset-button', 'n_clicks_timestamp')],
    prevent_initial_call=True
)
def reset_game(reset_clicks_timestamp):
    global board, selected_square, start_square, reset_timestamp
    if reset_clicks_timestamp > reset_timestamp:
        reset_timestamp = reset_clicks_timestamp
        board.reset()
        selected_square = None
        start_square = None
        return generate_board_figure()
    return dash.no_update

@app.callback(
    Output('selected-pos', 'data'),
    [Input('chessboard', 'clickData')]
)
def update_selected_position(click_data):
    if click_data:
        point = click_data['points'][0]
        clicked_rank = int(point['y'] - 0.5)
        clicked_file = int(point['x'] - 0.5)
        return chess.square(clicked_file, clicked_rank)
    return None

if __name__ == '__main__':
    app.run_server(debug=True)
