import os

os.environ["HF_HOME"] = 'E:/torch'
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = 'true'

from dash import dcc, html, Input, Output, State, dash_table
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess import Move
import pandas as pd
import plotly.graph_objects as go
import chess, random, dash, torch, os, re, logging, socket, threading, json

logging.basicConfig(level=logging.DEBUG)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True, prevent_initial_callbacks='initial_duplicate')
server = app.server  # For deployment purposes

board = chess.Board()
selected_square = None
start_square = None
reset_timestamp = 0
in_multiplayer_mode = False  # Flag for multiplayer mode
current_game_id = None  # To track the current game id

# Load the tokenizer and model for chesspythia-70m
tokenizer = AutoTokenizer.from_pretrained("mlabonne/chesspythia-70m")
model = AutoModelForCausalLM.from_pretrained("mlabonne/chesspythia-70m").to('cuda' if torch.cuda.is_available() else 'cpu')

class ChessNetwork:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.peers = []
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Reuse the port
        self.server.bind((self.host, self.port))
        self.server.listen()
        self.game_sessions = {}  # Dictionary to store game sessions with game_id as the key

        thread = threading.Thread(target=self.accept_connections)
        thread.start()

    def accept_connections(self):
        while True:
            client, address = self.server.accept()
            self.peers.append(client)
            threading.Thread(target=self.handle_client, args=(client,)).start()

    def handle_client(self, client):
        while True:
            try:
                data = client.recv(1024).decode('utf-8')
                if data:
                    message = json.loads(data)
                    if message['type'] == 'new_game':
                        self.broadcast_new_game(message['data'])
                    elif message['type'] == 'join_game':
                        self.handle_join_game(message['data'], client)
                    elif message['type'] == 'move':
                        self.broadcast_move(message['data'])
                    # Handle other message types as necessary
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                client.send(str(e).encode('utf-8'))
                break

    def broadcast_new_game(self, game_data):
        game_id = len(self.game_sessions) + 1
        self.game_sessions[game_id] = {'players': [game_data['Player 1']], 'status': 'Waiting', 'board': chess.Board().fen()}
        update = json.dumps({'type': 'update_table', 'data': self.game_sessions})
        self.send_message(update)

    def handle_join_game(self, game_data, client):
        game_id = int(game_data['game_id'])
        if game_id in self.game_sessions and self.game_sessions[game_id]['status'] == 'Waiting':
            self.game_sessions[game_id]['players'].append(game_data['Player 2'])
            self.game_sessions[game_id]['status'] = 'In Progress'
            update = json.dumps({'type': 'update_table', 'data': self.game_sessions})
            self.send_message(update)

    def broadcast_move(self, move_data):
        game_id = int(move_data['game_id'])
        if game_id in self.game_sessions and self.game_sessions[game_id]['status'] == 'In Progress':
            self.game_sessions[game_id]['board'] = move_data['board']
            update = json.dumps({'type': 'move', 'data': move_data})
            self.send_message(update)

    def send_message(self, message, sender=None):
        for peer in self.peers:
            if peer is not sender:
                peer.send(message.encode('utf-8'))

    def connect_to_peer(self, peer_host, peer_port):
        peer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        peer.connect((peer_host, peer_port))
        self.peers.append(peer)

# Initialize ChessNetwork instance
network = ChessNetwork(host="127.0.0.1", port=65432)

# Define initial and game layouts
def initial_layout():
    return html.Div([
        html.Img(src='assets/logo.png', style={'width': '500px', 'display': 'block', 'margin': 'auto', 'margin-bottom': '3rem', 'border-radius': '2rem'}),  # Adjust 'width' as needed
        html.Button("Vs AI", id="vs-ai-button", n_clicks=0, style={'marginRight': '10px', 'background': 'black', 'padding': '0.5rem 1rem', 'color': 'white', 'border': 'none', 'border-radius': '0.3rem', 'cursor': 'pointer'}),
        html.Button("Multiplayer", id="multiplayer-button", n_clicks=0, style={'background': 'black', 'padding': '0.5rem 1rem', 'color': 'white', 'border': 'none', 'border-radius': '0.3rem', 'cursor': 'pointer'}),
    ], style={'textAlign': 'center', 'padding-top': '5rem', 'height': '100%'})

# Define the layout for the multiplayer game mode
def multiplayer_layout():
    columns = [
        {"name": "Player 1", "id": "Player 1"},
        {"name": "Player 2", "id": "Player 2"},
        {"name": "Status", "id": "Status"},
        {"name": "Last Move", "id": "Last Move"},
        {"name": "Action", "id": "Action", "type": "text", "presentation": "markdown"}
    ]

    return html.Div([
        html.H3("Multiplayer Game", style={'textAlign': 'center'}),
        html.Button("Refresh", id="refresh-multiplayer-button", n_clicks=0, style={
            'margin': '10px', 'padding': '10px', 'background-color': '#007BFF', 'color': 'white',
            'border': 'none', 'border-radius': '5px'}),
        html.Button("New Game", id="new-multiplayer-game-button", n_clicks=0, style={
            'margin': '10px', 'padding': '10px', 'background-color': '#28a745', 'color': 'white',
            'border': 'none', 'border-radius': '5px'}),
        dash_table.DataTable(
            id="multiplayer-game-table",
            columns=columns,
            data=[],
            style_table={'margin': '20px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'center', 'padding': '10px'},
            markdown_options={"html": True}
        ),
        html.Div(id='feedback-message', style={'textAlign': 'center', 'fontSize': 20, 'color': 'red'}),
        dcc.Store(id='network-message-receiver'),  # Store for receiving network messages
        dcc.Store(id='current-game-id', data=None)  # Store for current game ID
    ], style={'textAlign': 'center', 'marginTop': '20px'})

def game_layout():
    return html.Div([
        dcc.Graph(
            id='chessboard',
            config={'staticPlot': False, 'scrollZoom': False, 'editable': False, 'displayModeBar': False}
        ),
        html.Div(id='feedback-message', style={'textAlign': 'center', 'fontSize': 20, 'color': 'red'}),
        dcc.Input(id='move-input', type='text', style={'display': 'none'}),  # Hidden input for moves
        dcc.Store(id='selected-pos'),
        dcc.Store(id='start-square'),
        dcc.Store(id='selected-square'),
        html.Div([
            dcc.Input(id='llm-input', type='text', placeholder='Ask a question'),
            html.Button('Submit', id='submit-llm-input', n_clicks=0)
        ], style={'textAlign': 'center', 'marginTop': '20px'}),
        html.Button('Reset Game', id='reset-button', n_clicks_timestamp=0, style={'marginTop': '20px'})
    ])

# Define app layout
app.layout = html.Div([
    html.Div(id="layout", children=initial_layout()),
    dcc.Store(id='current-game-id', data=None)
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
            # Square trace
            fig.add_trace(go.Scatter(
                x=[file + 0.5], y=[rank + 0.5],
                customdata=[{"square_id": chess.square_name(square_index)}],  # custom identifier
                marker=dict(color=cell_color, size=square_size),
                mode='markers',
                marker_symbol='square',
                line=dict(width=0)
            ))

            piece = board.piece_at(square_index)
            if piece:
                color = 'white' if piece.color == chess.WHITE else 'black'
                symbol = chess_symbols[piece.piece_type][color]
                # Piece trace
                fig.add_trace(go.Scatter(
                    x=[file + 0.5], y=[rank + 0.5],
                    text=[symbol],
                    mode='text',
                    textfont=dict(color=color, size=piece_font_size),
                    textposition='middle center',
                    showlegend=False,
                    marker=dict(size=piece_size),
                    customdata=[{"square_id": chess.square_name(square_index), "piece": symbol}],  # custom identifier
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

def clean_model_output(output_str):
    """Clean up unnecessary annotations from model output."""
    cleaned_output = re.sub(r"\{[^}]*\}", "", output_str)  # Removes anything inside curly braces
    return cleaned_output

def get_llm_response(question, fen_position, max_retries=5):
    """Get a response from the language model regarding the current game state, retrying until a valid move is found."""
    for _ in range(max_retries):
        prompt = generate_prompt(fen_position, question)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50)
        output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_output = clean_model_output(output_str)
        
        # Extract and validate moves
        moves = extract_moves(cleaned_output)
        valid_move = filter_to_one_valid_move(moves, board)
        if valid_move:
            move = board.san(chess.Move.from_uci(valid_move))
            return f'Best move according to the model: {move}', True

    return "Model suggested invalid moves repeatedly. Please try again.", False

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

def handle_chessboard_click(click_data):
    # Handles logic for when the user clicks on the chessboard
    message, move_status = process_click_data(click_data)
    if move_status:
        return generate_board_figure(selected_square), message
    else:
        return generate_board_figure(selected_square), 'Invalid move! Try again.'

def process_click_data(click_data):
    global selected_square, start_square, board
    message = ""
    move_status = False

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
            move_status = True

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

    return message, move_status

@app.callback(
    Output('chessboard', 'figure', allow_duplicate=True),
    [Input('move-input', 'value')],
    [State('chessboard', 'figure')]
)
def handle_move(move, current_figure):
    global board
    if move:
        try:
            chess_move = board.parse_san(move)
            board.push(chess_move)
            if in_multiplayer_mode and current_game_id is not None:
                move_data = {'game_id': current_game_id, 'board': board.fen()}
                network.send_message(json.dumps({'type': 'move', 'data': move_data}))
            return generate_board_figure()
        except Exception as e:
            return generate_board_figure()  # Maintain previous state if move is invalid
    raise dash.exceptions.PreventUpdate

@app.callback(
    Output('chessboard', 'figure'),
    Output('feedback-message', 'children', allow_duplicate=True),
    [Input('chessboard', 'clickData'), Input('submit-llm-input', 'n_clicks')],
    [State('selected-pos', 'data'), State('start-square', 'data'),
     State('llm-input', 'value'), State('reset-button', 'n_clicks_timestamp')]
)
def update_chessboard_and_get_response(click_data, llm_n_clicks, selected_pos, start_pos, question, reset_clicks_timestamp):
    global selected_square, start_square, board, reset_timestamp
    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if reset_clicks_timestamp > reset_timestamp:
        reset_timestamp = reset_clicks_timestamp
        board.reset()
        selected_square = None
        start_square = None
        return generate_board_figure(), 'Game has been reset.'

    if click_data and input_id == 'chessboard':
        # Handling a click on the chessboard
        return handle_chessboard_click(click_data)

    elif llm_n_clicks > 0 and question and input_id == 'submit-llm-input':
        # Update the feedback message immediately to indicate thinking
        message = 'Thinking...'
        response, valid = get_llm_response(question, board.fen())
        if not valid:
            return generate_board_figure(selected_square), message
        return generate_board_figure(selected_square), response

    return generate_board_figure(selected_square), 'Click on the board or ask a question.'

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

@app.callback(
    [Output('current-game-id', 'data', allow_duplicate=True),
     Output('layout', 'children', allow_duplicate=True)],
    [Input('vs-ai-button', 'n_clicks'), 
     Input('multiplayer-button', 'n_clicks')],
    [State('current-game-id', 'data')]
)
def update_layout(vs_ai_clicks, multiplayer_clicks, current_game_id):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_game_id, initial_layout()

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'vs-ai-button' and vs_ai_clicks:
        return None, game_layout()
    elif button_id == 'multiplayer-button' and multiplayer_clicks:
        return None, multiplayer_layout()
    
    return current_game_id, initial_layout()

@app.callback(
    Output('multiplayer-game-table', 'data'),
    Output('feedback-message', 'children', allow_duplicate=True),
    [Input('refresh-multiplayer-button', 'n_clicks'),
     Input('new-multiplayer-game-button', 'n_clicks'),
     Input('multiplayer-game-table', 'active_cell'),
     Input('network-message-receiver', 'data')],
    [State('multiplayer-game-table', 'data')]
)
def update_multiplayer_table(refresh_clicks, new_game_clicks, active_cell, network_data, data):
    global in_multiplayer_mode, current_game_id, board
    ctx = dash.callback_context

    if not ctx.triggered:
        return data, ""

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'refresh-multiplayer-button':
        return data, 'Multiplayer game data refreshed.'

    elif triggered_id == 'new-multiplayer-game-button':
        new_game = {"Player 1": "Player", "Player 2": "-", "Status": "Waiting", "Last Move": "-", "Action": "[Play](button)", "game_id": len(data) + 1}
        data.append(new_game)
        network.send_message(json.dumps({'type': 'new_game', 'data': new_game}))
        current_game_id = len(data)
        in_multiplayer_mode = True
        board.reset()
        return data, ""

    elif triggered_id == 'multiplayer-game-table' and active_cell:
        row = data[active_cell['row']]
        column_id = active_cell['column_id']
        if column_id == 'Action' and row['Action'] == '[Play](button)':  # Confirming that the "Play" button was indeed clicked
            if row['Status'] == 'Waiting':
                network.send_message(json.dumps({'type': 'join_game', 'data': row}))
                current_game_id = row['game_id']
                in_multiplayer_mode = True
                board.reset()
                return data, "Attempting to join game..."
            elif row['Status'] == 'In Progress':
                current_game_id = row['game_id']
                in_multiplayer_mode = True
                return data, "Joining game..."

    elif triggered_id == 'network-message-receiver' and network_data:
        current_table_data = json.loads(network_data)
        updated_data = []
        for game_id, game_info in current_table_data.items():
            updated_data.append({
                "Player 1": game_info['players'][0],
                "Player 2": game_info['players'][1] if len(game_info['players']) > 1 else "-",
                "Status": game_info['status'],
                "Last Move": "-",  # Update this if you have last move data
                "Action": f'[Play](button)',
                "game_id": game_id
            })
        return updated_data, ""

    return data, ""

@app.callback(
    Output('current-game-id', 'data', allow_duplicate=True),
    Output('layout', 'children', allow_duplicate=True),
    [Input('multiplayer-game-table', 'active_cell')],
    [State('multiplayer-game-table', 'data'), State('current-game-id', 'data')]
)
def join_game_from_table(active_cell, data, current_game_id):
    if active_cell:
        row = data[active_cell['row']]
        if row['Status'] == 'Waiting':
            network.send_message(json.dumps({'type': 'join_game', 'data': row}))
            current_game_id = row['game_id']
            return current_game_id, game_layout()
        elif row['Status'] == 'In Progress':
            current_game_id = row['game_id']
            return current_game_id, game_layout()
    return current_game_id, multiplayer_layout()

if __name__ == '__main__':
    app.run_server(debug=True)
