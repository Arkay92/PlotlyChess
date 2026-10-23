"""Pure transitions; replay preserves repetition and castling history."""
import chess

def new_game():
    return {"moves": [], "selected": None}

def restore(state):
    board = chess.Board()
    for uci in state["moves"]:
        board.push_uci(uci)
    return board

def status(board):
    outcome = board.outcome(claim_draw=True)
    if outcome:
        return f"Game over: {outcome.result()} ({outcome.termination.name.lower()})."
    return f"{'White' if board.turn else 'Black'} to move" + (" — check!" if board.is_check() else ".")

def play(state, notation):
    board = restore(state)
    if board.is_game_over(claim_draw=True):
        raise ValueError("Game over. Start a new game.")
    move = board.parse_san(notation)
    san = board.san(move)
    return {"moves": state["moves"] + [move.uci()], "selected": None}, san

def click(state, square, promotion="q"):
    board = restore(state)
    if board.is_game_over(claim_draw=True):
        return state, status(board), False
    piece = board.piece_at(square)
    if piece and piece.color == board.turn:
        return {**state, "selected": square}, f"Selected {chess.square_name(square)}.", False
    selected = state.get("selected")
    if selected is None:
        return state, "Select a piece belonging to the player to move.", False
    piece = board.piece_at(selected)
    promote = piece and piece.piece_type == chess.PAWN and chess.square_rank(square) in (0, 7)
    move = chess.Move(selected, square, promotion=chess.PIECE_SYMBOLS.index(promotion) if promote else None)
    try:
        updated, san = play(state, move.uci())
        return updated, san, True
    except ValueError:
        return {**state, "selected": None}, "Illegal move. Select a piece and try again.", False
