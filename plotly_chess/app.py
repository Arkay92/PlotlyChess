
"""Application factory and server-authoritative interactive chess UI."""
import os
from pathlib import Path

import chess
from dash import Dash, Input, Output, State, ctx, dcc, html

from . import game
from .ai import ChessAI
from .board import interactive_board, piece_image


def player_bar(name, subtitle, symbol, active=False):
    return html.Div([
        html.Div(html.Img(src=piece_image(symbol), alt=''), className='player-avatar'),
        html.Div([html.Strong(name), html.Small(subtitle)], className='player-name'),
        html.Span('To move' if active else 'Black' if symbol.islower() else 'White',
                  className='player-tag active' if active else 'player-tag'),
    ], className='player-bar')


def move_history(board):
    replay, moves = chess.Board(), []
    for move in board.move_stack:
        moves.append(replay.san(move))
        replay.push(move)
    if not moves:
        return html.Div([html.Span('Your next idea starts here.', className='empty-title'),
                         html.P('Make a move to begin your game.')], className='history-empty')
    return [html.Div([html.Span(f'{i//2+1}.', className='move-number'),
                     html.Span(moves[i]), html.Span(moves[i+1] if i+1 < len(moves) else '…')],
                     className='move-row latest' if i+2 >= len(moves) else 'move-row')
            for i in range(0,len(moves),2)]


def create_app(app=None):
    ai = ChessAI(os.environ.get('PLOTLY_CHESS_MODEL'))
    if app is None:
        app = Dash(__name__, assets_folder=str(Path(__file__).resolve().parent.parent/'assets'))
    app.title = 'Plotly Chess · A little room to think'
    board = chess.Board()
    app.layout = html.Main([
        dcc.Store(id='game', storage_type='session', data=game.new_game()),
        dcc.Store(id='board-action'),
        html.Header([
            html.A([html.Img(src=piece_image('N'), alt='', className='brand-mark'),
                    html.Span(['plotly', html.Strong('chess')])], href='/', className='brand'),
            html.Span('A little room to think.', className='brand-note'),
            html.Span([html.Span(className='online-dot'), 'Ready to play'], className='ready-label'),
        ], className='site-header'),
        html.Div([html.Div([html.P('THE BOARD IS YOURS',className='eyebrow'),
                            html.H1('Make your next move.')]),
                  html.P('A classic game. A fresh perspective.',className='intro-note')],className='page-intro'),
        html.Div([
            html.Section([
                html.Div(player_bar('Computer', 'Your opponent · Black', 'k'), id='opponent'),
                html.Div(interactive_board(board), id='board', className='board-shell',
                         **{'data-busy':'false','aria-busy':'false'}),
                html.Div(player_bar('You', 'Playing with the white pieces', 'K', True), id='player'),
                html.Div([html.Span('↗',className='instruction-icon'),
                          html.Span('Drag a piece, or click a piece and its destination.'),
                          html.Span('Tab + Enter works too.',className='keyboard-note')],className='board-instructions'),
            ],className='play-area', **{'aria-label':'Play chess'}),
            html.Aside([
                html.Section([
                    html.Div([html.H2('Your game'),html.Span('STANDARD',className='small-badge')],className='card-heading'),
                    html.Label('PLAY MODE',className='field-label'),
                    dcc.RadioItems(id='mode',options=[{'label':'Vs computer','value':'ai'},
                        {'label':'Two players','value':'local'}],value='ai',className='mode-switch',
                        persistence=True,persistence_type='session'),
                    html.Div([html.Span(className='turn-dot'),html.Strong('White to move.',id='status')],className='turn-status'),
                    html.Div('Choose your opening. Take your time.',id='feedback',className='feedback',role='status',**{'aria-live':'polite'}),
                    html.Div([html.Button('✦  Get a hint',id='hint',className='button secondary'),
                              html.Button('↺  New game',id='reset',className='button quiet')],className='game-actions'),
                ],className='panel game-panel'),
                html.Section([
                    html.Div([html.H2('Move history'),html.Span('0 moves',id='move-count',className='muted')],className='card-heading'),
                    html.Div([html.Span('#'),html.Span('WHITE'),html.Span('BLACK')],className='history-labels'),
                    html.Div(move_history(board),id='history',className='history-list',**{'aria-label':'Move history'}),
                ],className='panel history-panel'),
                html.Details([
                    html.Summary('Move options'),
                    html.Label('ENTER A MOVE',htmlFor='notation',className='field-label'),
                    html.Div([dcc.Input(id='notation',placeholder='e.g. Nf3 or e2e4',debounce=True),
                              html.Button('Play',id='play',className='button primary')],className='notation-row'),
                    html.Label('PROMOTE PAWNS TO',htmlFor='promotion',className='field-label'),
                    dcc.Dropdown(id='promotion', options=[{'label':n,'value':v} for n,v in
                        [('Queen','q'),('Rook','r'),('Bishop','b'),('Knight','n')]],value='q',clearable=False,searchable=False),
                ],className='panel options-panel'),
                html.Div([html.Span(className='engine-dot'),
                          html.Span('NPC expert routing' if ai.router else 'Material search',id='engine-label')],className='engine-label'),
            ],className='game-sidebar'),
        ],className='game-layout'),
        html.Footer([html.Span('Less distraction. More chess.'),html.Span('Built with Python · Made for your next move.')]),
    ],className='app-container')

    @app.callback(
        Output('game','data'),Output('board','children'),Output('status','children'),Output('history','children'),
        Output('feedback','children'),Output('opponent','children'),Output('player','children'),Output('move-count','children'),
        Input('board-action','data'),Input('play','n_clicks'),Input('reset','n_clicks'),
        Input('hint','n_clicks'),Input('mode','value'),State('notation','value'),
        State('promotion','value'),State('game','data'),
        running=[(Output('board','data-busy'),'true','false'),
                 (Output('hint','disabled'),True,False),(Output('play','disabled'),True,False)])
    def update(action, play_count, reset_count, hint_count, mode, notation, promotion, state):
        state = state or game.new_game()
        message = 'Drag to move. Click a piece to see its legal moves.'
        trigger = ctx.triggered_id
        try:
            if trigger in ('reset','mode'):
                state = game.new_game()
                message = 'A fresh board. Your move.'
            elif trigger == 'hint':
                position = game.restore(state)
                move, _ = ai.choose(position)
                message = f'Try {position.san(move)}.'
            elif trigger == 'play' and notation:
                state, san = game.play(state,notation.strip())
                message = f'Played {san}.'
            elif trigger == 'board-action' and action:
                if action.get('type') == 'click':
                    state,message,_ = game.click(state,int(action['square']),promotion)
                elif action.get('type') == 'drop':
                    source, target = int(action['from']), int(action['to'])
                    if source not in chess.SQUARES or target not in chess.SQUARES:
                        raise ValueError('Drop the piece on a board square.')
                    position = game.restore(state)
                    piece = position.piece_at(source)
                    if not piece or piece.color != position.turn:
                        raise ValueError('Choose a piece belonging to the player to move.')
                    suffix = promotion if piece.piece_type == chess.PAWN and chess.square_rank(target) in (0,7) else ''
                    state,san = game.play(state,chess.square_name(source)+chess.square_name(target)+suffix)
                    message = f'Played {san}.'
            position = game.restore(state)
            if mode == 'ai' and position.turn == chess.BLACK and not position.is_game_over(claim_draw=True):
                move, _ = ai.choose(position)
                state,san = game.play(state,move.uci())
                message += f' Computer played {san}.'
        except (ValueError,KeyError,TypeError):
            message = 'That move is not available. Try a highlighted square.'
            state = {**state,'selected':None}
        position = game.restore(state)
        ended = position.is_game_over(claim_draw=True)
        return (state, interactive_board(position,state.get('selected')),game.status(position),move_history(position),message,
                player_bar('Computer' if mode == 'ai' else 'Black', 'Your opponent · Black' if mode == 'ai' else 'Local player · Black',
                           'k',position.turn == chess.BLACK and not ended),
                player_bar('You' if mode == 'ai' else 'White','Playing with the white pieces' if mode == 'ai' else 'Local player · White',
                           'K',position.turn == chess.WHITE and not ended),
                f'{len(state["moves"]):,} {"move" if len(state["moves"]) == 1 else "moves"}')
    return app
