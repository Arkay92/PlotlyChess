"""Application factory and server-authoritative interactive chess UI."""
import os
from pathlib import Path
from urllib.parse import parse_qs

import chess
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from . import game
from .ai import ChessAI
from .board import interactive_board, piece_image
from .multiplayer import MultiplayerHub, attach_websocket

DEFAULT_HUB = MultiplayerHub(shared=True)


def extract_code_from_url(search, pathname=None):
    if search:
        params = parse_qs(search.lstrip('?'))
        for key in ('code', 'game', 'join', 'room'):
            values = params.get(key)
            if values and values[0].strip():
                return values[0].strip().upper()
    if pathname and pathname.startswith('/join/'):
        code = pathname.split('/join/', 1)[1].strip('/').upper()
        if code:
            return code
    return None


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
            for i in range(0, len(moves), 2)]


def render_room_status(mp):
    mp = mp or {}
    code = mp.get('code')
    if not code:
        return []
    connected = bool(mp.get('connected'))
    color_name = 'White' if mp.get('color') == 'white' else 'Black'
    session_attrs = {
        'data-room-code': code,
        'data-player-token': mp.get('token', ''),
        'data-mp-version': str(mp.get('version', 0)),
        'data-mp-connected': str(connected).lower(),
    }
    if not connected:
        return html.Div([
            html.Span('SEND THIS CODE OR LINK TO YOUR FRIEND', className='field-label'),
            html.Div([
                html.Strong(code, id='mp-room-code', className='mp-code-badge',
                            style={'flex': '1', 'background': '#fff', 'border': '1px dashed #c7cec2',
                                   'borderRadius': '6px', 'padding': '9px 12px', 'fontSize': '16px',
                                   'letterSpacing': '3.2px', 'textAlign': 'center'}),
                html.Button('Copy code', id='mp-copy-btn', type='button',
                            className='button quiet mp-copy-btn', **{'data-code': code}),
                html.Button('Copy link', id='mp-copy-link-btn', type='button',
                            className='button secondary mp-copy-btn', **{'data-code': code}),
            ], className='mp-code-row', style={'display': 'flex', 'gap': '6px', 'alignItems': 'center', 'flexWrap': 'wrap'}),
            html.Div([
                html.Span(className='mp-dot waiting',
                          style={'width': '7px', 'height': '7px', 'borderRadius': '50%', 'background': '#c56a49', 'display': 'inline-block'}),
                html.Span('Waiting for friend to join…', className='mp-status-text'),
            ], className='mp-status-row', style={'display': 'flex', 'alignItems': 'center', 'gap': '7px', 'marginTop': '11px', 'fontSize': '11px'}),
        ], className='mp-session', **session_attrs)

    return html.Div([
        html.Div([
            html.Span(f'ROOM {code}', id='mp-room-code', className='mp-code-pill',
                      style={'fontSize': '10px', 'fontWeight': '650', 'padding': '5px 8px', 'borderRadius': '5px', 'background': '#fff', 'border': '1px solid #e7e8e2'}),
            html.Button('Copy link', id='mp-copy-link-btn', type='button',
                        className='button quiet', style={'padding': '4px 8px', 'fontSize': '10px'}, **{'data-code': code}),
            html.Span(f'Playing as {color_name}', className='mp-role-pill',
                      style={'fontSize': '10px', 'fontWeight': '650', 'padding': '5px 8px', 'borderRadius': '5px', 'background': '#e8eee2', 'color': '#52674a', 'marginLeft': 'auto'}),
        ], className='mp-connected-head', style={'display': 'flex', 'gap': '6px', 'alignItems': 'center'}),
        html.Div([
            html.Span(className='mp-dot connected',
                      style={'width': '7px', 'height': '7px', 'borderRadius': '50%', 'background': '#6b9362', 'display': 'inline-block'}),
            html.Span('Connected · Take turns on the board', className='mp-status-text'),
        ], className='mp-status-row', style={'display': 'flex', 'alignItems': 'center', 'gap': '7px', 'marginTop': '11px', 'fontSize': '11px'}),
    ], className='mp-session', **session_attrs)


def multiplayer_styles(mode, mp):
    if mode != 'multiplayer':
        return (
            {'display': 'none'},
            {'display': 'none'},
            {'display': 'none'},
            {'display': 'none'},
            [],
            {'display': 'none'},
            True,
        )
    mp = mp or {'view': 'choice'}
    code = mp.get('code')
    panel_style = {'display': 'block', 'marginTop': '16px'}
    box_base = {'background': '#f9f8f4', 'border': '1px solid #e7e8e2', 'borderRadius': '8px', 'padding': '14px'}
    if code:
        return (
            panel_style,
            {'display': 'none'},
            {'display': 'none'},
            {**box_base, 'display': 'block'},
            render_room_status(mp),
            {'display': 'flex', 'marginTop': '8px', 'justifyContent': 'flex-end'},
            False,
        )
    if mp.get('view') == 'join':
        return (
            panel_style,
            {'display': 'none'},
            {**box_base, 'display': 'block'},
            {'display': 'none'},
            [],
            {'display': 'none'},
            True,
        )
    return (
        panel_style,
        {**box_base, 'display': 'block'},
        {'display': 'none'},
        {'display': 'none'},
        [],
        {'display': 'none'},
        True,
    )


def create_app(app=None, hub=None):
    ai = ChessAI(os.environ.get('PLOTLY_CHESS_MODEL'))
    hub = hub or DEFAULT_HUB
    if app is None:
        app = Dash(__name__, assets_folder=str(Path(__file__).resolve().parent.parent/'assets'))
    attach_websocket(app.server, hub)
    app.title = 'Plotly Chess · A little room to think'
    board = chess.Board()
    app.layout = html.Main([
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='game', storage_type='session', data=game.new_game()),
        dcc.Store(id='board-action'),
        dcc.Interval(id='mp-poll', interval=1000, n_intervals=0, disabled=True),
        html.Header([
            html.A([html.Img(src=piece_image('N'), alt='', className='brand-mark'),
                    html.Span(['plotly', html.Strong('chess')])], href='/', className='brand'),
            html.Span('A little room to think.', className='brand-note'),
            html.Span([html.Span(className='online-dot'), 'Ready to play'], className='ready-label'),
        ], className='site-header'),
        html.Div([html.Div([html.P('THE BOARD IS YOURS', className='eyebrow'),
                            html.H1('Make your next move.')]),
                  html.P('A classic game. A fresh perspective.', className='intro-note')], className='page-intro'),
        html.Div([
            html.Section([
                html.Div(player_bar('Computer', 'Your opponent · Black', 'k'), id='opponent'),
                html.Div(interactive_board(board), id='board', className='board-shell',
                         **{'data-busy': 'false', 'aria-busy': 'false'}),
                html.Div(player_bar('You', 'Playing with the white pieces', 'K', True), id='player'),
                html.Div([html.Span('↗', className='instruction-icon'),
                          html.Span('Drag a piece, or click a piece and its destination.'),
                          html.Span('Tab + Enter works too.', className='keyboard-note')], className='board-instructions'),
            ], className='play-area', **{'aria-label': 'Play chess'}),
            html.Aside([
                html.Section([
                    html.Div([html.H2('Your game'), html.Span('STANDARD', className='small-badge')], className='card-heading'),
                    html.Label('PLAY MODE', className='field-label'),
                    dcc.RadioItems(id='mode', options=[
                        {'label': 'Vs computer', 'value': 'ai'},
                        {'label': 'Two players', 'value': 'local'},
                        {'label': 'Multiplayer', 'value': 'multiplayer'},
                    ], value='ai', className='mode-switch', persistence=True, persistence_type='session'),
                    html.Div([
                        html.Div([
                            html.Span('MULTIPLAYER LOBBY', className='field-label'),
                            html.P('Create a game to get a shareable code, or join a friend’s game.',
                                   className='mp-note', style={'margin': '0 0 11px', 'fontSize': '11px'}),
                            html.Div([
                                html.Button('Create game', id='mp-create-btn', n_clicks=0, type='button',
                                            className='button primary mp-choice-btn', style={'flex': '1'}),
                                html.Button('Join game', id='mp-choose-join-btn', n_clicks=0, type='button',
                                            className='button secondary mp-choice-btn', style={'flex': '1'}),
                            ], className='mp-choice-row', style={'display': 'flex', 'gap': '8px'}),
                        ], id='mp-view-choice', className='mp-box', style={'display': 'none'}),
                        html.Div([
                            html.Span('JOIN A FRIEND’S GAME', className='field-label'),
                            html.P('Enter the 6-character code your friend shared with you.',
                                   className='mp-note', style={'margin': '0 0 11px', 'fontSize': '11px'}),
                            html.Div([
                                dcc.Input(id='join-code-input', type='text', value='', placeholder='e.g. K7M4P9',
                                          maxLength=6, autoComplete='off', n_submit=0, className='mp-code-input',
                                          style={'flex': '1', 'minWidth': '0', 'padding': '9px 10px', 'border': '1px solid #e7e8e2',
                                                 'borderRadius': '6px', 'fontSize': '12px', 'fontWeight': '650',
                                                 'letterSpacing': '2px', 'textTransform': 'uppercase'}),
                                html.Button('Join', id='mp-join-submit', n_clicks=0, type='button', className='button primary'),
                            ], className='mp-join-row', style={'display': 'flex', 'gap': '8px'}),
                            html.Div([
                                html.Button('← Back', id='mp-back-choice', n_clicks=0, type='button',
                                            className='button quiet mp-sub-btn', style={'padding': '7px 10px', 'fontSize': '10px'}),
                            ], className='mp-sub-actions', style={'display': 'flex', 'gap': '8px', 'marginTop': '9px'}),
                        ], id='mp-view-join', className='mp-box', style={'display': 'none'}),
                        html.Div([], id='mp-view-room', className='mp-box', style={'display': 'none'}),
                        html.Div([
                            html.Button('Leave game', id='mp-leave-btn', n_clicks=0, type='button',
                                        className='button quiet mp-sub-btn', style={'padding': '6px 10px', 'fontSize': '10px'}),
                        ], id='mp-leave-row', style={'display': 'none'}),
                    ], id='multiplayer-panel', className='mp-panel', style={'display': 'none'}),
                    html.Div([html.Span(className='turn-dot'), html.Strong('White to move.', id='status')], className='turn-status'),
                    html.Div('Choose your opening. Take your time.', id='feedback', className='feedback', role='status', **{'aria-live': 'polite'}),
                    html.Div([html.Button('✦  Get a hint', id='hint', className='button secondary'),
                              html.Button('↺  New game', id='reset', className='button quiet')], className='game-actions'),
                ], className='panel game-panel'),
                html.Section([
                    html.Div([html.H2('Move history'), html.Span('0 moves', id='move-count', className='muted')], className='card-heading'),
                    html.Div([html.Span('#'), html.Span('WHITE'), html.Span('BLACK')], className='history-labels'),
                    html.Div(move_history(board), id='history', className='history-list', **{'aria-label': 'Move history'}),
                ], className='panel history-panel'),
                html.Details([
                    html.Summary('Move options'),
                    html.Label('ENTER A MOVE', htmlFor='notation', className='field-label'),
                    html.Div([dcc.Input(id='notation', placeholder='e.g. Nf3 or e2e4', debounce=True),
                              html.Button('Play', id='play', className='button primary')], className='notation-row'),
                    html.Label('PROMOTE PAWNS TO', htmlFor='promotion', className='field-label'),
                    dcc.Dropdown(id='promotion', options=[{'label': n, 'value': v} for n, v in
                        [('Queen', 'q'), ('Rook', 'r'), ('Bishop', 'b'), ('Knight', 'n')]], value='q', clearable=False, searchable=False),
                ], className='panel options-panel'),
                html.Div([html.Span(className='engine-dot'),
                          html.Span('NPC expert routing' if ai.router else 'Material search', id='engine-label')], className='engine-label'),
            ], className='game-sidebar'),
        ], className='game-layout'),
        html.Footer([html.Span('Less distraction. More chess.'), html.Span('Built with Python · Made for your next move.')]),
    ], className='app-container')

    @app.callback(
        Output('game', 'data'), Output('board', 'children'), Output('status', 'children'), Output('history', 'children'),
        Output('feedback', 'children'), Output('opponent', 'children'), Output('player', 'children'), Output('move-count', 'children'),
        Output('multiplayer-panel', 'style'), Output('mp-view-choice', 'style'), Output('mp-view-join', 'style'),
        Output('mp-view-room', 'style'), Output('mp-view-room', 'children'), Output('mp-leave-row', 'style'),
        Output('mp-poll', 'disabled'), Output('mode', 'value'), Output('url', 'search'),
        Input('board-action', 'data'), Input('play', 'n_clicks'), Input('reset', 'n_clicks'),
        Input('hint', 'n_clicks'), Input('mode', 'value'),
        Input('mp-create-btn', 'n_clicks'), Input('mp-choose-join-btn', 'n_clicks'),
        Input('mp-join-submit', 'n_clicks'), Input('join-code-input', 'n_submit'),
        Input('mp-back-choice', 'n_clicks'), Input('mp-leave-btn', 'n_clicks'),
        Input('mp-poll', 'n_intervals'), Input('url', 'search'), Input('url', 'pathname'),
        State('notation', 'value'), State('promotion', 'value'), State('game', 'data'), State('join-code-input', 'value'))
    def update(action, play_count, reset_count, hint_count, mode,
               create_clicks, choose_join_clicks, join_clicks, join_submit, back_clicks, leave_clicks,
               poll_ticks, url_search, url_pathname,
               notation, promotion, state, join_code):
        state = state or game.new_game()
        mp = dict(state.get('mp') or {'view': 'choice'})
        message = 'Drag to move. Click a piece to see its legal moves.'
        trigger = ctx.triggered_id
        url_code = extract_code_from_url(url_search, url_pathname)

        if trigger == 'mp-poll':
            if mode != 'multiplayer' or not mp.get('code'):
                return (no_update,) * 17
            live = hub.get_room(mp['code'], mp.get('token'))
            if not live or (live['version'] == mp.get('version') and live['connected'] == mp.get('connected')):
                return (no_update,) * 17
            mp = {'view': 'room', **live}
            state = {'moves': live['moves'], 'selected': state.get('selected'), 'mp': mp}
            message = live['message']

        elif mode == 'multiplayer' and mp.get('code') and trigger not in ('mp-create-btn', 'mp-leave-btn', 'mode'):
            live = hub.get_room(mp['code'], mp.get('token'))
            if live:
                if live['moves'] != state.get('moves'):
                    state = {**state, 'moves': live['moves'], 'selected': None}
                mp = {'view': 'room', **live}
                state = {**state, 'mp': mp}

        try:
            if trigger in (None, 'url') and url_code:
                mode = 'multiplayer'
                existing_token = mp.get('token') if mp.get('code') == url_code else None
                snap = hub.join_room(url_code, existing_token)
                mp = {'view': 'room', **snap}
                state = {'moves': snap['moves'], 'selected': None, 'mp': mp}
                message = snap['message']
            elif trigger == 'mode':
                if mode == 'multiplayer':
                    mp = {'view': 'choice'}
                    state = {**game.new_game(), 'mp': mp}
                    message = 'Create a game to get a code for your friend, or join with their code.'
                else:
                    state = game.new_game()
                    message = 'A fresh board. Your move.'
            elif trigger == 'mp-create-btn':
                mode = 'multiplayer'
                snap = hub.create_room()
                mp = {'view': 'room', **snap}
                state = {'moves': snap['moves'], 'selected': None, 'mp': mp}
                message = snap['message']
            elif trigger == 'mp-choose-join-btn':
                mode = 'multiplayer'
                mp = {'view': 'join'}
                state = {'moves': [], 'selected': None, 'mp': mp}
                message = 'Enter the 6-character code from your friend.'
            elif trigger == 'mp-back-choice':
                mp = {'view': 'choice'}
                state = {'moves': [], 'selected': None, 'mp': mp}
                message = 'Create a game or join with a friend’s code.'
            elif trigger in ('mp-join-submit', 'join-code-input'):
                mode = 'multiplayer'
                snap = hub.join_room(join_code, mp.get('token'))
                mp = {'view': 'room', **snap}
                state = {'moves': snap['moves'], 'selected': None, 'mp': mp}
                message = snap['message']
            elif trigger == 'mp-leave-btn':
                mp = {'view': 'choice'}
                state = {'moves': [], 'selected': None, 'mp': mp}
                message = 'Left multiplayer room. Create or join a game.'
            elif trigger == 'reset':
                if mode == 'multiplayer' and mp.get('code'):
                    snap = hub.reset_room(mp['code'], mp.get('token'))
                    mp = {'view': 'room', **snap}
                    state = {'moves': snap['moves'], 'selected': None, 'mp': mp}
                    message = snap['message']
                else:
                    state = {**game.new_game(), **({'mp': mp} if mode == 'multiplayer' else {})}
                    message = 'A fresh board. Your move.'
            elif trigger == 'hint':
                position = game.restore(state)
                move, _ = ai.choose(position)
                message = f'Try {position.san(move)}.'
            elif trigger == 'play' and notation:
                if mode == 'multiplayer':
                    if not mp.get('code'):
                        raise ValueError('Create or join a multiplayer game first.')
                    snap, san = hub.play_move(mp['code'], mp.get('token'), notation.strip())
                    mp = {'view': 'room', **snap}
                    state = {'moves': snap['moves'], 'selected': None, 'mp': mp}
                    message = f'Played {san}.'
                else:
                    state, san = game.play(state, notation.strip())
                    message = f'Played {san}.'
            elif trigger == 'board-action' and action:
                kind = action.get('type')
                if kind == 'mp-create':
                    snap = hub.create_room()
                    mp = {'view': 'room', **snap}
                    state = {'moves': snap['moves'], 'selected': None, 'mp': mp}
                    message = snap['message']
                elif kind == 'mp-choose-join':
                    mp = {'view': 'join'}
                    state = {'moves': [], 'selected': None, 'mp': mp}
                    message = 'Enter the 6-character code from your friend.'
                elif kind == 'mp-back':
                    mp = {'view': 'choice'}
                    state = {'moves': [], 'selected': None, 'mp': mp}
                    message = 'Create a game or join with a friend’s code.'
                elif kind == 'mp-join':
                    snap = hub.join_room(action.get('code', ''), mp.get('token'))
                    mp = {'view': 'room', **snap}
                    state = {'moves': snap['moves'], 'selected': None, 'mp': mp}
                    message = snap['message']
                elif kind == 'mp-leave':
                    mp = {'view': 'choice'}
                    state = {'moves': [], 'selected': None, 'mp': mp}
                    message = 'Left multiplayer room. Create or join a game.'
                elif kind == 'mp-sync':
                    if mp.get('code'):
                        snap = hub.get_room(mp['code'], mp.get('token'))
                        if snap:
                            mp = {'view': 'room', **snap}
                            state = {'moves': snap['moves'], 'selected': state.get('selected'), 'mp': mp}
                            message = snap['message']
                elif kind == 'click':
                    if mode == 'multiplayer':
                        if not mp.get('code'):
                            raise ValueError('Create or join a multiplayer game first.')
                        if not mp.get('connected'):
                            raise ValueError(f'Waiting for your friend to join with code {mp["code"]}.')
                        position = game.restore(state)
                        my_turn = (mp.get('color') == 'white') == (position.turn == chess.WHITE)
                        if not my_turn:
                            raise ValueError(f'Wait for {"White" if position.turn == chess.WHITE else "Black"} to move.')
                        updated, click_msg, moved = game.click(state, int(action['square']), promotion)
                        if moved:
                            snap, san = hub.play_move(mp['code'], mp.get('token'), updated['moves'][-1])
                            mp = {'view': 'room', **snap}
                            state = {'moves': snap['moves'], 'selected': None, 'mp': mp}
                            message = f'Played {san}.'
                        else:
                            state = {**updated, 'mp': mp}
                            message = click_msg
                    else:
                        state, message, _ = game.click(state, int(action['square']), promotion)
                elif kind == 'drop':
                    source, target = int(action['from']), int(action['to'])
                    if source not in chess.SQUARES or target not in chess.SQUARES:
                        raise ValueError('Drop the piece on a board square.')
                    position = game.restore(state)
                    if mode == 'multiplayer':
                        if not mp.get('code'):
                            raise ValueError('Create or join a multiplayer game first.')
                        if not mp.get('connected'):
                            raise ValueError(f'Waiting for your friend to join with code {mp["code"]}.')
                        my_turn = (mp.get('color') == 'white') == (position.turn == chess.WHITE)
                        if not my_turn:
                            raise ValueError(f'Wait for {"White" if position.turn == chess.WHITE else "Black"} to move.')
                    piece = position.piece_at(source)
                    if not piece or piece.color != position.turn:
                        raise ValueError('Choose a piece belonging to the player to move.')
                    suffix = promotion if piece.piece_type == chess.PAWN and chess.square_rank(target) in (0, 7) else ''
                    uci = chess.square_name(source) + chess.square_name(target) + suffix
                    if mode == 'multiplayer':
                        snap, san = hub.play_move(mp['code'], mp.get('token'), uci)
                        mp = {'view': 'room', **snap}
                        state = {'moves': snap['moves'], 'selected': None, 'mp': mp}
                        message = f'Played {san}.'
                    else:
                        state, san = game.play(state, uci)
                        message = f'Played {san}.'
            position = game.restore(state)
            if mode == 'ai' and position.turn == chess.BLACK and not position.is_game_over(claim_draw=True):
                move, _ = ai.choose(position)
                state, san = game.play(state, move.uci())
                message += f' Computer played {san}.'
        except (ValueError, KeyError, TypeError) as exc:
            message = str(exc) if mode == 'multiplayer' and str(exc) else 'That move is not available. Try a highlighted square.'
            state = {**state, 'selected': None, **({'mp': mp} if mode == 'multiplayer' else {})}

        position = game.restore(state)
        ended = position.is_game_over(claim_draw=True)
        allowed_color = None
        if mode == 'multiplayer':
            mp = state.get('mp') or mp
            state = {**state, 'mp': mp}
            if not mp.get('code') or not mp.get('connected'):
                allowed_color = False
            else:
                allowed_color = chess.WHITE if mp.get('color') == 'white' else chess.BLACK
            code = mp.get('code')
            connected = bool(mp.get('connected'))
            if mp.get('color') == 'black':
                opponent_bar = player_bar('Friend (White)', f'Connected · Room {code}', 'K',
                                          position.turn == chess.WHITE and not ended and connected)
                self_bar = player_bar('You (Black)', f'Playing with the black pieces · Room {code}', 'k',
                                      position.turn == chess.BLACK and not ended and connected)
            else:
                opp_name = 'Friend (Black)' if connected else 'Waiting for friend…'
                opp_sub = f'Connected · Room {code}' if connected else (f'Share code {code} to connect' if code else 'Create or join a game')
                opponent_bar = player_bar(opp_name, opp_sub, 'k',
                                          position.turn == chess.BLACK and not ended and connected)
                self_bar = player_bar('You (White)' if code else 'You',
                                      f'Playing with the white pieces · Room {code}' if code else 'Multiplayer lobby', 'K',
                                      position.turn == chess.WHITE and not ended and connected)
        else:
            opponent_bar = player_bar('Computer' if mode == 'ai' else 'Black',
                                      'Your opponent · Black' if mode == 'ai' else 'Local player · Black',
                                      'k', position.turn == chess.BLACK and not ended)
            self_bar = player_bar('You' if mode == 'ai' else 'White',
                                  'Playing with the white pieces' if mode == 'ai' else 'Local player · White',
                                  'K', position.turn == chess.WHITE and not ended)

        panel_style, choice_style, join_style, room_style, room_children, leave_style, poll_disabled = multiplayer_styles(mode, state.get('mp'))
        active_code = (state.get('mp') or {}).get('code') if mode == 'multiplayer' else None
        target_search = f'?code={active_code}' if active_code else ''
        search_out = target_search if (url_search or '') != target_search else no_update

        return (
            state,
            interactive_board(position, state.get('selected'), allowed_color),
            game.status(position),
            move_history(position),
            message,
            opponent_bar,
            self_bar,
            f'{len(state["moves"]):,} {"move" if len(state["moves"]) == 1 else "moves"}',
            panel_style,
            choice_style,
            join_style,
            room_style,
            room_children,
            leave_style,
            poll_disabled,
            mode,
            search_out,
        )
    return app
