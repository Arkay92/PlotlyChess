"""Application factory and server-authoritative interactive chess UI."""
import os
from pathlib import Path
import secrets
import time
from urllib.parse import parse_qs

import chess
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from . import game
from .ai import ChessAI
from .auth import AuthStore
from .board import interactive_board, piece_image
from .multiplayer import MultiplayerHub, attach_websocket

DEFAULT_HUB = MultiplayerHub(shared=True)
DEFAULT_AUTH = AuthStore(shared=True)

AVATAR_OPTIONS = [
    {'label': '♔ White King', 'value': 'K'},
    {'label': '♕ White Queen', 'value': 'Q'},
    {'label': '♖ White Rook', 'value': 'R'},
    {'label': '♗ White Bishop', 'value': 'B'},
    {'label': '♘ White Knight', 'value': 'N'},
    {'label': '♙ White Pawn', 'value': 'P'},
    {'label': '♚ Black King', 'value': 'k'},
    {'label': '♛ Black Queen', 'value': 'q'},
    {'label': '♜ Black Rook', 'value': 'r'},
    {'label': '♝ Black Bishop', 'value': 'b'},
    {'label': '♞ Black Knight', 'value': 'n'},
    {'label': '♟ Black Pawn', 'value': 'p'},
]


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


def _current_game_id(mode, state):
    if mode == 'multiplayer':
        code = ((state or {}).get('mp') or {}).get('code')
        return f'mp:{code}' if code else None
    local_id = (state or {}).get('local_id')
    if local_id and (state or {}).get('moves'):
        return f'{mode}:{local_id}'
    return None


def _upsert_saved_game(saved_games, entry):
    items = [g for g in (saved_games or []) if isinstance(g, dict) and g.get('id') != entry.get('id')]
    return [entry, *items][:12]


def _sync_saved_games(saved_games, hub):
    synced = []
    seen = set()
    for item in saved_games or []:
        if not isinstance(item, dict) or not item.get('id') or item['id'] in seen:
            continue
        seen.add(item['id'])
        if item.get('mode') == 'multiplayer' and item.get('code'):
            live = hub.get_room(item['code'], item.get('token'))
            if live:
                item = {
                    **item,
                    'moves': live['moves'],
                    'connected': live['connected'],
                    'color': live.get('color') or item.get('color', 'white'),
                }
            else:
                continue
        synced.append(item)
    return synced


def _SnapshotCurrentIntoSaved(mode, state, saved_games, hub):
    saved_games = list(saved_games or [])
    if mode == 'multiplayer':
        mp = (state or {}).get('mp') or {}
        code = mp.get('code')
        if code:
            entry = {
                'id': f'mp:{code}',
                'mode': 'multiplayer',
                'code': code,
                'token': mp.get('token'),
                'color': mp.get('color') or 'white',
                'connected': bool(mp.get('connected')),
                'moves': list((state or {}).get('moves') or []),
            }
            saved_games = _upsert_saved_game(saved_games, entry)
    else:
        moves = list((state or {}).get('moves') or [])
        if moves:
            local_id = (state or {}).get('local_id') or secrets.token_hex(2).upper()
            state['local_id'] = local_id
            entry = {
                'id': f'{mode}:{local_id}',
                'local_id': local_id,
                'mode': mode,
                'moves': moves,
            }
            saved_games = _upsert_saved_game(saved_games, entry)
    return _sync_saved_games(saved_games, hub)


def render_active_games(saved_games, active_id):
    if not saved_games:
        return html.Div([
            html.Span('No active games saved yet.', className='empty-title'),
            html.P('Start or join a game and you can switch between multiple games here.'),
        ], className='history-empty', style={'minHeight': '96px'})

    rows = []
    for item in saved_games:
        gid = item['id']
        is_current = gid == active_id
        moves = item.get('moves') or []
        board = game.restore({'moves': moves, 'selected': None})
        ended = board.is_game_over(claim_draw=True)
        move_count = f"{len(moves)} {'move' if len(moves) == 1 else 'moves'}"

        if item.get('mode') == 'multiplayer':
            color_str = 'White' if item.get('color') == 'white' else 'Black'
            title = f"Room {item['code']} · {color_str}"
            if ended:
                sub = f"Completed · {move_count}"
            elif not item.get('connected'):
                sub = f"Waiting for friend · {move_count}"
            else:
                my_turn = (item.get('color') == 'white') == (board.turn == chess.WHITE)
                sub = f"{'Your turn' if my_turn else 'Opponent turn'} · {move_count}"
        else:
            mode_label = 'Vs computer' if item.get('mode') == 'ai' else 'Two players'
            title = f"{mode_label} · #{item.get('local_id', gid[-4:])}"
            sub = f"{'Completed' if ended else ('White to move' if board.turn else 'Black to move')} · {move_count}"

        action_el = (
            html.Span('Active', style={
                'fontSize': '10px', 'fontWeight': '650', 'padding': '5px 9px',
                'borderRadius': '5px', 'background': '#e8eee2', 'color': '#52674a',
            })
            if is_current else
            html.Button('Resume →', type='button', className='button secondary',
                        style={'padding': '5px 10px', 'fontSize': '10px', 'flex': '0'},
                        **{'data-resume-game': gid})
        )
        rows.append(html.Div([
            html.Div([
                html.Strong(title, style={'fontSize': '12px', 'display': 'block'}),
                html.Small(sub, style={'fontSize': '10px', 'color': '#7a827e'}),
            ], style={'flex': '1', 'minWidth': '0'}),
            action_el,
            html.Button('✕', type='button', title='Remove from list', className='button quiet',
                        style={'padding': '4px 7px', 'fontSize': '10px'},
                        **{'data-dismiss-game': gid}),
        ], style={
            'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'padding': '9px 10px',
            'borderRadius': '6px', 'border': '1px solid #e7e8e2',
            'background': '#f0f3eb' if is_current else '#fafbf8', 'marginBottom': '6px',
        }))
    return html.Div(rows)


def player_bar(name, subtitle, symbol, active=False, avatar_symbol=None):
    icon = avatar_symbol if avatar_symbol else symbol
    return html.Div([
        html.Div(html.Img(src=piece_image(icon), alt=''), className='player-avatar'),
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


def _external_badges(entry):
    badges = []
    lichess = entry.get('lichess')
    if lichess and lichess.get('username'):
        badges.append(html.Span(
            f"Lichess ({lichess['username']}): {lichess['rating']} · {lichess['wins']}W/{lichess['losses']}L/{lichess['draws']}D",
            style={'fontSize': '10px', 'padding': '3px 7px', 'borderRadius': '4px',
                   'background': '#eef2eb', 'color': '#3f5238', 'fontWeight': '600', 'display': 'inline-block'},
        ))
    chesscom = entry.get('chesscom')
    if chesscom and chesscom.get('username'):
        badges.append(html.Span(
            f"Chess.com ({chesscom['username']}): {chesscom['rating']} · {chesscom['wins']}W/{chesscom['losses']}L/{chesscom['draws']}D",
            style={'fontSize': '10px', 'padding': '3px 7px', 'borderRadius': '4px',
                   'background': '#f4efe6', 'color': '#6e5233', 'fontWeight': '600', 'display': 'inline-block'},
        ))
    if not badges:
        return html.Span('—', style={'color': '#9a9f99', 'fontSize': '11px'})
    return html.Div(badges, style={'display': 'flex', 'flexDirection': 'column', 'gap': '4px'})


def render_external_account_box(platform_label, linked):
    if not linked or not linked.get('username'):
        return html.Div(
            f'No {platform_label} account linked yet.',
            style={'fontSize': '11px', 'color': '#7a827e', 'marginTop': '8px'},
        )
    return html.Div([
        html.Div([
            html.Span(className='online-dot', style={'marginRight': '6px'}),
            html.Strong(f"Linked: {linked['username']}"),
            html.Span(f"Rating {linked['rating']}",
                      style={'marginLeft': 'auto', 'fontSize': '11px', 'fontWeight': '650',
                             'background': '#e8eee2', 'color': '#52674a', 'padding': '3px 8px', 'borderRadius': '4px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '4px'}),
        html.Small(
            f"Synced record: {linked['wins']:,} Wins · {linked['losses']:,} Losses · {linked['draws']:,} Draws",
            style={'color': '#68766c', 'fontSize': '11px'},
        ),
    ], style={'marginTop': '10px', 'padding': '10px 12px', 'background': '#f9f8f4',
              'border': '1px solid #e7e8e2', 'borderRadius': '6px'})


def render_leaderboard_content(entries, current_auth):
    current_user = (current_auth or {}).get('username', '').lower() if (current_auth or {}).get('role') == 'user' else None
    if current_auth and current_auth.get('role') == 'user':
        w, l, d = current_auth.get('wins', 0), current_auth.get('losses', 0), current_auth.get('draws', 0)
        total = w + l + d
        win_rate = f'{(w / total * 100):.0f}%' if total else '0%'
        avatar_sym = current_auth.get('avatar') or 'K'
        display_name = current_auth.get('display_name') or current_auth['username']
        profile_banner = html.Div([
            html.Div(html.Img(src=piece_image(avatar_sym), alt=''), className='player-avatar'),
            html.Div([
                html.Span('YOUR STANDING', className='field-label', style={'marginBottom': '4px'}),
                html.Strong(display_name, style={'fontSize': '18px'}),
                html.Span(current_auth.get('rank', 'Contender'),
                          style={'marginLeft': '10px', 'fontSize': '11px', 'fontWeight': '650', 'padding': '4px 9px',
                                 'borderRadius': '5px', 'background': '#e8eee2', 'color': '#52674a'}),
            ]),
            html.Div([
                html.Div([html.Small('RATING', className='muted'), html.Strong(str(current_auth.get('rating', 1000)), style={'fontSize': '16px'})],
                         style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
                html.Div([html.Small('RECORD (W-L-D)', className='muted'), html.Strong(f'{w} - {l} - {d}', style={'fontSize': '16px'})],
                         style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
                html.Div([html.Small('WIN RATE', className='muted'), html.Strong(win_rate, style={'fontSize': '16px'})],
                         style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
            ], style={'display': 'flex', 'gap': '28px', 'marginLeft': 'auto'}),
        ], className='panel', style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '16px', 'marginBottom': '18px', 'background': '#f9f8f4'})
    else:
        profile_banner = html.Div([
            html.Strong('Playing as Guest'),
            html.Span(' — Register or log in to record your wins, losses, rating, rank and linked Lichess/Chess.com stats on the leaderboard.',
                      style={'color': '#7a827e', 'fontSize': '12px', 'marginLeft': '6px'}),
        ], className='panel', style={'marginBottom': '18px', 'padding': '16px 20px', 'background': '#f9f8f4'})

    if not entries:
        table = html.Div([
            html.Span('No ranked players yet.', className='empty-title'),
            html.P('Register an account and complete a game to claim the #1 spot.'),
        ], className='history-empty', style={'minHeight': '160px'})
    else:
        grid_cols = '44px 1.4fr 1fr 80px 105px 65px 1.6fr'
        header = html.Div([
            html.Span('#'),
            html.Span('PLAYER'),
            html.Span('RANK'),
            html.Span('RATING'),
            html.Span('W / L / D'),
            html.Span('WIN %'),
            html.Span('LINKED ACCOUNTS'),
        ], style={'display': 'grid', 'gridTemplateColumns': grid_cols, 'gap': '8px',
                  'fontSize': '10px', 'fontWeight': '650', 'letterSpacing': '1px', 'color': '#7a827e',
                  'padding': '10px 14px', 'borderBottom': '1px solid #e7e8e2'})
        rows = []
        for entry in entries:
            w, l, d = entry['wins'], entry['losses'], entry['draws']
            total = entry['games']
            win_pct = f'{(w / total * 100):.0f}%' if total else '—'
            is_me = current_user and entry['username'].lower() == current_user
            disp = entry.get('display_name') or entry['username']
            name_label = disp + (f" (@{entry['username']})" if disp != entry['username'] else '') + (' (You)' if is_me else '')
            avatar_sym = entry.get('avatar') or 'K'
            rows.append(html.Div([
                html.Strong(f'#{entry["position"]}', style={'color': '#c56a49' if entry['position'] <= 3 else '#7a827e'}),
                html.Div([
                    html.Img(src=piece_image(avatar_sym), alt='', style={'width': '24px', 'height': '24px', 'borderRadius': '5px', 'background': '#eeeee7', 'padding': '2px'}),
                    html.Strong(name_label),
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}),
                html.Span(entry['rank'], style={'fontSize': '11px', 'fontWeight': '600', 'color': '#52674a'}),
                html.Strong(str(entry['rating'])),
                html.Span(f'{w}W · {l}L · {d}D', style={'fontSize': '12px', 'color': '#68766c'}),
                html.Span(win_pct, style={'fontSize': '12px'}),
                _external_badges(entry),
            ], className='leaderboard-row', style={
                'display': 'grid',
                'gridTemplateColumns': grid_cols,
                'gap': '8px',
                'alignItems': 'center',
                'padding': '12px 14px',
                'borderBottom': '1px solid #f0f1ec',
                'background': '#f0f3eb' if is_me else 'transparent',
                'borderRadius': '6px' if is_me else '0',
            }))
        table = html.Div([header, *rows])

    tiers_note = html.Div([
        html.Span('RANK TIERS: ', style={'fontWeight': '650', 'color': '#26312f'}),
        html.Span('Novice (<900) · Contender (900+) · Club Player (1050+) · Tactician (1200+) · Master (1400+) · Grandmaster (1600+)'),
    ], style={'marginTop': '16px', 'fontSize': '11px', 'color': '#7a827e'})

    return html.Div([
        profile_banner,
        html.Div([table], className='panel', style={'padding': '12px 16px', 'overflowX': 'auto'}),
        tiers_note,
    ])


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


def render_open_games(open_rooms, saved_games=None):
    open_rooms = open_rooms or []
    my_codes = {g.get('code') for g in (saved_games or []) if g.get('mode') == 'multiplayer' and g.get('code')}
    if not open_rooms:
        return html.P(
            'No open games waiting right now. Click "Host game" to open a table for other players.',
            className='muted',
            style={'fontSize': '11px', 'margin': '6px 0 0', 'lineHeight': '1.45'},
        )
    rows = []
    for room in open_rooms:
        code = room['code']
        is_mine = code in my_codes
        action_btn = (
            html.Button(
                'Resume',
                type='button',
                className='button secondary',
                style={'padding': '5px 10px', 'fontSize': '10px'},
                **{'data-resume-game': f'mp:{code}'},
            )
            if is_mine
            else html.Button(
                'Join game',
                type='button',
                className='button primary',
                style={'padding': '5px 10px', 'fontSize': '10px'},
                **{'data-join-open-game': code},
            )
        )
        rows.append(
            html.Div([
                html.Img(
                    src=piece_image(room.get('host_avatar') or 'K'),
                    alt='',
                    style={
                        'width': '26px', 'height': '26px', 'borderRadius': '6px',
                        'background': '#eeeee7', 'border': '1px solid #e1e3dc', 'padding': '2px', 'flexShrink': '0',
                    },
                ),
                html.Div([
                    html.Div([
                        html.Strong(room.get('host_name') or 'Player', style={'fontSize': '11px', 'color': '#26312f'}),
                        html.Span(
                            'Your game' if is_mine else f'#{code}',
                            style={
                                'fontSize': '9px', 'fontWeight': '650', 'padding': '2px 5px', 'borderRadius': '4px',
                                'background': '#e8eee2' if is_mine else '#eeeee7',
                                'color': '#52674a' if is_mine else '#5f6864',
                            },
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px'}),
                    html.Small(
                        f"{room.get('host_rank') or 'Contender'} · Waiting for opponent",
                        style={'fontSize': '10px', 'color': '#7a827e', 'display': 'block'},
                    ),
                ], style={'minWidth': '0', 'flex': '1'}),
                action_btn,
            ], style={
                'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'padding': '8px 9px',
                'borderRadius': '7px', 'border': '1px solid #e7e8e2', 'background': '#fff', 'marginBottom': '6px',
            })
        )
    return html.Div(rows, style={'maxHeight': '190px', 'overflowY': 'auto'})


def multiplayer_styles(mode, mp, hub=None, saved_games=None):
    tab_base = {
        'flex': '1', 'padding': '7px 10px', 'fontSize': '11px', 'fontWeight': '650',
        'borderRadius': '6px', 'cursor': 'pointer', 'border': '1px solid #e7e8e2',
    }
    tab_active = {**tab_base, 'background': '#2f3a36', 'color': '#fff', 'borderColor': '#2f3a36'}
    tab_inactive = {**tab_base, 'background': '#fff', 'color': '#525b57'}
    if mode != 'multiplayer':
        return (
            {'display': 'none'},
            {'display': 'none'},
            tab_active,
            tab_inactive,
            {'display': 'none'},
            {'display': 'none'},
            {'display': 'none'},
            [],
            {'display': 'none'},
            [],
            {'display': 'none'},
            True,
        )
    mp = mp or {'view': 'choice'}
    code = mp.get('code')
    panel_style = {'display': 'block', 'marginTop': '16px'}
    box_base = {'background': '#f9f8f4', 'border': '1px solid #e7e8e2', 'borderRadius': '8px', 'padding': '14px'}
    open_rooms = hub.list_open_rooms() if hub else []
    find_children = render_open_games(open_rooms, saved_games)
    if code:
        return (
            panel_style,
            {'display': 'none'},
            tab_active,
            tab_inactive,
            {'display': 'none'},
            {'display': 'none'},
            {'display': 'none'},
            find_children,
            {**box_base, 'display': 'block'},
            render_room_status(mp),
            {'display': 'flex', 'marginTop': '8px', 'justifyContent': 'flex-end'},
            False,
        )
    tabs_row_style = {'display': 'flex', 'gap': '6px', 'marginBottom': '8px'}
    if mp.get('view') == 'find':
        return (
            panel_style,
            tabs_row_style,
            tab_inactive,
            tab_active,
            {'display': 'none'},
            {'display': 'none'},
            {**box_base, 'display': 'block'},
            find_children,
            {'display': 'none'},
            [],
            {'display': 'none'},
            False,
        )
    if mp.get('view') == 'join':
        return (
            panel_style,
            tabs_row_style,
            tab_active,
            tab_inactive,
            {'display': 'none'},
            {**box_base, 'display': 'block'},
            {'display': 'none'},
            find_children,
            {'display': 'none'},
            [],
            {'display': 'none'},
            True,
        )
    return (
        panel_style,
        tabs_row_style,
        tab_active,
        tab_inactive,
        {**box_base, 'display': 'block'},
        {'display': 'none'},
        {'display': 'none'},
        find_children,
        {'display': 'none'},
        [],
        {'display': 'none'},
        True,
    )


def create_app(app=None, hub=None, auth_store=None):
    ai = ChessAI(os.environ.get('PLOTLY_CHESS_MODEL'))
    hub = hub or DEFAULT_HUB
    auth_store = auth_store or DEFAULT_AUTH
    session_states = {}
    server_seq = 0
    if app is None:
        app = Dash(__name__, assets_folder=str(Path(__file__).resolve().parent.parent/'assets'))
    attach_websocket(app.server, hub)
    app.title = 'Plotly Chess · A little room to think'
    board = chess.Board()
    initial_state = {**game.new_game(), 'sid': secrets.token_hex(8), 'seq': 0}
    app.layout = html.Main([
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='game', storage_type='session', data=initial_state),
        dcc.Store(id='board-action'),
        dcc.Interval(id='mp-poll', interval=3000, n_intervals=0, disabled=True),
        html.Header([
            html.Button([
                html.Img(src=piece_image('N'), alt='', className='brand-mark'),
                html.Span(['plotly', html.Strong('chess')]),
            ], id='brand-home-btn', n_clicks=0, type='button', className='brand',
               title='Return to game join screen',
               style={'background': 'none', 'border': '0', 'padding': '0', 'cursor': 'pointer'}),
            html.Span('A little room to think.', className='brand-note'),
            html.Div([
                html.Button('🏠 Home', id='nav-play-btn', n_clicks=0, type='button',
                            className='button quiet', style={'display': 'none', 'padding': '6px 10px', 'fontSize': '11px'}),
                html.Button('🏆 Leaderboard', id='nav-leaderboard-btn', n_clicks=0, type='button',
                            className='button secondary', style={'display': 'none', 'padding': '6px 10px', 'fontSize': '11px'}),
                # Clickable User Avatar Dropdown Menu
                html.Details([
                    html.Summary([
                        html.Img(src=piece_image('K'), alt='Avatar', id='header-avatar',
                                 style={'width': '28px', 'height': '28px', 'borderRadius': '7px',
                                        'background': '#eeeee7', 'border': '1px solid #e1e3dc', 'padding': '2px'}),
                        html.Span('Welcome', id='user-badge', style={'fontWeight': '600', 'fontSize': '12px', 'color': '#26312f'}),
                        html.Span('▾', style={'fontSize': '10px', 'color': '#7a827e'}),
                    ], id='user-avatar-trigger', style={
                        'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'cursor': 'pointer',
                        'listStyle': 'none', 'padding': '5px 10px', 'borderRadius': '8px',
                        'border': '1px solid #e7e8e2', 'background': '#fff',
                    }),
                    html.Div([
                        html.Button('👤  My Profile', id='nav-profile-btn', n_clicks=0, type='button',
                                    className='button quiet',
                                    style={'width': '100%', 'textAlign': 'left', 'border': '0', 'padding': '8px 10px'}),
                        html.Button('↪  Switch account', id='auth-logout-btn', n_clicks=0, type='button',
                                    className='button quiet',
                                    style={'width': '100%', 'textAlign': 'left', 'border': '0', 'padding': '8px 10px', 'color': '#c56a49'}),
                    ], className='user-dropdown-menu', style={
                        'position': 'absolute', 'right': '0', 'top': 'calc(100% + 6px)', 'minWidth': '175px',
                        'background': '#fff', 'border': '1px solid #e7e8e2', 'borderRadius': '9px',
                        'boxShadow': '0 10px 25px rgba(38,49,47,0.1)', 'padding': '6px', 'zIndex': '100',
                        'display': 'flex', 'flexDirection': 'column', 'gap': '3px',
                    }),
                ], id='user-menu', style={'position': 'relative', 'display': 'none'}),
            ], style={'marginLeft': 'auto', 'display': 'flex', 'alignItems': 'center', 'gap': '8px'}),
        ], className='site-header'),

        # Pre-Home Welcome / Authentication Screen
        html.Section([
            html.Div([
                html.P('WELCOME TO PLOTLY CHESS', className='eyebrow'),
                html.H1('Step up to the board.', style={'fontSize': '28px', 'marginBottom': '8px'}),
                html.P('Log in to your account, register a new profile, or continue as a guest.',
                       style={'color': '#7a827e', 'fontSize': '13px', 'margin': '0 0 22px', 'lineHeight': '1.5'}),
                html.Label('USERNAME', htmlFor='auth-username', className='field-label'),
                dcc.Input(id='auth-username', type='text', value='', placeholder='Enter username',
                          autoComplete='username', n_submit=0,
                          style={'width': '100%', 'padding': '10px 12px', 'border': '1px solid #e7e8e2',
                                 'borderRadius': '6px', 'fontSize': '13px', 'marginBottom': '14px', 'background': '#fafbf8'}),
                html.Label('PASSWORD', htmlFor='auth-password', className='field-label'),
                dcc.Input(id='auth-password', type='password', value='', placeholder='Enter password',
                          autoComplete='current-password', n_submit=0,
                          style={'width': '100%', 'padding': '10px 12px', 'border': '1px solid #e7e8e2',
                                 'borderRadius': '6px', 'fontSize': '13px', 'marginBottom': '16px', 'background': '#fafbf8'}),
                html.Div([
                    html.Button('Login', id='auth-login-btn', n_clicks=0, type='button',
                                className='button primary', style={'flex': '1', 'padding': '11px 14px'}),
                    html.Button('Register', id='auth-register-btn', n_clicks=0, type='button',
                                className='button secondary', style={'flex': '1', 'padding': '11px 14px'}),
                ], style={'display': 'flex', 'gap': '10px'}),
                html.Div('Enter your details to log in or register, or play right away as a guest.',
                         id='auth-feedback', role='status', **{'aria-live': 'polite'},
                         style={'color': '#c56a49', 'fontSize': '11px', 'minHeight': '20px', 'margin': '12px 0'}),
                html.Div([
                    html.Span(style={'flex': '1', 'height': '1px', 'background': '#e7e8e2'}),
                    html.Span('OR', style={'fontSize': '9px', 'letterSpacing': '1.2px', 'color': '#7a827e', 'fontWeight': '650'}),
                    html.Span(style={'flex': '1', 'height': '1px', 'background': '#e7e8e2'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'margin': '6px 0 14px'}),
                html.Button('Play as guest →', id='auth-guest-btn', n_clicks=0, type='button',
                            className='button quiet', style={'width': '100%', 'padding': '11px 14px', 'fontSize': '12px'}),
            ], className='panel auth-card', style={'maxWidth': '420px', 'width': '100%', 'margin': '48px auto', 'padding': '30px'}),
        ], id='auth-screen', style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'minHeight': '65vh'}),

        # My Profile Screen
        html.Section([
            html.Div([
                html.Div([
                    html.P('ACCOUNT & INTEGRATIONS', className='eyebrow'),
                    html.H1('My Profile'),
                ]),
                html.Button('← Back to board', id='profile-back-btn', n_clicks=0, type='button',
                            className='button primary', style={'padding': '10px 16px'}),
            ], className='page-intro'),
            html.Div('', id='profile-feedback', role='status', **{'aria-live': 'polite'},
                     style={'fontSize': '12px', 'fontWeight': '600', 'color': '#52674a', 'minHeight': '22px', 'marginBottom': '12px'}),
            html.Div([
                # Left Column: Appearance & Password
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div(html.Img(src=piece_image('K'), alt='', id='profile-avatar-preview'), className='player-avatar',
                                     style={'width': '54px', 'height': '54px'}),
                            html.Div([
                                html.Strong('Player', id='profile-header-name', style={'fontSize': '17px', 'display': 'block'}),
                                html.Small('Contender · 1000 pts', id='profile-header-sub', style={'color': '#7a827e', 'fontSize': '12px'}),
                            ]),
                        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '14px', 'marginBottom': '20px'}),
                        html.Label('DISPLAY NAME', htmlFor='profile-display-name', className='field-label'),
                        dcc.Input(id='profile-display-name', type='text', value='', placeholder='Enter display name',
                                  style={'width': '100%', 'padding': '9px 11px', 'border': '1px solid #e7e8e2',
                                         'borderRadius': '6px', 'fontSize': '13px', 'marginBottom': '14px', 'background': '#fafbf8'}),
                        html.Label('CHOOSE AVATAR PIECE', htmlFor='profile-avatar-select', className='field-label'),
                        dcc.Dropdown(id='profile-avatar-select', options=AVATAR_OPTIONS, value='K',
                                     clearable=False, searchable=False, style={'marginBottom': '16px'}),
                        html.Button('Save profile changes', id='profile-save-btn', n_clicks=0, type='button',
                                    className='button primary', style={'width': '100%'}),
                    ], className='panel', style={'marginBottom': '18px'}),

                    html.Div([
                        html.H2('Change password', style={'marginBottom': '16px'}),
                        html.Label('CURRENT PASSWORD', htmlFor='profile-current-password', className='field-label'),
                        dcc.Input(id='profile-current-password', type='password', value='', placeholder='Current password',
                                  style={'width': '100%', 'padding': '9px 11px', 'border': '1px solid #e7e8e2',
                                         'borderRadius': '6px', 'fontSize': '13px', 'marginBottom': '12px', 'background': '#fafbf8'}),
                        html.Label('NEW PASSWORD', htmlFor='profile-new-password', className='field-label'),
                        dcc.Input(id='profile-new-password', type='password', value='', placeholder='New password (4+ chars)',
                                  style={'width': '100%', 'padding': '9px 11px', 'border': '1px solid #e7e8e2',
                                         'borderRadius': '6px', 'fontSize': '13px', 'marginBottom': '16px', 'background': '#fafbf8'}),
                        html.Button('Update password', id='profile-password-btn', n_clicks=0, type='button',
                                    className='button secondary', style={'width': '100%'}),
                    ], className='panel'),
                ]),

                # Right Column: Link Lichess & Chess.com
                html.Div([
                    html.Div([
                        html.Div([
                            html.H2('Link Lichess Account'),
                            html.Span('LIVE STATS SYNC', className='small-badge'),
                        ], className='card-heading', style={'marginBottom': '10px'}),
                        html.P('Connect your Lichess username to sync your rating, wins, losses and draws to your account and the Leaderboard.',
                               style={'color': '#7a827e', 'fontSize': '12px', 'margin': '0 0 12px', 'lineHeight': '1.5'}),
                        html.Label('LICHESS USERNAME', htmlFor='profile-lichess-input', className='field-label'),
                        html.Div([
                            dcc.Input(id='profile-lichess-input', type='text', value='', placeholder='e.g. DrNykterstein',
                                      style={'flex': '1', 'minWidth': '0', 'padding': '9px 11px', 'border': '1px solid #e7e8e2',
                                             'borderRadius': '6px', 'fontSize': '13px', 'background': '#fafbf8'}),
                            html.Button('Link Lichess', id='profile-lichess-btn', n_clicks=0, type='button', className='button primary'),
                        ], style={'display': 'flex', 'gap': '8px'}),
                        html.Div(id='profile-lichess-status'),
                    ], className='panel', style={'marginBottom': '18px'}),

                    html.Div([
                        html.Div([
                            html.H2('Link Chess.com Account'),
                            html.Span('LIVE STATS SYNC', className='small-badge'),
                        ], className='card-heading', style={'marginBottom': '10px'}),
                        html.P('Connect your Chess.com username to fetch your published rapid/blitz rating and career W/L/D stats.',
                               style={'color': '#7a827e', 'fontSize': '12px', 'margin': '0 0 12px', 'lineHeight': '1.5'}),
                        html.Label('CHESS.COM USERNAME', htmlFor='profile-chesscom-input', className='field-label'),
                        html.Div([
                            dcc.Input(id='profile-chesscom-input', type='text', value='', placeholder='e.g. MagnusCarlsen',
                                      style={'flex': '1', 'minWidth': '0', 'padding': '9px 11px', 'border': '1px solid #e7e8e2',
                                             'borderRadius': '6px', 'fontSize': '13px', 'background': '#fafbf8'}),
                            html.Button('Link Chess.com', id='profile-chesscom-btn', n_clicks=0, type='button', className='button primary'),
                        ], style={'display': 'flex', 'gap': '8px'}),
                        html.Div(id='profile-chesscom-status'),
                    ], className='panel'),
                ]),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(320px, 1fr))', 'gap': '24px', 'alignItems': 'start'}),
        ], id='profile-screen', style={'display': 'none', 'minHeight': '65vh'}),

        # Leaderboard Screen
        html.Section([
            html.Div([
                html.Div([
                    html.P('RANKINGS & STANDINGS', className='eyebrow'),
                    html.H1('Player Leaderboard'),
                ]),
                html.Button('← Back to board', id='leaderboard-back-btn', n_clicks=0, type='button',
                            className='button primary', style={'padding': '10px 16px'}),
            ], className='page-intro'),
            html.Div(id='leaderboard-content'),
        ], id='leaderboard-screen', style={'display': 'none', 'minHeight': '65vh'}),

        # Main Home Game Screen (shown after Login, Register, or Play as guest)
        html.Div([
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
                ], id='play-area', className='play-area', **{'aria-label': 'Play chess'}),
                html.Aside([
                    html.Section([
                        html.Div([html.H2('Your game'), html.Span('STANDARD', className='small-badge')], className='card-heading'),
                        html.Label('PLAY MODE', className='field-label'),
                        dcc.RadioItems(id='mode', options=[
                            {'label': 'Vs computer', 'value': 'ai'},
                            {'label': 'Two players', 'value': 'local'},
                            {'label': 'Multiplayer', 'value': 'multiplayer'},
                        ], value='ai', className='mode-switch'),
                        html.Div([
                            # Lobby Tabs: Create / Join vs Find a game
                            html.Div([
                                html.Button('Host / Join', id='mp-tab-lobby-btn', n_clicks=0, type='button'),
                                html.Button('Find a game', id='mp-tab-find-btn', n_clicks=0, type='button'),
                            ], id='mp-tabs-row', style={'display': 'none'}),
                            html.Div([
                                html.Span('MULTIPLAYER LOBBY', className='field-label'),
                                html.P('Host an open game for anyone to join, create a room code, or join a friend’s game.',
                                       className='mp-note', style={'margin': '0 0 11px', 'fontSize': '11px'}),
                                html.Div([
                                    html.Button('Host game', id='mp-host-btn', n_clicks=0, type='button',
                                                className='button primary mp-choice-btn', style={'flex': '1'}),
                                    html.Button('Create game', id='mp-create-btn', n_clicks=0, type='button',
                                                className='button secondary mp-choice-btn', style={'flex': '1'}),
                                    html.Button('Join game', id='mp-choose-join-btn', n_clicks=0, type='button',
                                                className='button secondary mp-choice-btn', style={'flex': '1'}),
                                ], className='mp-choice-row', style={'display': 'flex', 'gap': '6px', 'flexWrap': 'wrap'}),
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
                            html.Div([
                                html.Span('OPEN GAMES WAITING FOR PLAYERS', className='field-label'),
                                html.P('Join any open table below to start playing immediately.',
                                       className='mp-note', style={'margin': '0 0 10px', 'fontSize': '11px'}),
                                html.Div(render_open_games([]), id='mp-open-games-list'),
                            ], id='mp-view-find', className='mp-box', style={'display': 'none'}),
                            html.Div([], id='mp-view-room', className='mp-box', style={'display': 'none'}),
                            html.Div([
                                html.Button('Leave / Switch game', id='mp-leave-btn', n_clicks=0, type='button',
                                            className='button quiet mp-sub-btn', style={'padding': '6px 10px', 'fontSize': '10px'}),
                            ], id='mp-leave-row', style={'display': 'none'}),
                        ], id='multiplayer-panel', className='mp-panel', style={'display': 'none'}),
                        html.Div([
                            html.Div([html.Span(className='turn-dot'), html.Strong('White to move.', id='status')], className='turn-status'),
                            html.Div('Choose your opening. Take your time.', id='feedback', className='feedback', role='status', **{'aria-live': 'polite'}),
                            html.Div([html.Button('✦  Get a hint', id='hint', className='button secondary'),
                                      html.Button('↺  New game', id='reset', className='button quiet')], className='game-actions'),
                        ], id='board-controls'),
                    ], className='panel game-panel'),

                    # In-Progress Games Panel
                    html.Section([
                        html.Div([
                            html.H2('In-progress games'),
                            html.Span('0 active', id='active-games-count', className='muted'),
                        ], className='card-heading', style={'marginBottom': '12px'}),
                        html.Div(render_active_games([], None), id='active-games-list',
                                 style={'maxHeight': '180px', 'overflowY': 'auto'}),
                    ], className='panel', id='active-games-panel'),

                    html.Section([
                        html.Div([html.H2('Move history'), html.Span('0 moves', id='move-count', className='muted')], className='card-heading'),
                        html.Div([html.Span('#'), html.Span('WHITE'), html.Span('BLACK')], className='history-labels'),
                        html.Div(move_history(board), id='history', className='history-list', **{'aria-label': 'Move history'}),
                    ], id='history-panel', className='panel history-panel'),
                    html.Details([
                        html.Summary('Move options'),
                        html.Label('ENTER A MOVE', htmlFor='notation', className='field-label'),
                        html.Div([dcc.Input(id='notation', placeholder='e.g. Nf3 or e2e4', debounce=True),
                                  html.Button('Play', id='play', className='button primary')], className='notation-row'),
                        html.Label('PROMOTE PAWNS TO', htmlFor='promotion', className='field-label'),
                        dcc.Dropdown(id='promotion', options=[{'label': n, 'value': v} for n, v in
                            [('Queen', 'q'), ('Rook', 'r'), ('Bishop', 'b'), ('Knight', 'n')]], value='q', clearable=False, searchable=False),
                    ], id='options-panel', className='panel options-panel'),
                    html.Div([html.Span(className='engine-dot'),
                              html.Span('NPC expert routing' if ai.router else 'Material search', id='engine-label')], className='engine-label'),
                ], className='game-sidebar'),
            ], id='game-layout', className='game-layout'),
        ], id='home-screen', style={'display': 'none'}),

        html.Footer([html.Span('Less distraction. More chess.'), html.Span('Built with Python · Made for your next move.')]),
    ], className='app-container')

    @app.callback(
        Output('game', 'data'), Output('board', 'children'), Output('status', 'children'), Output('history', 'children'),
        Output('feedback', 'children'), Output('opponent', 'children'), Output('player', 'children'), Output('move-count', 'children'),
        Output('multiplayer-panel', 'style'), Output('mp-tabs-row', 'style'),
        Output('mp-tab-lobby-btn', 'style'), Output('mp-tab-find-btn', 'style'),
        Output('mp-view-choice', 'style'), Output('mp-view-join', 'style'),
        Output('mp-view-find', 'style'), Output('mp-open-games-list', 'children'),
        Output('mp-view-room', 'style'), Output('mp-view-room', 'children'), Output('mp-leave-row', 'style'),
        Output('mp-poll', 'disabled'), Output('mode', 'value'), Output('url', 'search'),
        Output('auth-screen', 'style'), Output('home-screen', 'style'), Output('auth-feedback', 'children'),
        Output('user-badge', 'children'), Output('user-menu', 'style'),
        Output('leaderboard-screen', 'style'), Output('leaderboard-content', 'children'),
        Output('nav-play-btn', 'style'), Output('nav-leaderboard-btn', 'style'),
        Output('profile-screen', 'style'), Output('profile-feedback', 'children'),
        Output('header-avatar', 'src'), Output('profile-avatar-preview', 'src'),
        Output('profile-header-name', 'children'), Output('profile-header-sub', 'children'),
        Output('profile-display-name', 'value'), Output('profile-avatar-select', 'value'),
        Output('profile-lichess-status', 'children'), Output('profile-chesscom-status', 'children'),
        Output('active-games-list', 'children'), Output('active-games-count', 'children'),
        Output('play-area', 'style'), Output('game-layout', 'style'),
        Output('board-controls', 'style'), Output('history-panel', 'style'), Output('options-panel', 'style'),
        Input('board-action', 'data'), Input('play', 'n_clicks'), Input('reset', 'n_clicks'),
        Input('hint', 'n_clicks'), Input('mode', 'value'),
        Input('mp-host-btn', 'n_clicks'), Input('mp-create-btn', 'n_clicks'), Input('mp-choose-join-btn', 'n_clicks'),
        Input('mp-tab-lobby-btn', 'n_clicks'), Input('mp-tab-find-btn', 'n_clicks'),
        Input('mp-join-submit', 'n_clicks'), Input('join-code-input', 'n_submit'),
        Input('mp-back-choice', 'n_clicks'), Input('mp-leave-btn', 'n_clicks'),
        Input('mp-poll', 'n_intervals'), Input('url', 'search'), Input('url', 'pathname'),
        Input('auth-login-btn', 'n_clicks'), Input('auth-register-btn', 'n_clicks'),
        Input('auth-guest-btn', 'n_clicks'), Input('auth-logout-btn', 'n_clicks'),
        Input('auth-username', 'n_submit'), Input('auth-password', 'n_submit'),
        Input('nav-leaderboard-btn', 'n_clicks'), Input('nav-play-btn', 'n_clicks'), Input('leaderboard-back-btn', 'n_clicks'),
        Input('nav-profile-btn', 'n_clicks'), Input('profile-back-btn', 'n_clicks'),
        Input('profile-save-btn', 'n_clicks'), Input('profile-password-btn', 'n_clicks'),
        Input('profile-lichess-btn', 'n_clicks'), Input('profile-chesscom-btn', 'n_clicks'),
        Input('brand-home-btn', 'n_clicks'),
        State('notation', 'value'), State('promotion', 'value'), State('game', 'data'),
        State('join-code-input', 'value'), State('auth-username', 'value'), State('auth-password', 'value'),
        State('profile-display-name', 'value'), State('profile-avatar-select', 'value'),
        State('profile-current-password', 'value'), State('profile-new-password', 'value'),
        State('profile-lichess-input', 'value'), State('profile-chesscom-input', 'value'))
    def update(action, play_count, reset_count, hint_count, mode,
               host_clicks, create_clicks, choose_join_clicks, tab_lobby_clicks, tab_find_clicks,
               join_clicks, join_submit, back_clicks, leave_clicks,
               poll_ticks, url_search, url_pathname,
               login_clicks, register_clicks, guest_clicks, logout_clicks, user_submit, pass_submit,
               nav_lb_clicks, nav_play_clicks, lb_back_clicks,
               nav_prof_clicks, prof_back_clicks, prof_save_clicks, prof_pw_clicks,
               prof_lichess_clicks, prof_chesscom_clicks, brand_home_clicks,
               notation, promotion, state, join_code, auth_username, auth_password,
               prof_display_name, prof_avatar, prof_curr_pw, prof_new_pw,
               prof_lichess_user, prof_chesscom_user):
        nonlocal server_seq
        input_mode = mode
        trigger = ctx.triggered_id
        raw_state = dict(state or {})
        client_seq = raw_state.get('seq')
        sid = raw_state.get('sid')
        state = dict(state or game.new_game())
        cached = (hub.get_session(sid, max_age=5.0) or session_states.get(sid)) if (sid and trigger is not None) else None
        if (
            cached
            and (client_seq is not None or trigger in ('mode', 'url', 'mp-poll'))
            and cached['state'].get('seq', 0) >= int(client_seq or 0)
        ):
            state = dict(cached['state'])
            if trigger != 'mode' and state.get('mode'):
                mode = state['mode']
        if not sid:
            sid = secrets.token_hex(8)
            state['sid'] = sid
        auth = state.get('auth')
        url_code = extract_code_from_url(url_search, url_pathname)
        if not auth and url_code:
            auth = auth_store.guest()
            state['auth'] = auth
        page = state.get('page', 'home')
        in_lobby = bool(state.get('in_lobby', False))
        mp = dict(state.get('mp') or {'view': 'choice'})
        saved_games = list(state.get('saved_games') or (auth or {}).get('saved_games') or [])
        message = 'Drag to move. Click a piece to see its legal moves.'
        auth_feedback = ''
        profile_feedback = ''
        is_browser_premount = (trigger is None and url_search is None and url_pathname is None)
        if is_browser_premount and raw_state.get('mode'):
            mode = raw_state['mode']
        is_initial_url_load = (
            (trigger is None and not is_browser_premount)
            or (trigger == 'url' and (bool(state.get('awaiting_initial_url')) or not raw_state.get('sid')))
        )
        if is_browser_premount:
            state['awaiting_initial_url'] = True
        elif trigger == 'url' and state.get('awaiting_initial_url'):
            state = {**state, 'awaiting_initial_url': False}

        # If URL has no ?code=... on initial page load / reload, exit any stale room from sessionStorage
        if (is_initial_url_load or trigger == 'url') and not url_code and mp.get('code'):
            prev_code = mp.get('code')
            saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
            page = 'home'
            in_lobby = True
            mode = 'multiplayer'
            mp = {'view': 'choice'}
            state = {
                **game.new_game(),
                **({'sid': sid} if sid else {}),
                'mp': mp,
                'page': page,
                'in_lobby': True,
                'left_code': prev_code,
                'saved_games': saved_games,
                **({'auth': auth} if auth else {}),
            }
            message = 'Returned to the game lobby. Create or join a game, or resume an active game.'
        elif is_initial_url_load and url_code:
            state = {**state, 'left_code': None}

        # Handle authentication, profile & navigation actions
        if trigger in ('auth-login-btn', 'auth-username', 'auth-password'):
            try:
                auth = auth_store.login(auth_username, auth_password)
                for g in auth.get('saved_games') or []:
                    saved_games = _upsert_saved_game(saved_games, g)
                page = 'home'
                if url_code:
                    mode = 'multiplayer'
                    in_lobby = False
                else:
                    mode = 'multiplayer'
                    in_lobby = True
                    mp = {'view': 'choice'}
                state = {**state, 'auth': auth, 'page': page, 'in_lobby': in_lobby, 'mp': mp, 'saved_games': saved_games}
                message = f'Welcome back, {auth["display_name"]} ({auth["rank"]}). Choose or start a game.'
            except ValueError as exc:
                auth = None
                auth_feedback = str(exc)
        elif trigger == 'auth-register-btn':
            try:
                auth = auth_store.register(auth_username, auth_password)
                page = 'home'
                if url_code:
                    mode = 'multiplayer'
                    in_lobby = False
                else:
                    mode = 'multiplayer'
                    in_lobby = True
                    mp = {'view': 'choice'}
                state = {**state, 'auth': auth, 'page': page, 'in_lobby': in_lobby, 'mp': mp, 'saved_games': saved_games}
                message = f'Account created. Welcome, {auth["display_name"]} ({auth["rank"]})!'
            except ValueError as exc:
                auth = None
                auth_feedback = str(exc)
        elif trigger == 'auth-guest-btn':
            auth = auth_store.guest()
            page = 'home'
            if url_code:
                mode = 'multiplayer'
                in_lobby = False
            else:
                mode = 'multiplayer'
                in_lobby = True
                mp = {'view': 'choice'}
            state = {**state, 'auth': auth, 'page': page, 'in_lobby': in_lobby, 'mp': mp, 'saved_games': saved_games}
            message = 'Playing as guest. Host, join, or start a game.'
        elif trigger == 'auth-logout-btn':
            auth = None
            page = 'home'
            in_lobby = False
            saved_games = []
            state = {**game.new_game(), **({'sid': sid} if sid else {})}
            auth_feedback = 'Signed out. Log in, register, or play as guest.'
        elif trigger in ('brand-home-btn', 'nav-play-btn'):
            # Save current game into in-progress list and exit view/game back to the game join screen
            prev_code = mp.get('code') or state.get('left_code') or url_code
            saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
            page = 'home'
            in_lobby = True
            mode = 'multiplayer'
            mp = {'view': 'choice'}
            state = {
                **game.new_game(),
                **({'sid': sid} if sid else {}),
                'mp': mp,
                'page': page,
                'in_lobby': True,
                'left_code': prev_code,
                'saved_games': saved_games,
                **({'auth': auth} if auth else {}),
            }
            message = 'Returned to the game lobby. Create or join a game, or resume an active game.'
        elif trigger == 'nav-leaderboard-btn':
            prev_code = mp.get('code') or state.get('left_code') or url_code
            saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
            page = 'leaderboard'
            if mp.get('code'):
                in_lobby = True
                mp = {'view': 'choice'}
            state = {**state, 'mp': mp, 'page': page, 'in_lobby': in_lobby, 'left_code': prev_code, 'saved_games': saved_games}
        elif trigger == 'nav-profile-btn':
            prev_code = mp.get('code') or state.get('left_code') or url_code
            saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
            page = 'profile'
            if mp.get('code'):
                in_lobby = True
                mp = {'view': 'choice'}
            state = {**state, 'mp': mp, 'page': page, 'in_lobby': in_lobby, 'left_code': prev_code, 'saved_games': saved_games}
        elif trigger in ('leaderboard-back-btn', 'profile-back-btn'):
            page = 'home'
            state = {**state, 'page': page}
        elif trigger == 'profile-save-btn':
            page = 'profile'
            try:
                if not auth or auth.get('role') != 'user':
                    auth = {
                        **(auth or auth_store.guest()),
                        'display_name': (prof_display_name or 'Guest').strip() or 'Guest',
                        'avatar': prof_avatar or 'K',
                    }
                    profile_feedback = 'Updated guest appearance for this session.'
                else:
                    auth = auth_store.update_profile(auth['username'], display_name=prof_display_name, avatar=prof_avatar)
                    profile_feedback = 'Profile display name and avatar saved.'
                state = {**state, 'auth': auth, 'page': page}
            except ValueError as exc:
                profile_feedback = str(exc)
        elif trigger == 'profile-password-btn':
            page = 'profile'
            try:
                if not auth or auth.get('role') != 'user':
                    raise ValueError('Register an account to set a password.')
                auth = auth_store.change_password(auth['username'], prof_curr_pw, prof_new_pw)
                state = {**state, 'auth': auth, 'page': page}
                profile_feedback = 'Password updated successfully.'
            except ValueError as exc:
                profile_feedback = str(exc)
        elif trigger == 'profile-lichess-btn':
            page = 'profile'
            try:
                if not auth or auth.get('role') != 'user':
                    raise ValueError('Register or log in to link a Lichess account to the database.')
                auth = auth_store.link_external_account(auth['username'], 'lichess', prof_lichess_user)
                state = {**state, 'auth': auth, 'page': page}
                profile_feedback = f"Linked Lichess account '{auth['lichess']['username']}' (Rating {auth['lichess']['rating']})."
            except ValueError as exc:
                profile_feedback = str(exc)
        elif trigger == 'profile-chesscom-btn':
            page = 'profile'
            try:
                if not auth or auth.get('role') != 'user':
                    raise ValueError('Register or log in to link a Chess.com account to the database.')
                auth = auth_store.link_external_account(auth['username'], 'chesscom', prof_chesscom_user)
                state = {**state, 'auth': auth, 'page': page}
                profile_feedback = f"Linked Chess.com account '{auth['chesscom']['username']}' (Rating {auth['chesscom']['rating']})."
            except ValueError as exc:
                profile_feedback = str(exc)

        host_display = (auth or {}).get('display_name') or (auth['username'] if auth and auth.get('role') == 'user' else 'Guest')
        host_rank_label = f'{auth["rank"]} ({auth["rating"]})' if auth and auth.get('role') == 'user' else 'Guest'
        host_avatar_sym = (auth or {}).get('avatar') or 'K'

        if trigger == 'mp-poll':
            if mode == 'multiplayer' and mp.get('view') == 'find' and not mp.get('code'):
                message = 'Showing open games waiting for players.'
            elif mode != 'multiplayer' or not mp.get('code'):
                return (no_update,) * 48
            else:
                live = hub.get_room(mp['code'], mp.get('token'))
                if not live or (live['version'] == mp.get('version') and live['connected'] == mp.get('connected')):
                    return (no_update,) * 48
                mp = {'view': 'room', **live}
                state = {**state, 'moves': live['moves'], 'selected': state.get('selected'), 'mp': mp}
                message = live['message']

        elif mode == 'multiplayer' and mp.get('code') and trigger not in (
            None, 'url', 'mp-host-btn', 'mp-create-btn', 'mp-leave-btn', 'mp-tab-lobby-btn', 'mp-tab-find-btn',
            'mode', 'auth-logout-btn', 'brand-home-btn', 'nav-play-btn', 'nav-profile-btn', 'nav-leaderboard-btn',
        ):
            live = hub.get_room(mp['code'], mp.get('token'))
            if live:
                if live['moves'] != state.get('moves'):
                    state = {**state, 'moves': live['moves'], 'selected': None}
                mp = {'view': 'room', **live}
                state = {**state, 'mp': mp}

        auth_triggers = ('auth-login-btn', 'auth-register-btn', 'auth-guest-btn', 'auth-username', 'auth-password')
        seen_codes = list(state.get('seen_codes') or [])
        try:
            should_join_url_code = bool(
                url_code
                and (
                    is_initial_url_load
                    or (trigger in auth_triggers and auth)
                    or (trigger == 'url' and (in_lobby or (state.get('mp') or {}).get('code') != url_code))
                )
                and (is_initial_url_load or url_code != state.get('left_code') or trigger == 'url')
            )
            if should_join_url_code:
                saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
                mode = 'multiplayer'
                page = 'home'
                in_lobby = False
                existing_token = mp.get('token') if mp.get('code') == url_code else None
                if not existing_token:
                    for g in saved_games:
                        if g.get('code') == url_code and g.get('token'):
                            existing_token = g['token']
                            break
                snap = hub.get_room(url_code, existing_token) if (mp.get('code') == url_code and existing_token) else hub.join_room(url_code, existing_token)
                if not snap:
                    snap = hub.join_room(url_code, existing_token)
                mp = {'view': 'room', **snap}
                seen_codes = list(set(seen_codes + [url_code]))
                state = {**state, 'moves': snap['moves'], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': False, 'left_code': None, 'seen_codes': seen_codes}
                message = snap['message']
            elif trigger == 'mode':
                if state.get('expected_mode_echo') == mode:
                    state = {**state, 'expected_mode_echo': None}
                    if mode == 'multiplayer' and mp.get('code') and mp.get('message'):
                        message = mp['message']
                else:
                    prev_code = mp.get('code') or state.get('left_code') or url_code
                    page = 'home'
                    saved_games = _SnapshotCurrentIntoSaved('multiplayer' if mp.get('code') else mode, state, saved_games, hub)
                    if mode == 'multiplayer':
                        in_lobby = True
                        mp = {'view': 'choice'}
                        state = {
                            **game.new_game(),
                            **({'sid': sid} if sid else {}),
                            'mp': mp,
                            'page': page,
                            'in_lobby': True,
                            'left_code': prev_code,
                            'saved_games': saved_games,
                            **({'auth': auth} if auth else {}),
                        }
                        message = 'Host an open game, create a room code, or browse Find a game.'
                    else:
                        in_lobby = False
                        state = {
                            **game.new_game(),
                            **({'sid': sid} if sid else {}),
                            'page': page,
                            'in_lobby': False,
                            'left_code': prev_code,
                            'saved_games': saved_games,
                            **({'auth': auth} if auth else {}),
                        }
                        message = 'A fresh board. Your move.'
            elif trigger in ('mp-host-btn', 'mp-create-btn'):
                saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
                mode = 'multiplayer'
                page = 'home'
                in_lobby = False
                snap = hub.create_room(host_name=host_display, host_rank=host_rank_label, host_avatar=host_avatar_sym, is_public=True)
                mp = {'view': 'room', **snap}
                state = {**state, 'moves': snap['moves'], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': False, 'left_code': url_code, 'recorded_game_key': None}
                message = snap['message']
            elif trigger == 'mp-tab-lobby-btn':
                prev_code = mp.get('code') or state.get('left_code') or url_code
                saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
                mode = 'multiplayer'
                page = 'home'
                in_lobby = True
                mp = {'view': 'choice'}
                state = {**state, 'moves': [], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': True, 'left_code': prev_code}
                message = 'Host an open game, create a code, or join a friend’s game.'
            elif trigger == 'mp-tab-find-btn':
                prev_code = mp.get('code') or state.get('left_code') or url_code
                saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
                mode = 'multiplayer'
                page = 'home'
                in_lobby = True
                mp = {'view': 'find'}
                state = {**state, 'moves': [], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': True, 'left_code': prev_code}
                open_count = len(hub.list_open_rooms())
                message = f'{open_count} open {"game" if open_count == 1 else "games"} waiting for players.'
            elif trigger == 'mp-choose-join-btn':
                prev_code = mp.get('code') or state.get('left_code') or url_code
                saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
                mode = 'multiplayer'
                page = 'home'
                in_lobby = True
                mp = {'view': 'join'}
                state = {**state, 'moves': [], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': True, 'left_code': prev_code}
                message = 'Enter the 6-character code from your friend.'
            elif trigger == 'mp-back-choice':
                prev_code = mp.get('code') or state.get('left_code') or url_code
                mode = 'multiplayer'
                page = 'home'
                in_lobby = True
                mp = {'view': 'choice'}
                state = {**state, 'moves': [], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': True, 'left_code': prev_code}
                message = 'Create a game or join with a friend’s code.'
            elif trigger in ('mp-join-submit', 'join-code-input'):
                saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
                mode = 'multiplayer'
                page = 'home'
                in_lobby = False
                cleaned_join = (join_code or '').strip().upper()
                existing_token = mp.get('token') if mp.get('code') == cleaned_join else None
                if not existing_token:
                    for g in saved_games:
                        if g.get('code') == cleaned_join and g.get('token'):
                            existing_token = g['token']
                            break
                snap = hub.join_room(cleaned_join, existing_token)
                mp = {'view': 'room', **snap}
                state = {**state, 'moves': snap['moves'], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': False, 'left_code': url_code, 'recorded_game_key': None}
                message = snap['message']
            elif trigger == 'mp-leave-btn':
                prev_code = mp.get('code') or state.get('left_code') or url_code
                saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
                mode = 'multiplayer'
                page = 'home'
                in_lobby = True
                mp = {'view': 'choice'}
                state = {**state, 'moves': [], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': True, 'left_code': prev_code, 'recorded_game_key': None}
                message = 'Game saved to your in-progress list. Create or join another game.'
            elif trigger == 'reset':
                page = 'home'
                if mode == 'multiplayer' and mp.get('code'):
                    snap = hub.reset_room(mp['code'], mp.get('token'))
                    mp = {'view': 'room', **snap}
                    state = {**state, 'moves': snap['moves'], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': False, 'recorded_game_key': None}
                    message = snap['message']
                else:
                    prev_code = mp.get('code') or state.get('left_code') or url_code
                    saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
                    state = {
                        **game.new_game(),
                        **({'sid': sid} if sid else {}),
                        'page': page,
                        'in_lobby': False,
                        'left_code': prev_code,
                        'saved_games': saved_games,
                        **({'mp': mp} if mode == 'multiplayer' else {}),
                        **({'auth': auth} if auth else {}),
                    }
                    message = 'A fresh board. Your move.'
            elif trigger == 'hint':
                position = game.restore(state)
                move, _ = ai.choose(position)
                message = f'Try {position.san(move)}.'
            elif trigger == 'play' and notation:
                page = 'home'
                in_lobby = False
                if mode == 'multiplayer':
                    if not mp.get('code'):
                        raise ValueError('Create or join a multiplayer game first.')
                    snap, san = hub.play_move(mp['code'], mp.get('token'), notation.strip())
                    mp = {'view': 'room', **snap}
                    state = {**state, 'moves': snap['moves'], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': False}
                    message = f'Played {san}.'
                else:
                    updated, san = game.play(state, notation.strip())
                    state = {**state, **updated, 'page': page, 'in_lobby': False}
                    message = f'Played {san}.'
            elif trigger == 'board-action' and action:
                kind = action.get('type')
                if kind == 'go-home':
                    prev_code = mp.get('code') or state.get('left_code') or url_code
                    saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
                    page = 'home'
                    in_lobby = True
                    mode = 'multiplayer'
                    mp = {'view': 'choice'}
                    state = {
                        **game.new_game(),
                        **({'sid': sid} if sid else {}),
                        'mp': mp,
                        'page': page,
                        'in_lobby': True,
                        'left_code': prev_code,
                        'saved_games': saved_games,
                        **({'auth': auth} if auth else {}),
                    }
                    message = 'Returned to the game lobby. Create or join a game, or resume an active game.'
                elif kind == 'resume-game':
                    target_id = action.get('game_id')
                    prev_code = mp.get('code') or state.get('left_code') or url_code
                    saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
                    found = next((g for g in saved_games if g.get('id') == target_id), None)
                    if not found and isinstance(target_id, str) and target_id.startswith('mp:'):
                        fallback_code = target_id.split(':', 1)[1].strip().upper()
                        if fallback_code:
                            found = {'id': f'mp:{fallback_code}', 'mode': 'multiplayer', 'code': fallback_code}
                    if found:
                        page = 'home'
                        in_lobby = False
                        if found.get('mode') == 'multiplayer' and found.get('code'):
                            mode = 'multiplayer'
                            resume_code = found['code']
                            resume_token = found.get('token')
                            live = hub.get_room(resume_code, resume_token)
                            if live and not live.get('color') and not resume_token:
                                live = hub.join_room(resume_code, None)
                            mp = {'view': 'room', **(live or found)}
                            if not mp.get('color'):
                                mp['color'] = found.get('color') or 'white'
                            if not mp.get('token') and resume_token:
                                mp['token'] = resume_token
                            state = {
                                **state,
                                'moves': list(mp.get('moves') or []),
                                'selected': None,
                                'mp': mp,
                                'page': page,
                                'in_lobby': False,
                                'left_code': prev_code,
                            }
                            message = f"Resumed Room {resume_code}."
                        else:
                            mode = found.get('mode', 'ai')
                            mp = {'view': 'choice'}
                            state = {
                                **game.new_game(),
                                **({'sid': sid} if sid else {}),
                                'moves': list(found.get('moves') or []),
                                'local_id': found.get('local_id'),
                                'mp': mp,
                                'page': page,
                                'in_lobby': False,
                                'left_code': prev_code,
                                'saved_games': saved_games,
                                **({'auth': auth} if auth else {}),
                            }
                            message = f"Resumed {'Vs computer' if mode == 'ai' else 'Two players'} game."
                elif kind == 'dismiss-game':
                    target_id = action.get('game_id')
                    prev_code = mp.get('code') or state.get('left_code') or url_code
                    saved_games = [g for g in saved_games if g.get('id') != target_id]
                    if _current_game_id(mode, state) == target_id:
                        mode = 'multiplayer'
                        in_lobby = True
                        mp = {'view': 'choice'}
                        state = {
                            **game.new_game(),
                            **({'sid': sid} if sid else {}),
                            'mp': mp,
                            'page': page,
                            'in_lobby': True,
                            'left_code': prev_code,
                            'saved_games': saved_games,
                            **({'auth': auth} if auth else {}),
                        }
                        message = 'Removed game.'
                elif kind == 'mp-create':
                    saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
                    mode = 'multiplayer'
                    page = 'home'
                    in_lobby = False
                    snap = hub.create_room(host_name=host_display, host_rank=host_rank_label, host_avatar=host_avatar_sym, is_public=True)
                    mp = {'view': 'room', **snap}
                    state = {**state, 'moves': snap['moves'], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': False, 'left_code': url_code, 'recorded_game_key': None}
                    message = snap['message']
                elif kind == 'mp-choose-join':
                    prev_code = mp.get('code') or state.get('left_code') or url_code
                    saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
                    mode = 'multiplayer'
                    page = 'home'
                    in_lobby = True
                    mp = {'view': 'join'}
                    state = {**state, 'moves': [], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': True, 'left_code': prev_code}
                    message = 'Enter the 6-character code from your friend.'
                elif kind == 'mp-back':
                    prev_code = mp.get('code') or state.get('left_code') or url_code
                    mode = 'multiplayer'
                    page = 'home'
                    in_lobby = True
                    mp = {'view': 'choice'}
                    state = {**state, 'moves': [], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': True, 'left_code': prev_code}
                    message = 'Create a game or join with a friend’s code.'
                elif kind == 'mp-join':
                    saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
                    mode = 'multiplayer'
                    page = 'home'
                    in_lobby = False
                    cleaned_code = (action.get('code', '') or '').strip().upper()
                    existing_token = mp.get('token') if mp.get('code') == cleaned_code else None
                    if not existing_token:
                        for g in saved_games:
                            if g.get('code') == cleaned_code and g.get('token'):
                                existing_token = g['token']
                                break
                    snap = hub.join_room(cleaned_code, existing_token)
                    mp = {'view': 'room', **snap}
                    state = {**state, 'moves': snap['moves'], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': False, 'left_code': url_code, 'recorded_game_key': None}
                    message = snap['message']
                elif kind == 'mp-leave':
                    prev_code = mp.get('code') or state.get('left_code') or url_code
                    saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
                    mode = 'multiplayer'
                    page = 'home'
                    in_lobby = True
                    mp = {'view': 'choice'}
                    state = {**state, 'moves': [], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': True, 'left_code': prev_code, 'recorded_game_key': None}
                    message = 'Left multiplayer room. Create or join a game.'
                elif kind == 'mp-sync':
                    if mp.get('code'):
                        snap = hub.get_room(mp['code'], mp.get('token'))
                        if snap:
                            mp = {'view': 'room', **snap}
                            state = {**state, 'moves': snap['moves'], 'selected': state.get('selected'), 'mp': mp}
                            message = snap['message']
                elif kind == 'click':
                    page = 'home'
                    in_lobby = False
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
                            state = {**state, 'moves': snap['moves'], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': False}
                            message = f'Played {san}.'
                        else:
                            state = {**state, **updated, 'mp': mp, 'page': page, 'in_lobby': False}
                            message = click_msg
                    else:
                        updated, message, _ = game.click(state, int(action['square']), promotion)
                        state = {**state, **updated, 'page': page, 'in_lobby': False}
                elif kind == 'drop':
                    page = 'home'
                    in_lobby = False
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
                        state = {**state, 'moves': snap['moves'], 'selected': None, 'mp': mp, 'page': page, 'in_lobby': False}
                        message = f'Played {san}.'
                    else:
                        updated, san = game.play(state, uci)
                        state = {**state, **updated, 'page': page, 'in_lobby': False}
                        message = f'Played {san}.'
            position = game.restore(state)
            if mode == 'ai' and position.turn == chess.BLACK and not position.is_game_over(claim_draw=True):
                move, _ = ai.choose(position)
                updated, san = game.play(state, move.uci())
                state = {**state, **updated}
                message += f' Computer played {san}.'
        except (ValueError, KeyError, TypeError) as exc:
            message = str(exc) if mode == 'multiplayer' and str(exc) else 'That move is not available. Try a highlighted square.'
            state = {**state, 'selected': None, **({'mp': mp} if mode == 'multiplayer' else {}), **({'auth': auth} if auth else {})}

        # Keep current game synced into in-progress games list
        if not (trigger == 'board-action' and (action or {}).get('type') == 'dismiss-game'):
            saved_games = _SnapshotCurrentIntoSaved(mode, state, saved_games, hub)
        else:
            saved_games = _sync_saved_games(saved_games, hub)

        if mode != input_mode:
            state['expected_mode_echo'] = mode
        elif trigger == 'mode':
            state['expected_mode_echo'] = None
        current_mp_code = (state.get('mp') or {}).get('code')
        if current_mp_code and current_mp_code not in seen_codes:
            seen_codes.append(current_mp_code)
        if url_code and url_code not in seen_codes and not is_browser_premount:
            seen_codes.append(url_code)
        in_lobby = bool(state.get('in_lobby')) or (mode == 'multiplayer' and not current_mp_code)
        any_cached = hub.get_session(sid, max_age=3600.0) if sid else None
        cached_seq = int((any_cached or {}).get('state', {}).get('seq', 0))
        server_seq = max(server_seq, int(client_seq or 0), int(state.get('seq', 0)), cached_seq) + 1
        seq = server_seq
        state = {
            **state,
            'mode': mode,
            'saved_games': saved_games,
            'seen_codes': seen_codes[-24:],
            'page': page,
            'in_lobby': in_lobby,
            'seq': seq,
            'sid': sid,
        }

        position = game.restore(state)
        ended = position.is_game_over(claim_draw=True)

        # Record completed game outcome once for registered users in AI or Multiplayer mode
        if ended and auth and auth.get('role') == 'user' and mode in ('ai', 'multiplayer'):
            game_key = f"{mode}:{(state.get('mp') or {}).get('code', 'local')}:{','.join(state.get('moves', []))}"
            if state.get('recorded_game_key') != game_key:
                outcome = position.outcome(claim_draw=True)
                if outcome:
                    if outcome.winner is None:
                        result_kind = 'draw'
                    elif mode == 'ai':
                        result_kind = 'win' if outcome.winner == chess.WHITE else 'loss'
                    else:
                        my_white = (state.get('mp') or {}).get('color') == 'white'
                        result_kind = 'win' if (outcome.winner == chess.WHITE) == my_white else 'loss'
                    updated_auth = auth_store.record_result(auth['username'], result_kind)
                    if updated_auth:
                        auth = updated_auth
                        state = {**state, 'recorded_game_key': game_key}
                        message += f' Rating updated: {auth["rating"]} ({auth["rank"]}).'

        if auth and auth.get('role') == 'user':
            fresh = auth_store.save_user_games(auth['username'], saved_games) or auth_store.get_profile(auth['username'])
            if fresh:
                auth = fresh
        if auth:
            state = {**state, 'auth': auth, 'page': page}

        now_ts = time.monotonic()
        if sid:
            session_states[sid] = {'state': state, 'ts': now_ts}
            hub.save_session(sid, state)

        allowed_color = None
        user_avatar = (auth or {}).get('avatar') or 'K'
        display_name = (auth or {}).get('display_name') or (auth['username'] if auth and auth.get('role') == 'user' else 'You')
        rank_suffix = f' · {auth["rank"]} ({auth["rating"]})' if auth and auth.get('role') == 'user' else ''
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
                self_bar = player_bar(f'{display_name} (Black)', f'Playing with the black pieces · Room {code}{rank_suffix}', 'k',
                                      position.turn == chess.BLACK and not ended and connected, avatar_symbol=user_avatar)
            else:
                opp_name = 'Friend (Black)' if connected else 'Waiting for friend…'
                opp_sub = f'Connected · Room {code}' if connected else (f'Share code {code} to connect' if code else 'Create or join a game')
                opponent_bar = player_bar(opp_name, opp_sub, 'k',
                                          position.turn == chess.BLACK and not ended and connected)
                self_bar = player_bar(f'{display_name} (White)' if code else display_name,
                                      f'Playing with the white pieces · Room {code}{rank_suffix}' if code else f'Multiplayer lobby{rank_suffix}', 'K',
                                      position.turn == chess.WHITE and not ended and connected, avatar_symbol=user_avatar)
        else:
            opponent_bar = player_bar('Computer' if mode == 'ai' else 'Black',
                                      'Your opponent · Black' if mode == 'ai' else 'Local player · Black',
                                      'k', position.turn == chess.BLACK and not ended)
            self_bar = player_bar(display_name if mode == 'ai' else 'White',
                                  f'Playing with the white pieces{rank_suffix}' if mode == 'ai' else 'Local player · White',
                                  'K', position.turn == chess.WHITE and not ended,
                                  avatar_symbol=user_avatar if mode == 'ai' else None)

        (
            panel_style,
            tabs_row_style,
            tab_lobby_style,
            tab_find_style,
            choice_style,
            join_style,
            find_style,
            find_children,
            room_style,
            room_children,
            leave_style,
            poll_disabled,
        ) = multiplayer_styles(mode, state.get('mp'), hub=hub, saved_games=saved_games)
        if page != 'home':
            poll_disabled = True
        active_code = (state.get('mp') or {}).get('code') if (mode == 'multiplayer' and page == 'home' and not in_lobby) else None
        target_search = f'?code={active_code}' if active_code else ''
        search_out = no_update if is_browser_premount else (target_search if (url_search or '') != target_search else no_update)

        is_authenticated = bool(auth)
        auth_screen_style = {'display': 'none'} if is_authenticated else {'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'minHeight': '65vh'}
        home_screen_style = {'display': 'block'} if (is_authenticated and page == 'home') else {'display': 'none'}
        leaderboard_screen_style = {'display': 'block', 'minHeight': '65vh'} if (is_authenticated and page == 'leaderboard') else {'display': 'none'}
        profile_screen_style = {'display': 'block', 'minHeight': '65vh'} if (is_authenticated and page == 'profile') else {'display': 'none'}
        leaderboard_children = render_leaderboard_content(auth_store.leaderboard(), auth) if (is_authenticated and page == 'leaderboard') else []

        if auth and auth.get('role') == 'user':
            badge_text = f'{display_name} · {auth["rank"]} ({auth["rating"]})'
        elif is_authenticated:
            badge_text = f'Playing as {display_name}'
        else:
            badge_text = 'Welcome'

        btn_base = {'padding': '6px 10px', 'fontSize': '11px'}
        user_menu_style = {'position': 'relative', 'display': 'inline-block'} if is_authenticated else {'display': 'none'}
        nav_play_style = {**btn_base, 'display': 'inline-block'} if (is_authenticated and (page != 'home' or not in_lobby)) else {'display': 'none'}
        nav_lb_style = {**btn_base, 'display': 'inline-block'} if is_authenticated else {'display': 'none'}

        avatar_img = piece_image(user_avatar)
        prof_name_text = display_name if is_authenticated else 'Guest'
        prof_sub_text = (f'@{auth["username"]} · {auth["rank"]} · {auth["rating"]} pts'
                         if auth and auth.get('role') == 'user' else 'Guest session')
        lichess_box = render_external_account_box('Lichess', (auth or {}).get('lichess'))
        chesscom_box = render_external_account_box('Chess.com', (auth or {}).get('chesscom'))

        active_game_id = _current_game_id(mode, state)
        active_games_children = render_active_games(saved_games, active_game_id)
        active_games_count = f'{len(saved_games)} active'

        if in_lobby:
            play_area_style = {'display': 'none'}
            game_layout_style = {'display': 'block', 'maxWidth': '640px', 'margin': '0 auto'}
            board_controls_style = {'display': 'none'}
            history_panel_style = {'display': 'none'}
            options_panel_style = {'display': 'none'}
        else:
            play_area_style = {'display': 'block'}
            game_layout_style = {}
            board_controls_style = {'display': 'block'}
            history_panel_style = {'display': 'block'}
            options_panel_style = {'display': 'block'}

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
            tabs_row_style,
            tab_lobby_style,
            tab_find_style,
            choice_style,
            join_style,
            find_style,
            find_children,
            room_style,
            room_children,
            leave_style,
            poll_disabled,
            mode,
            search_out,
            auth_screen_style,
            home_screen_style,
            auth_feedback,
            badge_text,
            user_menu_style,
            leaderboard_screen_style,
            leaderboard_children,
            nav_play_style,
            nav_lb_style,
            profile_screen_style,
            profile_feedback,
            avatar_img,
            avatar_img,
            prof_name_text,
            prof_sub_text,
            display_name if is_authenticated else '',
            user_avatar,
            lichess_box,
            chesscom_box,
            active_games_children,
            active_games_count,
            play_area_style,
            game_layout_style,
            board_controls_style,
            history_panel_style,
            options_panel_style,
        )
    return app
