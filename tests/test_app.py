from plotly_chess.app import create_app


def _make_inputs(mode='local', action=None, search='', pathname='/'):
    return [
        {'id': 'board-action', 'property': 'data', 'value': action},
        {'id': 'play', 'property': 'n_clicks', 'value': 1},
        {'id': 'reset', 'property': 'n_clicks', 'value': 0},
        {'id': 'hint', 'property': 'n_clicks', 'value': 0},
        {'id': 'mode', 'property': 'value', 'value': mode},
        {'id': 'mp-host-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'mp-create-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'mp-choose-join-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'mp-tab-lobby-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'mp-tab-find-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'mp-join-submit', 'property': 'n_clicks', 'value': 0},
        {'id': 'join-code-input', 'property': 'n_submit', 'value': 0},
        {'id': 'mp-back-choice', 'property': 'n_clicks', 'value': 0},
        {'id': 'mp-leave-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'mp-poll', 'property': 'n_intervals', 'value': 0},
        {'id': 'url', 'property': 'search', 'value': search},
        {'id': 'url', 'property': 'pathname', 'value': pathname},
        {'id': 'auth-login-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'auth-register-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'auth-guest-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'auth-logout-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'auth-username', 'property': 'n_submit', 'value': 0},
        {'id': 'auth-password', 'property': 'n_submit', 'value': 0},
        {'id': 'nav-leaderboard-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'nav-play-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'leaderboard-back-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'nav-profile-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'profile-back-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'profile-save-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'profile-password-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'profile-lichess-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'profile-chesscom-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'brand-home-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'nav-about-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'footer-about-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'menu-about-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'about-back-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'about-tab-tech-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'about-tab-why-btn', 'property': 'n_clicks', 'value': 0},
    ]


def _make_state(
    state,
    notation='e4',
    join_code='',
    username='',
    password='',
    display_name='Alice',
    avatar='K',
    curr_pw='',
    new_pw='',
    lichess_user='',
    chesscom_user='',
):
    return [
        {'id': 'notation', 'property': 'value', 'value': notation},
        {'id': 'promotion', 'property': 'value', 'value': 'q'},
        {'id': 'game', 'property': 'data', 'value': state},
        {'id': 'join-code-input', 'property': 'value', 'value': join_code},
        {'id': 'auth-username', 'property': 'value', 'value': username},
        {'id': 'auth-password', 'property': 'value', 'value': password},
        {'id': 'profile-display-name', 'property': 'value', 'value': display_name},
        {'id': 'profile-avatar-select', 'property': 'value', 'value': avatar},
        {'id': 'profile-current-password', 'property': 'value', 'value': curr_pw},
        {'id': 'profile-new-password', 'property': 'value', 'value': new_pw},
        {'id': 'profile-lichess-input', 'property': 'value', 'value': lichess_user},
        {'id': 'profile-chesscom-input', 'property': 'value', 'value': chesscom_user},
    ]


def test_app_layout_and_callback_registration():
    app = create_app()
    client = app.server.test_client()
    assert client.get('/').status_code == 200
    res_layout = client.get('/_dash-layout')
    assert res_layout.status_code == 200
    assert 'burger-menu-btn' in str(res_layout.get_json())
    assert 'site-nav' in str(res_layout.get_json())
    assert client.get('/_dash-dependencies').status_code == 200
    assert len(app.callback_map) == 1


def test_auth_profile_external_links_and_leaderboard(tmp_path):
    from plotly_chess.auth import AuthStore

    def fake_http_get(url):
        if 'lichess.org/api/user/' in url:
            return {
                'username': 'AliceLichess',
                'perfs': {'rapid': {'rating': 1680}},
                'count': {'win': 140, 'loss': 95, 'draw': 18},
            }
        if 'api.chess.com/pub/player/' in url:
            return {
                'chess_rapid': {'last': {'rating': 1520}, 'record': {'win': 88, 'loss': 60, 'draw': 9}},
            }
        raise ValueError(f'Unexpected URL {url}')

    auth_store = AuthStore(db_path=tmp_path / 'users.sqlite3', http_get=fake_http_get)
    app = create_app(auth_store=auth_store)
    client = app.server.test_client()
    output = next(iter(app.callback_map))

    def call_auth(state, trigger, username='', password='', mode='ai', **kwargs):
        response = client.post('/_dash-update-component', json={
            'output': output,
            'outputs': [{'id': o.component_id, 'property': o.component_property} for o in app.callback_map[output]['output']],
            'inputs': _make_inputs(mode),
            'state': _make_state(state, username=username, password=password, **kwargs),
            'changedPropIds': [trigger] if trigger else [],
        })
        assert response.status_code == 200
        return response.get_json()['response']

    # 1. Before logging in, auth-screen is visible and home-screen is hidden
    initial = call_auth({'moves': [], 'selected': None}, '')
    assert initial['auth-screen']['style']['display'] == 'flex'
    assert initial['home-screen']['style']['display'] == 'none'

    # 2. Play as guest directs to home screen
    guest = call_auth({'moves': [], 'selected': None}, 'auth-guest-btn.n_clicks')
    assert guest['auth-screen']['style']['display'] == 'none'
    assert guest['home-screen']['style']['display'] == 'block'
    assert guest['user-badge']['children'] == 'Playing as Guest'

    # 3. Switch account / logout returns to auth screen
    logged_out = call_auth(guest['game']['data'], 'auth-logout-btn.n_clicks')
    assert logged_out['auth-screen']['style']['display'] == 'flex'
    assert logged_out['home-screen']['style']['display'] == 'none'

    # 4. Register a new account directs to home screen with initial rank and rating
    registered = call_auth({'moves': [], 'selected': None}, 'auth-register-btn.n_clicks', username='Alice', password='secret')
    assert registered['auth-screen']['style']['display'] == 'none'
    assert registered['home-screen']['style']['display'] == 'block'
    assert 'Alice · Contender (1000)' in registered['user-badge']['children']

    # 5. Open My Profile page via avatar dropdown menu, update display name and avatar
    prof_view = call_auth(registered['game']['data'], 'nav-profile-btn.n_clicks')
    assert prof_view['profile-screen']['style']['display'] == 'block'
    assert prof_view['home-screen']['style']['display'] == 'none'

    saved_prof = call_auth(prof_view['game']['data'], 'profile-save-btn.n_clicks', display_name='Queen Alice', avatar='Q')
    assert saved_prof['game']['data']['auth']['display_name'] == 'Queen Alice'
    assert saved_prof['game']['data']['auth']['avatar'] == 'Q'
    assert 'Queen Alice · Contender (1000)' in saved_prof['user-badge']['children']

    # 6. Change password on My Profile page
    pw_updated = call_auth(saved_prof['game']['data'], 'profile-password-btn.n_clicks', curr_pw='secret', new_pw='newsecret')
    assert 'Password updated' in pw_updated['profile-feedback']['children']
    assert auth_store.login('Alice', 'newsecret')['display_name'] == 'Queen Alice'

    # 7. Link Lichess and Chess.com accounts on My Profile page and verify DB persistence
    linked_lichess = call_auth(pw_updated['game']['data'], 'profile-lichess-btn.n_clicks', lichess_user='AliceLichess')
    assert linked_lichess['game']['data']['auth']['lichess']['rating'] == 1680
    assert linked_lichess['game']['data']['auth']['lichess']['wins'] == 140

    linked_chesscom = call_auth(linked_lichess['game']['data'], 'profile-chesscom-btn.n_clicks', chesscom_user='AliceCC')
    assert linked_chesscom['game']['data']['auth']['chesscom']['rating'] == 1520
    assert linked_chesscom['game']['data']['auth']['chesscom']['wins'] == 88

    # 8. Record game results, verify rank progression and linked stats on Leaderboard page
    auth_store.register('Bob', 'secret')
    auth_store.record_result('Alice', 'win')
    auth_store.record_result('Alice', 'win')
    auth_store.record_result('Bob', 'loss')
    board_entries = auth_store.leaderboard()
    assert board_entries[0]['username'] == 'Alice'
    assert board_entries[0]['display_name'] == 'Queen Alice'
    assert board_entries[0]['avatar'] == 'Q'
    assert board_entries[0]['wins'] == 2
    assert board_entries[0]['rating'] == 1050
    assert board_entries[0]['rank'] == 'Club Player'
    assert board_entries[0]['lichess']['rating'] == 1680
    assert board_entries[0]['chesscom']['rating'] == 1520

    # Navigate to Leaderboard screen and back to board
    lb_view = call_auth(linked_chesscom['game']['data'], 'nav-leaderboard-btn.n_clicks')
    assert lb_view['leaderboard-screen']['style']['display'] == 'block'
    assert lb_view['home-screen']['style']['display'] == 'none'

    back_view = call_auth(lb_view['game']['data'], 'leaderboard-back-btn.n_clicks')
    assert back_view['leaderboard-screen']['style']['display'] == 'none'
    assert back_view['home-screen']['style']['display'] == 'block'

    # 9. Clicking Profile then Home logo (even with stale State('game', 'data') echo on mode.value)
    # hides Profile, hides the game board (play-area), and shows the Home lobby screen
    prof_again = call_auth(back_view['game']['data'], 'nav-profile-btn.n_clicks', mode='ai')
    assert prof_again['profile-screen']['style']['display'] == 'block'
    home_from_prof = call_auth(prof_again['game']['data'], 'brand-home-btn.n_clicks', mode='ai')
    assert home_from_prof['profile-screen']['style']['display'] == 'none'
    assert home_from_prof['home-screen']['style']['display'] == 'block'
    assert home_from_prof['play-area']['style']['display'] == 'none'
    # Simulate Dash browser echo on mode.value before sessionStorage flushed prof_again state
    echo_after_home = call_auth(prof_again['game']['data'], 'mode.value', mode='multiplayer')
    assert echo_after_home['profile-screen']['style']['display'] == 'none'
    assert echo_after_home['home-screen']['style']['display'] == 'block'
    assert echo_after_home['play-area']['style']['display'] == 'none'


def test_callback_move_and_reset():
    app = create_app()
    client = app.server.test_client()
    output = next(iter(app.callback_map))
    inputs = _make_inputs('local')

    def request(state, trigger):
        response = client.post('/_dash-update-component', json={
            'output': output,
            'outputs': [{'id': o.component_id, 'property': o.component_property} for o in app.callback_map[output]['output']],
            'inputs': inputs,
            'state': _make_state(state),
            'changedPropIds': [trigger],
        })
        assert response.status_code == 200
        return response.get_json()['response']

    moved = request({'moves': [], 'selected': None}, 'play.n_clicks')
    assert moved['game']['data']['moves'] == ['e2e4']
    reset = request(moved['game']['data'], 'reset.n_clicks')
    assert reset['game']['data']['moves'] == []

    # Drag-and-drop is validated on the server independently of client hints.
    inputs[0]['value'] = {'type': 'drop', 'from': 12, 'to': 28}
    dropped = request({'moves': [], 'selected': None}, 'board-action.data')
    assert dropped['game']['data']['moves'] == ['e2e4']
    inputs[0]['value'] = {'type': 'drop', 'from': 12, 'to': 36}
    invalid = request({'moves': [], 'selected': None}, 'board-action.data')
    assert invalid['game']['data']['moves'] == []
    assert 'not available' in invalid['feedback']['children']
    inputs[0]['value'] = {'type': 'click', 'square': 12}
    selected = request({'moves': [], 'selected': None}, 'board-action.data')
    assert selected['game']['data']['selected'] == 12
    inputs[0]['value'] = {'type': 'click', 'square': 28}
    clicked = request(selected['game']['data'], 'board-action.data')
    assert clicked['game']['data']['moves'] == ['e2e4']


def test_multiplayer_flow_and_websocket_sync():
    import socket
    import threading
    from plotly_chess.multiplayer import MultiplayerHub, WebSocketClient

    hub = MultiplayerHub()
    app = create_app(hub=hub)
    client = app.server.test_client()
    output = next(iter(app.callback_map))

    def call_mp(state, trigger, action=None, notation='e4', join_code='', mode='multiplayer', search='', pathname='/'):
        inputs = _make_inputs(mode, action, search=search, pathname=pathname)
        response = client.post('/_dash-update-component', json={
            'output': output,
            'outputs': [{'id': o.component_id, 'property': o.component_property} for o in app.callback_map[output]['output']],
            'inputs': inputs,
            'state': _make_state(state, notation=notation, join_code=join_code),
            'changedPropIds': [trigger] if trigger else [],
        })
        assert response.status_code == 200
        return response.get_json()['response']

    # 1. Selecting Multiplayer shows Create game / Join game lobby via inline style
    lobby = call_mp({'moves': [], 'selected': None}, 'mode.value')
    assert lobby['multiplayer-panel']['style']['display'] == 'block'
    assert lobby['mp-view-choice']['style']['display'] == 'block'
    assert lobby['game']['data']['mp']['view'] == 'choice'

    # 2. Clicking Create game button (native Dash n_clicks) creates room and shows 6-character code
    created = call_mp(lobby['game']['data'], 'mp-create-btn.n_clicks')
    host_state = created['game']['data']
    code = host_state['mp']['code']
    assert len(code) == 6
    assert host_state['mp']['color'] == 'white'
    assert host_state['mp']['connected'] is False
    assert created['mp-view-room']['style']['display'] == 'block'
    assert created['mp-poll']['disabled'] is False
    assert created['url']['search'] == f'?code={code}'

    # Host cannot move before friend connects
    early = call_mp(host_state, 'board-action.data', {'type': 'drop', 'from': 12, 'to': 28})
    assert early['game']['data']['moves'] == []
    assert 'Waiting for your friend' in early['feedback']['children']

    # 3. Connect a live WebSocket subscriber for the room to verify real-time push
    server_sock, client_sock = socket.socketpair()
    ws_reader = WebSocketClient(client_sock)
    events = []

    def run_ws():
        environ = {
            'PATH_INFO': '/ws',
            'QUERY_STRING': f'code={code}',
            'HTTP_UPGRADE': 'websocket',
            'HTTP_SEC_WEBSOCKET_KEY': 'dGhlIHNhbXBsZSBub25jZQ==',
            'werkzeug.socket': server_sock,
        }
        try:
            app.server.wsgi_app(environ, lambda *_: None)
        except ConnectionResetError:
            pass

    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()
    handshake = b''
    while b'\r\n\r\n' not in handshake:
        handshake += client_sock.recv(256)
    assert b'101 Switching Protocols' in handshake
    events.append(ws_reader.receive_json())
    assert events[-1]['code'] == code and events[-1]['connected'] is False

    # 4. Friend opens invite link (?code=...) directly and automatically joins in multiplayer mode
    joined = call_mp({'moves': [], 'selected': None}, 'url.search', mode='ai', search=f'?code={code}')
    assert joined['mode']['value'] == 'multiplayer'
    guest_state = joined['game']['data']
    assert guest_state['mp']['color'] == 'black'
    assert guest_state['mp']['connected'] is True

    # Host WebSocket receives immediate connection broadcast
    events.append(ws_reader.receive_json())
    assert events[-1]['connected'] is True

    # 5. Connected players take turns on the board (and mp-poll also syncs host)
    synced_host = call_mp(host_state, 'mp-poll.n_intervals')['game']['data']
    assert synced_host['mp']['connected'] is True

    # Black cannot move on White's turn
    wrong_turn = call_mp(guest_state, 'board-action.data', {'type': 'drop', 'from': 52, 'to': 36})
    assert 'Wait for White' in wrong_turn['feedback']['children']

    # White plays e2e4
    white_moved = call_mp(synced_host, 'board-action.data', {'type': 'drop', 'from': 12, 'to': 28})
    assert white_moved['game']['data']['moves'] == ['e2e4']
    events.append(ws_reader.receive_json())
    assert events[-1]['moves'] == ['e2e4']

    # Black syncs and replies e7e5
    guest_synced = call_mp(guest_state, 'mp-poll.n_intervals')['game']['data']
    assert guest_synced['moves'] == ['e2e4']
    black_moved = call_mp(guest_synced, 'board-action.data', {'type': 'drop', 'from': 52, 'to': 36})
    assert black_moved['game']['data']['moves'] == ['e2e4', 'e7e5']

    # 6. Clicking the top-left home logo (brand-home-btn) returns to the game join screen
    # while keeping the in-progress room tracked in saved_games so the player can dive back in
    home_lobby = call_mp(black_moved['game']['data'], 'brand-home-btn.n_clicks', search=f'?code={code}')
    assert home_lobby['mode']['value'] == 'multiplayer'
    assert home_lobby['game']['data']['mp']['view'] == 'choice'
    assert home_lobby['mp-view-choice']['style']['display'] == 'block'
    assert home_lobby['play-area']['style']['display'] == 'none'
    assert home_lobby['url']['search'] == ''
    assert home_lobby['game']['data']['seq'] > black_moved['game']['data']['seq']
    saved = home_lobby['game']['data']['saved_games']
    assert any(g['id'] == f'mp:{code}' for g in saved)

    # Simulate Dash browser follow-up url.search='' and mp-sync callbacks carrying stale pre-Home state
    url_echo = call_mp(black_moved['game']['data'], 'url.search', search='')
    assert url_echo['play-area']['style']['display'] == 'none'
    assert url_echo['game']['data']['in_lobby'] is True
    assert url_echo['game']['data']['mp']['view'] == 'choice'

    sync_echo = call_mp(black_moved['game']['data'], 'board-action.data', {'type': 'mp-sync'})
    assert sync_echo['play-area']['style']['display'] == 'none'
    assert sync_echo['game']['data']['in_lobby'] is True

    # Start a second local game and play a move, then click Home logo and verify it exits the game board
    local_game = call_mp(home_lobby['game']['data'], 'mode.value', mode='local')
    local_moved = call_mp(local_game['game']['data'], 'board-action.data', {'type': 'drop', 'from': 6, 'to': 21}, mode='local')
    assert local_moved['game']['data']['moves'] == ['g1f3']
    assert local_moved['play-area']['style']['display'] == 'block'
    assert len(local_moved['game']['data']['saved_games']) == 2

    home_from_local = call_mp(local_moved['game']['data'], 'brand-home-btn.n_clicks', mode='local')
    assert home_from_local['play-area']['style']['display'] == 'none'
    assert home_from_local['game']['data']['in_lobby'] is True
    mode_echo_from_local = call_mp(local_moved['game']['data'], 'mode.value', mode='multiplayer')
    assert mode_echo_from_local['play-area']['style']['display'] == 'none'
    assert mode_echo_from_local['game']['data']['in_lobby'] is True

    resumed_mp = call_mp(
        mode_echo_from_local['game']['data'],
        'board-action.data',
        {'type': 'resume-game', 'game_id': f'mp:{code}'},
        mode='multiplayer',
    )
    assert resumed_mp['mode']['value'] == 'multiplayer'
    assert resumed_mp['game']['data']['mp']['code'] == code
    assert resumed_mp['game']['data']['moves'] == ['e2e4', 'e7e5']

    # Clicking Profile or Leaderboard while inside a multiplayer game exits the room, clears ?code=...,
    # disables polling, and keeps Profile/Leaderboard open even if url.search='' echoes with stale state
    authed_room_state = {**resumed_mp['game']['data'], 'auth': {'username': 'Guest', 'display_name': 'Guest', 'role': 'guest'}}
    prof_from_mp = call_mp(authed_room_state, 'nav-profile-btn.n_clicks', search=f'?code={code}')
    assert prof_from_mp['profile-screen']['style']['display'] == 'block'
    assert prof_from_mp['home-screen']['style']['display'] == 'none'
    assert prof_from_mp['url']['search'] == ''
    assert prof_from_mp['mp-poll']['disabled'] is True
    url_echo_after_prof = call_mp(authed_room_state, 'url.search', search='')
    assert url_echo_after_prof['profile-screen']['style']['display'] == 'block'
    assert url_echo_after_prof['home-screen']['style']['display'] == 'none'

    # Removing ?code=... from the URL bar and hitting Return (initial load trigger='') does not redirect back to the game
    manual_url_clear = call_mp(authed_room_state, '', search='')
    assert manual_url_clear['home-screen']['style']['display'] == 'block'
    assert manual_url_clear['play-area']['style']['display'] == 'none'
    assert manual_url_clear['game']['data']['in_lobby'] is True
    assert 'url' not in manual_url_clear or manual_url_clear['url']['search'] == ''

    # 7. Host a new open game via mp-host-btn, then another player browses "Find a game" tab and joins it
    hosted = call_mp(home_lobby['game']['data'], 'mp-host-btn.n_clicks')
    hosted_code = hosted['game']['data']['mp']['code']
    assert any(r['code'] == hosted_code for r in hub.list_open_rooms())

    find_tab = call_mp({'moves': [], 'selected': None}, 'mp-tab-find-btn.n_clicks')
    assert find_tab['mp-view-find']['style']['display'] == 'block'
    assert find_tab['mp-poll']['disabled'] is False

    joined_open = call_mp(find_tab['game']['data'], 'board-action.data', {'type': 'mp-join', 'code': hosted_code})
    assert joined_open['game']['data']['mp']['code'] == hosted_code
    assert joined_open['game']['data']['mp']['color'] == 'black'
    assert joined_open['game']['data']['mp']['connected'] is True
    assert not any(r['code'] == hosted_code for r in hub.list_open_rooms())

    client_sock.close()
    ws_thread.join(timeout=2)


def test_about_and_why_plotly_screen():
    app = create_app()
    client = app.server.test_client()
    output = next(iter(app.callback_map))

    def call_app(state, trigger):
        response = client.post('/_dash-update-component', json={
            'output': output,
            'outputs': [{'id': o.component_id, 'property': o.component_property} for o in app.callback_map[output]['output']],
            'inputs': _make_inputs(),
            'state': _make_state(state),
            'changedPropIds': [trigger] if trigger else [],
        })
        assert response.status_code == 200
        return response.get_json()['response']

    # 1. Clicking footer "Why Plotly & Architecture" link opens the About screen
    about_res = call_app({'moves': [], 'selected': None}, 'footer-about-btn.n_clicks')
    assert about_res['about-screen']['style']['display'] == 'block'
    assert about_res['home-screen']['style']['display'] == 'none'
    assert about_res['auth-screen']['style']['display'] == 'none'
    assert about_res['about-tab-tech-btn']['className'] == 'button primary'
    assert about_res['about-tab-why-btn']['className'] == 'button quiet'

    # Verify author profile and links in the tech architecture tab
    content_str = str(about_res['about-content']['children'])
    assert 'Robert McMenemy' in content_str
    assert 'https://www.linkedin.com/in/robert-mcmenemy' in content_str
    assert 'https://github.com/arkay92' in content_str
    assert 'https://medium.com/@rabmcmenemy' in content_str
    assert 'scholar.google' in content_str
    assert 'Full-Stack Reactive Architecture' in content_str

    # 2. Switching to the Why Plotly Enterprise tab
    why_res = call_app(about_res['game']['data'], 'about-tab-why-btn.n_clicks')
    assert why_res['about-tab-tech-btn']['className'] == 'button quiet'
    assert why_res['about-tab-why-btn']['className'] == 'button primary'
    why_str = str(why_res['about-content']['children'])
    assert 'Why Plotly: A Production Platform IT Can Govern' in why_str
    assert 'SOC 2 Type II' in why_str
    assert 'ISO 27001' in why_str
    assert 'Dashboards vs. DIY vs. A Governed App Platform' in why_str
    assert 'dash.plotly.com/dash-enterprise' in why_str

    # 3. Switching back to the Tech Architecture tab
    tech_res = call_app(why_res['game']['data'], 'about-tab-tech-btn.n_clicks')
    tech_str = str(tech_res['about-content']['children'])
    assert 'Robert McMenemy' in tech_str

    # 4. Returning to the game
    back_res = call_app(tech_res['game']['data'], 'about-back-btn.n_clicks')
    assert back_res['about-screen']['style']['display'] == 'none'
    assert back_res['game']['data']['page'] == 'home'


def test_friends_system_and_online_presence(tmp_path):
    from plotly_chess.auth import AuthStore
    import time

    store = AuthStore(db_path=tmp_path / 'friends_test.sqlite3')
    u1 = store.register('alice', 'password123')
    u2 = store.register('bob', 'password456')

    # Initially not friends
    assert not store.is_friend('alice', 'bob')
    assert store.get_friends('alice') == []

    # Add friend
    assert store.add_friend('alice', 'bob') is True
    assert store.is_friend('alice', 'bob') is True
    assert not store.is_friend('bob', 'alice')  # Directed friendship

    # Online status detection
    assert not store.is_online('bob')
    store.touch_user_activity('bob')
    assert store.is_online('bob') is True

    # Retrieve friends with online status
    friends = store.get_friends('alice')
    assert len(friends) == 1
    assert friends[0]['username'] == 'bob'
    assert friends[0]['is_online'] is True

    # Remove friend
    assert store.remove_friend('alice', 'bob') is True
    assert not store.is_friend('alice', 'bob')
    assert store.get_friends('alice') == []


def test_multiplayer_completed_games_feed(tmp_path):
    from plotly_chess.multiplayer import MultiplayerHub

    hub = MultiplayerHub(db_path=tmp_path / 'mp_test.sqlite3')
    # Fresh hub returns empty list (no dummy data)
    default_games = hub.get_recent_completed_games()
    assert default_games == []

    # Record a completed game
    entry = hub.record_completed_game(
        code='TEST01',
        white_name='Alice',
        white_user='alice',
        white_avatar='Q',
        black_name='Bob',
        black_user='bob',
        black_avatar='k',
        result='1-0',
        result_desc='White won by checkmate',
        moves_count=42,
    )
    assert entry['code'] == 'TEST01'

    # Retrieve recent games
    recent = hub.get_recent_completed_games(limit=5)
    assert recent[0]['code'] == 'TEST01'
    assert recent[0]['white_user'] == 'alice'
    assert recent[0]['black_user'] == 'bob'
    assert recent[0]['result'] == '1-0'
    assert recent[0]['moves_count'] == 42


def test_profile_view_other_user_and_toggle_friend_action(tmp_path):
    from plotly_chess.auth import AuthStore

    store = AuthStore(db_path=tmp_path / 'app_friends.sqlite3')
    store.register('alice', 'password123')
    store.register('bob', 'password456')

    app = create_app(auth_store=store)
    client = app.server.test_client()
    output = next(iter(app.callback_map))

    def call_action(state, action, mode='ai'):
        res = client.post('/_dash-update-component', json={
            'output': output,
            'outputs': [{'id': o.component_id, 'property': o.component_property} for o in app.callback_map[output]['output']],
            'inputs': _make_inputs(mode, action=action),
            'state': _make_state(state),
            'changedPropIds': ['board-action.data'] if action else [],
        })
        assert res.status_code == 200
        return res.get_json()['response']

    # Log in Alice
    alice_login = client.post('/_dash-update-component', json={
        'output': output,
        'outputs': [{'id': o.component_id, 'property': o.component_property} for o in app.callback_map[output]['output']],
        'inputs': _make_inputs('ai'),
        'state': _make_state({'moves': [], 'selected': None}, username='alice', password='password123'),
        'changedPropIds': ['auth-login-btn.n_clicks'],
    }).get_json()['response']
    assert alice_login['user-badge']['children'].startswith('alice')

    # View Bob's profile via view-profile action (as happens when clicking on feed or leaderboard)
    view_bob = call_action(alice_login['game']['data'], {'type': 'view-profile', 'username': 'bob'})
    assert view_bob['profile-screen']['style']['display'] == 'block'
    assert 'bob' in view_bob['profile-header-name']['children'].lower()
    # Settings controls are hidden when viewing another user
    assert view_bob['profile-settings-section']['style']['display'] == 'none'
    # Friend action button is present
    assert '➕ Add Friend' in str(view_bob['profile-friend-action-container']['children'])

    # Toggle friend action adds Bob as friend
    add_bob = call_action(view_bob['game']['data'], {'type': 'toggle-friend', 'username': 'bob'})
    assert 'Added @bob as a friend' in add_bob['profile-feedback']['children']
    assert store.is_friend('alice', 'bob') is True

    # View Bob's profile again - button now reflects friend status
    view_bob_again = call_action(add_bob['game']['data'], {'type': 'view-profile', 'username': 'bob'})
    assert '✓ Friends' in str(view_bob_again['profile-friend-action-container']['children'])


