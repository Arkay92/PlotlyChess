from plotly_chess.app import create_app


def _make_inputs(mode='local', action=None, search='', pathname='/'):
    return [
        {'id': 'board-action', 'property': 'data', 'value': action},
        {'id': 'play', 'property': 'n_clicks', 'value': 1},
        {'id': 'reset', 'property': 'n_clicks', 'value': 0},
        {'id': 'hint', 'property': 'n_clicks', 'value': 0},
        {'id': 'mode', 'property': 'value', 'value': mode},
        {'id': 'mp-create-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'mp-choose-join-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'mp-join-submit', 'property': 'n_clicks', 'value': 0},
        {'id': 'join-code-input', 'property': 'n_submit', 'value': 0},
        {'id': 'mp-back-choice', 'property': 'n_clicks', 'value': 0},
        {'id': 'mp-leave-btn', 'property': 'n_clicks', 'value': 0},
        {'id': 'mp-poll', 'property': 'n_intervals', 'value': 0},
        {'id': 'url', 'property': 'search', 'value': search},
        {'id': 'url', 'property': 'pathname', 'value': pathname},
    ]


def test_app_layout_and_callback_registration():
    app = create_app()
    client = app.server.test_client()
    assert client.get('/').status_code == 200
    assert client.get('/_dash-layout').status_code == 200
    assert client.get('/_dash-dependencies').status_code == 200
    assert len(app.callback_map) == 1


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
            'state': [
                {'id': 'notation', 'property': 'value', 'value': 'e4'},
                {'id': 'promotion', 'property': 'value', 'value': 'q'},
                {'id': 'game', 'property': 'data', 'value': state},
                {'id': 'join-code-input', 'property': 'value', 'value': ''},
            ],
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
            'state': [
                {'id': 'notation', 'property': 'value', 'value': notation},
                {'id': 'promotion', 'property': 'value', 'value': 'q'},
                {'id': 'game', 'property': 'data', 'value': state},
                {'id': 'join-code-input', 'property': 'value', 'value': join_code},
            ],
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

    client_sock.close()
    ws_thread.join(timeout=2)
