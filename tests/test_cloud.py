from dash import Dash


def test_cloud_entry_point_has_configured_app():
    import app as entry
    import main
    assert isinstance(entry.app,Dash)
    assert main.app is entry.app
    assert main.server is entry.server
    client = entry.server.test_client()
    assert client.get('/_dash-layout').status_code == 200
    assert client.get('/assets/board.js').status_code == 200
    assert client.get('/assets/styles.css').status_code == 200
    assert len(entry.app.callback_map) == 1


def test_cloud_multi_worker_shared_hub_and_poll_fallback(tmp_path):
    from plotly_chess.app import DEFAULT_HUB
    from plotly_chess.multiplayer import MultiplayerHub
    import app as entry

    # Verify two independent worker instances sharing a SQLite path see the same room
    db_file = tmp_path / 'rooms.sqlite3'
    worker_a = MultiplayerHub(db_path=db_file)
    worker_b = MultiplayerHub(db_path=db_file)
    created = worker_a.create_room()
    joined = worker_b.join_room(created['code'])
    assert joined['connected'] is True
    assert worker_a.get_room(created['code'])['connected'] is True

    # Verify HTTP poll fallback on /ws for cloud proxies that strip WebSocket upgrades
    room = DEFAULT_HUB.create_room()
    client = entry.server.test_client()
    res = client.get(f'/ws?code={room["code"]}&poll=1')
    assert res.status_code == 200
    payload = res.get_json()
    assert payload['code'] == room['code']
    assert payload['version'] == 1

    # Verify Gunicorn sync workers (wsgi.multithread=False) return 204 immediately instead of blocking
    statuses = []
    body = list(entry.server.wsgi_app({
        'PATH_INFO': '/ws',
        'QUERY_STRING': f'code={room["code"]}',
        'HTTP_UPGRADE': 'websocket',
        'HTTP_SEC_WEBSOCKET_KEY': 'dGhlIHNhbXBsZSBub25jZQ==',
        'SERVER_SOFTWARE': 'gunicorn/23.0.0',
        'wsgi.multithread': False,
        'gunicorn.socket': object(),
    }, lambda status, headers: statuses.append(status)))
    assert statuses == ['204 No Content']
    assert body == [b'']
