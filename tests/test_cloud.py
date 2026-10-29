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


def test_cloud_persistent_filesystem_detection(monkeypatch, tmp_path):
    from plotly_chess.storage import get_cloud_db_paths, get_cloud_storage_dir

    # 1. Test GAMES_DIR env var
    games_mount = tmp_path / "Games"
    games_mount.mkdir()
    monkeypatch.setenv("GAMES_DIR", str(games_mount))
    assert get_cloud_storage_dir("Games") == games_mount.resolve()
    rooms_p, users_p = get_cloud_db_paths("Games")
    assert rooms_p == games_mount.resolve() / "plotly_chess_rooms.sqlite3"
    assert users_p == games_mount.resolve() / "plotly_chess_users.sqlite3"

    # 2. Test DASH_DATA_DIR env var fallback
    monkeypatch.delenv("GAMES_DIR", raising=False)
    data_mount = tmp_path / "data"
    data_mount.mkdir()
    monkeypatch.setenv("DASH_DATA_DIR", str(data_mount))
    assert get_cloud_storage_dir("Games") == data_mount.resolve()


def test_cloud_sqlite_journal_mode_delete_and_persistence(tmp_path):
    from plotly_chess.auth import AuthStore
    from plotly_chess.multiplayer import MultiplayerHub

    db_hub = tmp_path / "hub.sqlite3"
    db_auth = tmp_path / "auth.sqlite3"

    hub = MultiplayerHub(db_path=db_hub)
    auth = AuthStore(db_path=db_auth)

    with hub._connect_db() as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "delete"

    with auth._connect_db() as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "delete"


def test_cloud_shared_stores_adopt_cloud_storage(monkeypatch, tmp_path):
    from plotly_chess.auth import AuthStore
    from plotly_chess.multiplayer import MultiplayerHub

    games_mount = tmp_path / "Games"
    games_mount.mkdir()
    monkeypatch.setenv("GAMES_DIR", str(games_mount))

    hub = MultiplayerHub(shared=True)
    auth = AuthStore(shared=True)

    assert hub._db_path == games_mount.resolve() / "plotly_chess_rooms.sqlite3"
    assert auth._db_path == games_mount.resolve() / "plotly_chess_users.sqlite3"


