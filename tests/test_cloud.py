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
