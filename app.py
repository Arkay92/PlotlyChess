"""Plotly Cloud entry point with an explicit, discoverable Dash instance."""
from pathlib import Path

from dash import Dash

from plotly_chess.app import create_app
from plotly_chess.auth import AuthStore
from plotly_chess.multiplayer import MultiplayerHub
from plotly_chess.storage import get_cloud_storage_dir

# Detect Plotly Cloud persistent filesystem (named 'Games' or via environment)
STORAGE_DIR = get_cloud_storage_dir("Games")
if STORAGE_DIR is not None:
    HUB = MultiplayerHub(db_path=STORAGE_DIR / "plotly_chess_rooms.sqlite3")
    AUTH = AuthStore(db_path=STORAGE_DIR / "plotly_chess_users.sqlite3")
else:
    HUB = None
    AUTH = None

app = Dash(
    __name__,
    assets_folder=str(Path(__file__).resolve().parent / "assets"),
    suppress_callback_exceptions=True,
)
create_app(app, hub=HUB, auth_store=AUTH)
server = app.server

if __name__ == "__main__":
    app.run(debug=False)
