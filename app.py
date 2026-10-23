"""Plotly Cloud entry point with an explicit, discoverable Dash instance."""
from pathlib import Path

from dash import Dash

from plotly_chess.app import create_app

app = Dash(__name__, assets_folder=str(Path(__file__).resolve().parent / 'assets'))
create_app(app)
server = app.server

if __name__ == '__main__':
    app.run(debug=False)
