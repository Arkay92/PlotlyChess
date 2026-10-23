"""Development and WSGI entry point."""
from app import app, server
if __name__ == '__main__':
    app.run(debug=False)
