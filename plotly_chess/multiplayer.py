"""Server-authoritative multiplayer rooms and RFC 6455 WebSocket transport."""
import base64
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import struct
import tempfile
import threading
from urllib.parse import parse_qs

import chess

from . import game

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _recv_exact(sock, length):
    buf = bytearray()
    while len(buf) < length:
        chunk = sock.recv(length - len(buf))
        if not chunk:
            raise ConnectionError("WebSocket closed")
        buf.extend(chunk)
    return bytes(buf)


class WebSocketClient:
    """Minimal RFC 6455 WebSocket connection wrapper over a raw WSGI socket."""

    def __init__(self, sock):
        self.sock = sock
        self._lock = threading.Lock()
        self.closed = False

    def send_json(self, payload):
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_frame(0x1, data)

    def _send_frame(self, opcode, data=b""):
        if self.closed:
            return
        length = len(data)
        if length < 126:
            header = bytes([0x80 | opcode, length])
        elif length < 65536:
            header = struct.pack(">BBH", 0x80 | opcode, 126, length)
        else:
            header = struct.pack(">BBQ", 0x80 | opcode, 127, length)
        with self._lock:
            try:
                self.sock.sendall(header + data)
            except OSError:
                self.closed = True
                raise ConnectionError("WebSocket send failed")

    def receive_json(self):
        while not self.closed:
            b1, b2 = _recv_exact(self.sock, 2)
            opcode = b1 & 0x0F
            masked = bool(b2 & 0x80)
            length = b2 & 0x7F
            if length == 126:
                (length,) = struct.unpack(">H", _recv_exact(self.sock, 2))
            elif length == 127:
                (length,) = struct.unpack(">Q", _recv_exact(self.sock, 8))
            mask = _recv_exact(self.sock, 4) if masked else b""
            payload = _recv_exact(self.sock, length) if length else b""
            if masked and payload:
                payload = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
            if opcode == 0x8:  # Close
                try:
                    self._send_frame(0x8, b"")
                except OSError:
                    pass
                self.closed = True
                return None
            if opcode == 0x9:  # Ping
                self._send_frame(0xA, payload)
                continue
            if opcode == 0x1:  # Text
                return json.loads(payload.decode("utf-8"))
        return None

    def close(self):
        self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass


class MultiplayerHub:
    """Room registry with WebSocket pub/sub broadcast and optional SQLite persistence for multi-worker cloud deployments."""

    def __init__(self, shared=False, db_path=None):
        self._lock = threading.Lock()
        self._rooms = {}
        self._clients = {}
        if db_path is not None:
            self._db_path = Path(db_path)
        elif shared:
            self._db_path = Path(tempfile.gettempdir()) / "plotly_chess_rooms.sqlite3"
        else:
            self._db_path = None
        if self._db_path:
            self._init_db()

    def _connect_db(self):
        conn = sqlite3.connect(str(self._db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        try:
            with self._connect_db() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rooms (
                        code TEXT PRIMARY KEY,
                        white_token TEXT NOT NULL,
                        black_token TEXT,
                        moves_json TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        message TEXT NOT NULL
                    )
                    """
                )
        except sqlite3.Error:
            self._db_path = None

    def _load_room_locked(self, code):
        if self._db_path:
            try:
                with self._connect_db() as conn:
                    row = conn.execute("SELECT * FROM rooms WHERE code = ?", (code,)).fetchone()
                    if row:
                        room = {
                            "code": row["code"],
                            "white_token": row["white_token"],
                            "black_token": row["black_token"],
                            "moves": json.loads(row["moves_json"]),
                            "version": int(row["version"]),
                            "message": row["message"],
                        }
                        self._rooms[code] = room
                        return room
            except sqlite3.Error:
                pass
        return self._rooms.get(code)

    def _save_room_locked(self, room):
        self._rooms[room["code"]] = room
        if self._db_path:
            try:
                with self._connect_db() as conn:
                    conn.execute(
                        """
                        INSERT INTO rooms (code, white_token, black_token, moves_json, version, message)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(code) DO UPDATE SET
                            white_token=excluded.white_token,
                            black_token=excluded.black_token,
                            moves_json=excluded.moves_json,
                            version=excluded.version,
                            message=excluded.message
                        """,
                        (
                            room["code"],
                            room["white_token"],
                            room["black_token"],
                            json.dumps(room["moves"]),
                            room["version"],
                            room["message"],
                        ),
                    )
            except sqlite3.Error:
                pass

    def _generate_code_locked(self):
        while True:
            code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(6))
            if self._load_room_locked(code) is None:
                return code

    def _snapshot(self, room, token=None):
        color = "white" if token == room["white_token"] else ("black" if token == room["black_token"] else None)
        return {
            "code": room["code"],
            "token": token,
            "color": color,
            "connected": room["black_token"] is not None,
            "moves": list(room["moves"]),
            "version": room["version"],
            "message": room["message"],
        }

    def _sync_event(self, room):
        return {
            "type": "sync",
            "code": room["code"],
            "connected": room["black_token"] is not None,
            "moves": list(room["moves"]),
            "version": room["version"],
            "message": room["message"],
        }

    def _broadcast_locked(self, room):
        event = self._sync_event(room)
        clients = self._clients.get(room["code"], set())
        stale = []
        for client in list(clients):
            try:
                client.send_json(event)
            except Exception:
                stale.append(client)
        for client in stale:
            clients.discard(client)

    def create_room(self):
        with self._lock:
            code = self._generate_code_locked()
            token = secrets.token_hex(8)
            room = {
                "code": code,
                "white_token": token,
                "black_token": None,
                "moves": [],
                "version": 1,
                "message": f"Game created. Share code {code} with your friend.",
            }
            self._save_room_locked(room)
            return self._snapshot(room, token)

    def join_room(self, code, token=None):
        cleaned = (code or "").strip().upper()
        if not cleaned:
            raise ValueError("Enter the 6-character code from your friend.")
        with self._lock:
            room = self._load_room_locked(cleaned)
            if not room:
                raise ValueError(f"Game code '{cleaned}' was not found.")
            if token and token == room["white_token"]:
                return self._snapshot(room, token)
            if token and token == room["black_token"]:
                return self._snapshot(room, token)
            if room["black_token"] is not None:
                raise ValueError("That game already has two players connected.")
            room["black_token"] = secrets.token_hex(8)
            room["version"] += 1
            room["message"] = "Friend connected! White to move."
            self._save_room_locked(room)
            self._broadcast_locked(room)
            return self._snapshot(room, room["black_token"])

    def get_room(self, code, token=None):
        cleaned = (code or "").strip().upper()
        with self._lock:
            room = self._load_room_locked(cleaned)
            if not room:
                return None
            return self._snapshot(room, token)

    def play_move(self, code, token, notation):
        cleaned = (code or "").strip().upper()
        with self._lock:
            room = self._load_room_locked(cleaned)
            if not room:
                raise ValueError("Multiplayer game not found.")
            if room["black_token"] is None:
                raise ValueError(f"Waiting for your friend to join with code {cleaned}.")
            board = game.restore({"moves": room["moves"], "selected": None})
            expected_token = room["white_token"] if board.turn == chess.WHITE else room["black_token"]
            if token != expected_token:
                turn_name = "White" if board.turn == chess.WHITE else "Black"
                raise ValueError(f"Wait for {turn_name} to move.")
            updated, san = game.play({"moves": room["moves"], "selected": None}, notation)
            mover = "White" if board.turn == chess.WHITE else "Black"
            room["moves"] = updated["moves"]
            room["version"] += 1
            room["message"] = f"{mover} played {san}."
            self._save_room_locked(room)
            self._broadcast_locked(room)
            return self._snapshot(room, token), san

    def reset_room(self, code, token):
        cleaned = (code or "").strip().upper()
        with self._lock:
            room = self._load_room_locked(cleaned)
            if not room:
                raise ValueError("Multiplayer game not found.")
            if token not in (room["white_token"], room["black_token"]):
                raise ValueError("Not a player in this game.")
            room["moves"] = []
            room["version"] += 1
            room["message"] = "A fresh board. White to move."
            self._save_room_locked(room)
            self._broadcast_locked(room)
            return self._snapshot(room, token)

    def subscribe(self, code, client):
        cleaned = (code or "").strip().upper()
        with self._lock:
            room = self._load_room_locked(cleaned)
            if not room:
                return None
            self._clients.setdefault(cleaned, set()).add(client)
            return self._sync_event(room)

    def unsubscribe(self, code, client):
        cleaned = (code or "").strip().upper()
        with self._lock:
            clients = self._clients.get(cleaned)
            if clients:
                clients.discard(client)


def attach_websocket(server, hub):
    """Attach a WSGI WebSocket upgrade handler (with HTTP JSON poll fallback for cloud proxies) at /ws."""
    inner_app = server.wsgi_app

    def wsgi_app(environ, start_response):
        if environ.get("PATH_INFO") != "/ws":
            return inner_app(environ, start_response)
        query = parse_qs(environ.get("QUERY_STRING", ""))
        code = (query.get("code", [""])[0] or "").strip().upper()
        upgrade = environ.get("HTTP_UPGRADE", "").lower()
        sec_key = environ.get("HTTP_SEC_WEBSOCKET_KEY", "")
        raw_sock = environ.get("werkzeug.socket")
        is_gunicorn = "gunicorn.socket" in environ or "gunicorn" in environ.get("SERVER_SOFTWARE", "").lower()
        can_persist_ws = bool(raw_sock) and not is_gunicorn and bool(environ.get("wsgi.multithread", True))

        # Cloud proxy / multi-worker HTTP fallback when not upgrading to WebSocket
        if upgrade != "websocket" and query.get("poll"):
            room = hub.get_room(code) if code else None
            if not room:
                start_response("404 Not Found", [("Content-Type", "application/json")])
                return [b'{"error":"not_found"}']
            body = json.dumps({
                "type": "sync",
                "code": room["code"],
                "connected": room["connected"],
                "moves": room["moves"],
                "version": room["version"],
                "message": room["message"],
            }, separators=(",", ":")).encode("utf-8")
            start_response("200 OK", [("Content-Type", "application/json"), ("Cache-Control", "no-store")])
            return [body]

        # Never block Gunicorn sync workers; return immediately so dcc.Interval (#mp-poll) handles sync
        if not can_persist_ws:
            start_response("204 No Content", [("Cache-Control", "no-store")])
            return [b""]

        if upgrade != "websocket" or not sec_key:
            start_response("400 Bad Request", [("Content-Type", "text/plain")])
            return [b"WebSocket upgrade required."]

        accept = base64.b64encode(hashlib.sha1((sec_key.strip() + _WS_GUID).encode("ascii")).digest()).decode("ascii")
        handshake = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        )
        raw_sock.sendall(handshake.encode("ascii"))
        client = WebSocketClient(raw_sock)
        try:
            if code:
                initial = hub.subscribe(code, client)
                if initial:
                    client.send_json(initial)
            while not client.closed:
                msg = client.receive_json()
                if msg is None:
                    break
                if msg.get("type") == "subscribe":
                    new_code = (msg.get("code") or "").strip().upper()
                    if code and code != new_code:
                        hub.unsubscribe(code, client)
                    code = new_code
                    snap = hub.subscribe(code, client)
                    if snap:
                        client.send_json(snap)
                elif msg.get("type") == "ping":
                    client.send_json({"type": "pong"})
        except Exception:
            pass
        finally:
            if code:
                hub.unsubscribe(code, client)
            client.close()
        raise ConnectionResetError()

    server.wsgi_app = wsgi_app
    return hub
