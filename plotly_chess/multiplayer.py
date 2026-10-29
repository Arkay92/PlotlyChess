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
import time
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
        self._sessions = {}
        self._clients = {}
        self._memory_completed_games = []
        if db_path is not None:
            self._db_path = Path(db_path)
        elif shared:
            from .storage import get_cloud_storage_dir
            cloud_dir = get_cloud_storage_dir("Games")
            if cloud_dir is not None:
                self._db_path = cloud_dir / "plotly_chess_rooms.sqlite3"
            else:
                self._db_path = Path(tempfile.gettempdir()) / "plotly_chess_rooms.sqlite3"
        else:
            self._db_path = None
        if self._db_path:
            self._init_db()

    def _connect_db(self):
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            pass
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
                        message TEXT NOT NULL,
                        host_name TEXT NOT NULL DEFAULT 'Player',
                        host_rank TEXT NOT NULL DEFAULT 'Contender (1000)',
                        host_avatar TEXT NOT NULL DEFAULT 'K',
                        is_public INTEGER NOT NULL DEFAULT 1
                    )
                    """
                )
                for col_sql in (
                    "ALTER TABLE rooms ADD COLUMN host_name TEXT NOT NULL DEFAULT 'Player'",
                    "ALTER TABLE rooms ADD COLUMN host_rank TEXT NOT NULL DEFAULT 'Contender (1000)'",
                    "ALTER TABLE rooms ADD COLUMN host_avatar TEXT NOT NULL DEFAULT 'K'",
                    "ALTER TABLE rooms ADD COLUMN is_public INTEGER NOT NULL DEFAULT 1",
                    "ALTER TABLE rooms ADD COLUMN host_user TEXT",
                    "ALTER TABLE rooms ADD COLUMN guest_name TEXT NOT NULL DEFAULT 'Friend'",
                    "ALTER TABLE rooms ADD COLUMN guest_user TEXT",
                    "ALTER TABLE rooms ADD COLUMN guest_avatar TEXT NOT NULL DEFAULT 'k'",
                ):
                    try:
                        conn.execute(col_sql)
                    except sqlite3.Error:
                        pass
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        sid TEXT PRIMARY KEY,
                        state_json TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS completed_games (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        code TEXT NOT NULL,
                        white_name TEXT NOT NULL,
                        white_user TEXT,
                        white_avatar TEXT NOT NULL DEFAULT 'K',
                        black_name TEXT NOT NULL,
                        black_user TEXT,
                        black_avatar TEXT NOT NULL DEFAULT 'k',
                        result TEXT NOT NULL,
                        result_desc TEXT NOT NULL,
                        moves_count INTEGER NOT NULL DEFAULT 0,
                        completed_at REAL NOT NULL
                    )
                    """
                )
        except sqlite3.Error:
            self._db_path = None

    def save_session(self, sid, state):
        if not sid:
            return
        now = time.time()
        entry = {"state": dict(state), "ts": now}
        with self._lock:
            self._sessions[sid] = entry
            if self._db_path:
                try:
                    with self._connect_db() as conn:
                        conn.execute(
                            """
                            INSERT INTO sessions (sid, state_json, updated_at)
                            VALUES (?, ?, ?)
                            ON CONFLICT(sid) DO UPDATE SET
                                state_json=excluded.state_json,
                                updated_at=excluded.updated_at
                            """,
                            (sid, json.dumps(state), now),
                        )
                except sqlite3.Error:
                    pass

    def get_session(self, sid, max_age=5.0):
        if not sid:
            return None
        now = time.time()
        with self._lock:
            best = self._sessions.get(sid)
            if self._db_path:
                try:
                    with self._connect_db() as conn:
                        row = conn.execute(
                            "SELECT state_json, updated_at FROM sessions WHERE sid = ?",
                            (sid,),
                        ).fetchone()
                        if row:
                            db_entry = {
                                "state": json.loads(row["state_json"]),
                                "ts": float(row["updated_at"]),
                            }
                            if not best or db_entry["state"].get("seq", 0) >= best["state"].get("seq", 0):
                                best = db_entry
                                self._sessions[sid] = best
                except (sqlite3.Error, ValueError, TypeError):
                    pass
            if best and (now - best.get("ts", 0)) <= max_age:
                return best
            return None

    def _row_to_room(self, row):
        keys = row.keys()
        return {
            "code": row["code"],
            "white_token": row["white_token"],
            "black_token": row["black_token"],
            "moves": json.loads(row["moves_json"]),
            "version": int(row["version"]),
            "message": row["message"],
            "host_name": row["host_name"] if "host_name" in keys else "Player",
            "host_rank": row["host_rank"] if "host_rank" in keys else "Contender (1000)",
            "host_avatar": row["host_avatar"] if "host_avatar" in keys else "K",
            "host_user": row["host_user"] if "host_user" in keys else None,
            "guest_name": row["guest_name"] if "guest_name" in keys else "Friend",
            "guest_user": row["guest_user"] if "guest_user" in keys else None,
            "guest_avatar": row["guest_avatar"] if "guest_avatar" in keys else "k",
            "is_public": bool(row["is_public"]) if "is_public" in keys else True,
        }

    def _load_room_locked(self, code):
        if self._db_path:
            try:
                with self._connect_db() as conn:
                    row = conn.execute("SELECT * FROM rooms WHERE code = ?", (code,)).fetchone()
                    if row:
                        room = self._row_to_room(row)
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
                        INSERT INTO rooms (
                            code, white_token, black_token, moves_json, version, message,
                            host_name, host_rank, host_avatar, is_public,
                            host_user, guest_name, guest_user, guest_avatar
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(code) DO UPDATE SET
                            white_token=excluded.white_token,
                            black_token=excluded.black_token,
                            moves_json=excluded.moves_json,
                            version=excluded.version,
                            message=excluded.message,
                            host_name=excluded.host_name,
                            host_rank=excluded.host_rank,
                            host_avatar=excluded.host_avatar,
                            is_public=excluded.is_public,
                            host_user=excluded.host_user,
                            guest_name=excluded.guest_name,
                            guest_user=excluded.guest_user,
                            guest_avatar=excluded.guest_avatar
                        """,
                        (
                            room["code"],
                            room["white_token"],
                            room["black_token"],
                            json.dumps(room["moves"]),
                            room["version"],
                            room["message"],
                            room.get("host_name", "Player"),
                            room.get("host_rank", "Contender (1000)"),
                            room.get("host_avatar", "K"),
                            1 if room.get("is_public", True) else 0,
                            room.get("host_user"),
                            room.get("guest_name", "Friend"),
                            room.get("guest_user"),
                            room.get("guest_avatar", "k"),
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
            "host_name": room.get("host_name", "Player"),
            "host_rank": room.get("host_rank", "Contender (1000)"),
            "host_avatar": room.get("host_avatar", "K"),
            "is_public": bool(room.get("is_public", True)),
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

    def create_room(self, host_name="Player", host_rank="Contender (1000)", host_avatar="K", is_public=True, host_user=None):
        with self._lock:
            code = self._generate_code_locked()
            token = secrets.token_hex(8)
            room = {
                "code": code,
                "white_token": token,
                "black_token": None,
                "moves": [],
                "version": 1,
                "message": (
                    f"Game hosted in open lobby (Code {code}). Waiting for an opponent to join…"
                    if is_public
                    else f"Game created. Share code {code} with your friend."
                ),
                "host_name": (host_name or "Player").strip() or "Player",
                "host_rank": (host_rank or "Contender (1000)").strip() or "Contender (1000)",
                "host_avatar": (host_avatar or "K").strip() or "K",
                "host_user": (host_user or "").strip() or None,
                "guest_name": "Friend",
                "guest_user": None,
                "guest_avatar": "k",
                "is_public": bool(is_public),
            }
            self._save_room_locked(room)
            return self._snapshot(room, token)

    def list_open_rooms(self):
        """Return all open rooms waiting for a second player."""
        with self._lock:
            if self._db_path:
                try:
                    with self._connect_db() as conn:
                        rows = conn.execute(
                            "SELECT * FROM rooms WHERE black_token IS NULL AND is_public = 1 ORDER BY rowid DESC"
                        ).fetchall()
                        for row in rows:
                            room = self._row_to_room(row)
                            self._rooms[room["code"]] = room
                except sqlite3.Error:
                    pass
            open_rooms = []
            for room in reversed(list(self._rooms.values())):
                if room.get("black_token") is None and room.get("is_public", True):
                    open_rooms.append(
                        {
                            "code": room["code"],
                            "host_name": room.get("host_name", "Player"),
                            "host_rank": room.get("host_rank", "Contender (1000)"),
                            "host_avatar": room.get("host_avatar", "K"),
                            "version": room.get("version", 1),
                            "message": room.get("message", ""),
                        }
                    )
            return open_rooms

    def join_room(self, code, token=None, guest_name="Friend", guest_user=None, guest_avatar="k"):
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
            room["guest_name"] = (guest_name or "Friend").strip() or "Friend"
            room["guest_user"] = (guest_user or "").strip() or None
            room["guest_avatar"] = (guest_avatar or "k").strip() or "k"
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
            
            # Check if this move ends the game
            new_board = game.restore({"moves": room["moves"], "selected": None})
            if new_board.is_game_over(claim_draw=True):
                outcome = new_board.outcome(claim_draw=True)
                if outcome and outcome.winner == chess.WHITE:
                    res = "1-0"
                    desc = "White won by checkmate" if new_board.is_checkmate() else "White won"
                elif outcome and outcome.winner == chess.BLACK:
                    res = "0-1"
                    desc = "Black won by checkmate" if new_board.is_checkmate() else "Black won"
                else:
                    res = "½-½"
                    desc = "Draw by stalemate" if new_board.is_stalemate() else "Draw"
                self.record_completed_game(
                    code=cleaned,
                    white_name=room.get("host_name", "Player"),
                    white_user=room.get("host_user"),
                    white_avatar=room.get("host_avatar", "K"),
                    black_name=room.get("guest_name", "Friend"),
                    black_user=room.get("guest_user"),
                    black_avatar=room.get("guest_avatar", "k"),
                    result=res,
                    result_desc=desc,
                    moves_count=len(room["moves"]),
                )
            
            self._save_room_locked(room)
            self._broadcast_locked(room)
            return self._snapshot(room, token), san

    def record_completed_game(
        self,
        code,
        white_name="Player",
        white_user=None,
        white_avatar="K",
        black_name="Friend",
        black_user=None,
        black_avatar="k",
        result="1-0",
        result_desc="Checkmate",
        moves_count=0,
    ):
        now = time.time()
        entry = {
            "code": code,
            "white_name": white_name,
            "white_user": (white_user or "").strip() or None,
            "white_avatar": white_avatar,
            "black_name": black_name,
            "black_user": (black_user or "").strip() or None,
            "black_avatar": black_avatar,
            "result": result,
            "result_desc": result_desc,
            "moves_count": int(moves_count),
            "completed_at": now,
        }
        with self._lock:
            self._memory_completed_games.insert(0, entry)
            if len(self._memory_completed_games) > 100:
                self._memory_completed_games.pop()
            if self._db_path:
                try:
                    with self._connect_db() as conn:
                        conn.execute(
                            """
                            INSERT INTO completed_games (
                                code, white_name, white_user, white_avatar,
                                black_name, black_user, black_avatar,
                                result, result_desc, moves_count, completed_at
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                code,
                                entry["white_name"],
                                entry["white_user"],
                                entry["white_avatar"],
                                entry["black_name"],
                                entry["black_user"],
                                entry["black_avatar"],
                                entry["result"],
                                entry["result_desc"],
                                entry["moves_count"],
                                now,
                            ),
                        )
                except sqlite3.Error:
                    pass
        return entry

    def get_recent_completed_games(self, limit=10):
        with self._lock:
            games = []
            if self._db_path:
                try:
                    with self._connect_db() as conn:
                        rows = conn.execute(
                            "SELECT * FROM completed_games ORDER BY id DESC LIMIT ?",
                            (limit,),
                        ).fetchall()
                        for row in rows:
                            games.append(dict(row))
                except sqlite3.Error:
                    pass
            if not games:
                games = list(self._memory_completed_games[:limit])
            return games[:limit]

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
