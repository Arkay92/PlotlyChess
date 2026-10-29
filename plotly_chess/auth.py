"""User registration, login, profile settings, external chess account sync and leaderboard store."""
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sqlite3
import ssl
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

INITIAL_RATING = 1000
VALID_AVATARS = ("K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p")


def rank_for_rating(rating):
    """Assign a chess rank title from a player's rating."""
    if rating >= 1600:
        return "Grandmaster"
    if rating >= 1400:
        return "Master"
    if rating >= 1200:
        return "Tactician"
    if rating >= 1050:
        return "Club Player"
    if rating >= 900:
        return "Contender"
    return "Novice"


def _http_get_json(url):
    req = Request(url, headers={"User-Agent": "PlotlyChess/1.0 (+https://plotly.com)", "Accept": "application/json"})
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    try:
        with urlopen(req, timeout=6, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except ssl.SSLError:
        unverified = ssl._create_unverified_context()
        with urlopen(req, timeout=6, context=unverified) as resp:
            return json.loads(resp.read().decode("utf-8"))


def fetch_lichess_stats(username, http_get=None):
    """Fetch public profile rating and W/L/D counts from Lichess API."""
    cleaned = (username or "").strip()
    if not cleaned:
        raise ValueError("Enter a Lichess username to link.")
    getter = http_get or _http_get_json
    try:
        data = getter(f"https://lichess.org/api/user/{quote(cleaned)}")
    except HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"Lichess user '{cleaned}' was not found.") from exc
        raise ValueError(f"Lichess returned HTTP {exc.code}.") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise ValueError(f"Could not reach Lichess to verify '{cleaned}'.") from exc

    perfs = data.get("perfs") or {}
    rating = None
    for mode in ("rapid", "blitz", "classical", "bullet", "correspondence"):
        r = (perfs.get(mode) or {}).get("rating")
        if isinstance(r, int) and not (perfs.get(mode) or {}).get("prov", False):
            rating = r
            break
    if rating is None:
        for mode in ("rapid", "blitz", "classical", "bullet", "correspondence"):
            r = (perfs.get(mode) or {}).get("rating")
            if isinstance(r, int):
                rating = r
                break
    counts = data.get("count") or {}
    return {
        "username": data.get("username") or cleaned,
        "rating": int(rating or INITIAL_RATING),
        "wins": int(counts.get("win") or 0),
        "losses": int(counts.get("loss") or 0),
        "draws": int(counts.get("draw") or 0),
    }


def fetch_chesscom_stats(username, http_get=None):
    """Fetch public profile rating and W/L/D counts from Chess.com Published-Data API."""
    cleaned = (username or "").strip()
    if not cleaned:
        raise ValueError("Enter a Chess.com username to link.")
    getter = http_get or _http_get_json
    try:
        stats = getter(f"https://api.chess.com/pub/player/{quote(cleaned.lower())}/stats")
    except HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"Chess.com user '{cleaned}' was not found.") from exc
        raise ValueError(f"Chess.com returned HTTP {exc.code}.") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise ValueError(f"Could not reach Chess.com to verify '{cleaned}'.") from exc

    rating = None
    wins = losses = draws = 0
    for mode in ("chess_rapid", "chess_blitz", "chess_bullet", "chess_daily"):
        section = stats.get(mode) or {}
        last_rating = (section.get("last") or {}).get("rating")
        if rating is None and isinstance(last_rating, int):
            rating = last_rating
        rec = section.get("record") or {}
        wins += int(rec.get("win") or 0)
        losses += int(rec.get("loss") or 0)
        draws += int(rec.get("draw") or 0)
    return {
        "username": cleaned,
        "rating": int(rating or INITIAL_RATING),
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }


def _format_profile(record):
    wins = int(record.get("wins") or 0)
    losses = int(record.get("losses") or 0)
    draws = int(record.get("draws") or 0)
    rating = int(record.get("rating") or INITIAL_RATING)
    avatar = record.get("avatar") or "K"
    if avatar not in VALID_AVATARS:
        avatar = "K"
    display_name = (record.get("display_name") or "").strip() or record["username"]
    lichess_user = record.get("lichess_username") or None
    chesscom_user = record.get("chesscom_username") or None
    try:
        saved_games = json.loads(record.get("saved_games_json") or "[]")
        if not isinstance(saved_games, list):
            saved_games = []
    except Exception:
        saved_games = []
    return {
        "username": record["username"],
        "display_name": display_name,
        "avatar": avatar,
        "role": "user",
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": wins + losses + draws,
        "rating": rating,
        "rank": rank_for_rating(rating),
        "saved_games": saved_games,
        "lichess": {
            "username": lichess_user,
            "rating": int(record.get("lichess_rating") or 0),
            "wins": int(record.get("lichess_wins") or 0),
            "losses": int(record.get("lichess_losses") or 0),
            "draws": int(record.get("lichess_draws") or 0),
        } if lichess_user else None,
        "chesscom": {
            "username": chesscom_user,
            "rating": int(record.get("chesscom_rating") or 0),
            "wins": int(record.get("chesscom_wins") or 0),
            "losses": int(record.get("chesscom_losses") or 0),
            "draws": int(record.get("chesscom_draws") or 0),
        } if chesscom_user else None,
    }


_EXTRA_COLUMNS = (
    ("wins", "INTEGER NOT NULL DEFAULT 0"),
    ("losses", "INTEGER NOT NULL DEFAULT 0"),
    ("draws", "INTEGER NOT NULL DEFAULT 0"),
    ("rating", f"INTEGER NOT NULL DEFAULT {INITIAL_RATING}"),
    ("display_name", "TEXT NOT NULL DEFAULT ''"),
    ("avatar", "TEXT NOT NULL DEFAULT 'K'"),
    ("lichess_username", "TEXT"),
    ("lichess_rating", "INTEGER NOT NULL DEFAULT 0"),
    ("lichess_wins", "INTEGER NOT NULL DEFAULT 0"),
    ("lichess_losses", "INTEGER NOT NULL DEFAULT 0"),
    ("lichess_draws", "INTEGER NOT NULL DEFAULT 0"),
    ("chesscom_username", "TEXT"),
    ("chesscom_rating", "INTEGER NOT NULL DEFAULT 0"),
    ("chesscom_wins", "INTEGER NOT NULL DEFAULT 0"),
    ("chesscom_losses", "INTEGER NOT NULL DEFAULT 0"),
    ("chesscom_draws", "INTEGER NOT NULL DEFAULT 0"),
    ("saved_games_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("last_active", "REAL NOT NULL DEFAULT 0"),
)


class AuthStore:
    """Thread-safe and multi-worker-safe user account, linked external profiles and leaderboard store."""

    def __init__(self, shared=False, db_path=None, http_get=None):
        self._lock = threading.Lock()
        self._memory_users = {}
        self._memory_friends = set()
        self._memory_activity = {}
        self._http_get = http_get
        if db_path is not None:
            self._db_path = Path(db_path)
        elif shared:
            from .storage import get_cloud_storage_dir
            cloud_dir = get_cloud_storage_dir("Games")
            if cloud_dir is not None:
                self._db_path = cloud_dir / "plotly_chess_users.sqlite3"
            else:
                self._db_path = Path(tempfile.gettempdir()) / "plotly_chess_users.sqlite3"
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
                    CREATE TABLE IF NOT EXISTS users (
                        username_key TEXT PRIMARY KEY,
                        username TEXT NOT NULL,
                        salt TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        wins INTEGER NOT NULL DEFAULT 0,
                        losses INTEGER NOT NULL DEFAULT 0,
                        draws INTEGER NOT NULL DEFAULT 0,
                        rating INTEGER NOT NULL DEFAULT 1000,
                        display_name TEXT NOT NULL DEFAULT '',
                        avatar TEXT NOT NULL DEFAULT 'K',
                        lichess_username TEXT,
                        lichess_rating INTEGER NOT NULL DEFAULT 0,
                        lichess_wins INTEGER NOT NULL DEFAULT 0,
                        lichess_losses INTEGER NOT NULL DEFAULT 0,
                        lichess_draws INTEGER NOT NULL DEFAULT 0,
                        chesscom_username TEXT,
                        chesscom_rating INTEGER NOT NULL DEFAULT 0,
                        chesscom_wins INTEGER NOT NULL DEFAULT 0,
                        chesscom_losses INTEGER NOT NULL DEFAULT 0,
                        chesscom_draws INTEGER NOT NULL DEFAULT 0,
                        saved_games_json TEXT NOT NULL DEFAULT '[]',
                        last_active REAL NOT NULL DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS friends (
                        user_id TEXT NOT NULL,
                        friend_id TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        PRIMARY KEY (user_id, friend_id)
                    )
                    """
                )
                for col, col_def in _EXTRA_COLUMNS:
                    try:
                        conn.execute(f"ALTER TABLE users ADD COLUMN {col} {col_def}")
                    except sqlite3.Error:
                        pass
        except sqlite3.Error:
            self._db_path = None

    @staticmethod
    def _hash_password(password, salt_hex):
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            100_000,
        )
        return dk.hex()

    @staticmethod
    def _validate_credentials(username, password):
        cleaned_user = (username or "").strip()
        cleaned_pass = password or ""
        if len(cleaned_user) < 2 or len(cleaned_user) > 24:
            raise ValueError("Enter a username between 2 and 24 characters.")
        if len(cleaned_pass) < 4:
            raise ValueError("Enter a password of at least 4 characters.")
        return cleaned_user, cleaned_pass

    def _get_user_locked(self, username_key):
        if self._db_path:
            try:
                with self._connect_db() as conn:
                    row = conn.execute("SELECT * FROM users WHERE username_key = ?", (username_key,)).fetchone()
                    if row:
                        rec = dict(row)
                        self._memory_users[username_key] = rec
                        return rec
            except sqlite3.Error:
                pass
        return self._memory_users.get(username_key)

    def _save_user_locked(self, record):
        self._memory_users[record["username_key"]] = record
        if self._db_path:
            try:
                with self._connect_db() as conn:
                    conn.execute(
                        """
                        INSERT INTO users (
                            username_key, username, salt, password_hash,
                            wins, losses, draws, rating, display_name, avatar,
                            lichess_username, lichess_rating, lichess_wins, lichess_losses, lichess_draws,
                            chesscom_username, chesscom_rating, chesscom_wins, chesscom_losses, chesscom_draws,
                            saved_games_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(username_key) DO UPDATE SET
                            salt=excluded.salt,
                            password_hash=excluded.password_hash,
                            wins=excluded.wins,
                            losses=excluded.losses,
                            draws=excluded.draws,
                            rating=excluded.rating,
                            display_name=excluded.display_name,
                            avatar=excluded.avatar,
                            lichess_username=excluded.lichess_username,
                            lichess_rating=excluded.lichess_rating,
                            lichess_wins=excluded.lichess_wins,
                            lichess_losses=excluded.lichess_losses,
                            lichess_draws=excluded.lichess_draws,
                            chesscom_username=excluded.chesscom_username,
                            chesscom_rating=excluded.chesscom_rating,
                            chesscom_wins=excluded.chesscom_wins,
                            chesscom_losses=excluded.chesscom_losses,
                            chesscom_draws=excluded.chesscom_draws,
                            saved_games_json=excluded.saved_games_json
                        """,
                        (
                            record["username_key"],
                            record["username"],
                            record["salt"],
                            record["password_hash"],
                            int(record.get("wins") or 0),
                            int(record.get("losses") or 0),
                            int(record.get("draws") or 0),
                            int(record.get("rating") or INITIAL_RATING),
                            record.get("display_name") or record["username"],
                            record.get("avatar") or "K",
                            record.get("lichess_username"),
                            int(record.get("lichess_rating") or 0),
                            int(record.get("lichess_wins") or 0),
                            int(record.get("lichess_losses") or 0),
                            int(record.get("lichess_draws") or 0),
                            record.get("chesscom_username"),
                            int(record.get("chesscom_rating") or 0),
                            int(record.get("chesscom_wins") or 0),
                            int(record.get("chesscom_losses") or 0),
                            int(record.get("chesscom_draws") or 0),
                            record.get("saved_games_json") or "[]",
                        ),
                    )
            except sqlite3.Error:
                pass

    def register(self, username, password):
        cleaned_user, cleaned_pass = self._validate_credentials(username, password)
        key = cleaned_user.lower()
        with self._lock:
            if self._get_user_locked(key) is not None:
                raise ValueError(f"Username '{cleaned_user}' is already registered. Please log in.")
            salt = secrets.token_hex(16)
            pw_hash = self._hash_password(cleaned_pass, salt)
            record = {
                "username_key": key,
                "username": cleaned_user,
                "display_name": cleaned_user,
                "avatar": "K",
                "salt": salt,
                "password_hash": pw_hash,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "rating": INITIAL_RATING,
            }
            self._save_user_locked(record)
            return _format_profile(record)

    def login(self, username, password):
        cleaned_user = (username or "").strip()
        cleaned_pass = password or ""
        if not cleaned_user or not cleaned_pass:
            raise ValueError("Enter both your username and password.")
        key = cleaned_user.lower()
        with self._lock:
            record = self._get_user_locked(key)
            if not record:
                raise ValueError("Account not found. Check your username or register.")
            expected = record["password_hash"]
            actual = self._hash_password(cleaned_pass, record["salt"])
            if not hmac.compare_digest(expected, actual):
                raise ValueError("Incorrect password. Please try again.")
            return _format_profile(record)

    def get_profile(self, username):
        key = (username or "").strip().lower()
        if not key:
            return None
        with self._lock:
            record = self._get_user_locked(key)
            return _format_profile(record) if record else None

    def update_profile(self, username, display_name=None, avatar=None):
        key = (username or "").strip().lower()
        with self._lock:
            record = self._get_user_locked(key)
            if not record:
                raise ValueError("Account not found.")
            if display_name is not None:
                cleaned_display = display_name.strip()
                if len(cleaned_display) < 2 or len(cleaned_display) > 28:
                    raise ValueError("Display name must be between 2 and 28 characters.")
                record["display_name"] = cleaned_display
            if avatar is not None:
                if avatar not in VALID_AVATARS:
                    raise ValueError("Choose a valid chess piece avatar.")
                record["avatar"] = avatar
            self._save_user_locked(record)
            return _format_profile(record)

    def change_password(self, username, current_password, new_password):
        key = (username or "").strip().lower()
        if not current_password or not new_password:
            raise ValueError("Enter both your current password and a new password.")
        if len(new_password) < 4:
            raise ValueError("New password must be at least 4 characters.")
        with self._lock:
            record = self._get_user_locked(key)
            if not record:
                raise ValueError("Account not found.")
            actual = self._hash_password(current_password, record["salt"])
            if not hmac.compare_digest(record["password_hash"], actual):
                raise ValueError("Current password is incorrect.")
            salt = secrets.token_hex(16)
            record["salt"] = salt
            record["password_hash"] = self._hash_password(new_password, salt)
            self._save_user_locked(record)
            return _format_profile(record)

    def link_external_account(self, username, provider, external_username):
        """Fetch live stats from Lichess or Chess.com and save them to the user's DB record."""
        key = (username or "").strip().lower()
        provider = (provider or "").strip().lower()
        if provider == "lichess":
            stats = fetch_lichess_stats(external_username, http_get=self._http_get)
        elif provider == "chesscom":
            stats = fetch_chesscom_stats(external_username, http_get=self._http_get)
        else:
            raise ValueError("Unsupported chess platform.")

        with self._lock:
            record = self._get_user_locked(key)
            if not record:
                raise ValueError("Register or log in to an account before linking external profiles.")
            if provider == "lichess":
                record["lichess_username"] = stats["username"]
                record["lichess_rating"] = stats["rating"]
                record["lichess_wins"] = stats["wins"]
                record["lichess_losses"] = stats["losses"]
                record["lichess_draws"] = stats["draws"]
            else:
                record["chesscom_username"] = stats["username"]
                record["chesscom_rating"] = stats["rating"]
                record["chesscom_wins"] = stats["wins"]
                record["chesscom_losses"] = stats["losses"]
                record["chesscom_draws"] = stats["draws"]
            self._save_user_locked(record)
            return _format_profile(record)

    def save_user_games(self, username, saved_games):
        """Persist a registered user's list of in-progress games."""
        key = (username or "").strip().lower()
        if not key:
            return None
        with self._lock:
            record = self._get_user_locked(key)
            if not record:
                return None
            record["saved_games_json"] = json.dumps(saved_games or [])
            self._save_user_locked(record)
            return _format_profile(record)

    def record_result(self, username, outcome):
        """Record a 'win', 'loss', or 'draw' for a registered user and update their rating & rank."""
        key = (username or "").strip().lower()
        if not key or outcome not in ("win", "loss", "draw"):
            return None
        with self._lock:
            record = self._get_user_locked(key)
            if not record:
                return None
            if outcome == "win":
                record["wins"] = int(record.get("wins") or 0) + 1
                record["rating"] = int(record.get("rating") or INITIAL_RATING) + 25
            elif outcome == "loss":
                record["losses"] = int(record.get("losses") or 0) + 1
                record["rating"] = max(100, int(record.get("rating") or INITIAL_RATING) - 20)
            elif outcome == "draw":
                record["draws"] = int(record.get("draws") or 0) + 1
                record["rating"] = int(record.get("rating") or INITIAL_RATING) + 5
            self._save_user_locked(record)
            return _format_profile(record)

    def leaderboard(self, limit=50):
        """Return ranked player profiles ordered by rating, wins, and fewest losses."""
        with self._lock:
            records = []
            if self._db_path:
                try:
                    with self._connect_db() as conn:
                        rows = conn.execute("SELECT * FROM users").fetchall()
                        records = [dict(r) for r in rows]
                except sqlite3.Error:
                    records = list(self._memory_users.values())
            else:
                records = list(self._memory_users.values())

            profiles = [_format_profile(r) for r in records]
            profiles.sort(key=lambda p: (-p["rating"], -p["wins"], p["losses"], p["username"].lower()))
            result = []
            for idx, profile in enumerate(profiles[:limit], start=1):
                result.append({**profile, "position": idx})
            return result

    def touch_user_activity(self, username):
        """Update last active timestamp for online presence detection."""
        key = (username or "").strip().lower()
        if not key:
            return
        now = time.time()
        with self._lock:
            self._memory_activity[key] = now
            if self._db_path:
                try:
                    with self._connect_db() as conn:
                        conn.execute("UPDATE users SET last_active = ? WHERE username_key = ?", (now, key))
                except sqlite3.Error:
                    pass

    def is_online(self, username, max_inactive=120.0):
        """Check whether a user was active within max_inactive seconds."""
        key = (username or "").strip().lower()
        if not key:
            return False
        now = time.time()
        with self._lock:
            ts = self._memory_activity.get(key)
            if self._db_path:
                try:
                    with self._connect_db() as conn:
                        row = conn.execute("SELECT last_active FROM users WHERE username_key = ?", (key,)).fetchone()
                        if row and row["last_active"]:
                            db_ts = float(row["last_active"])
                            if ts is None or db_ts > ts:
                                ts = db_ts
                except sqlite3.Error:
                    pass
            return bool(ts and (now - ts) <= max_inactive)

    def add_friend(self, username, friend_username):
        """Add friend_username to username's friends list."""
        u_key = (username or "").strip().lower()
        f_key = (friend_username or "").strip().lower()
        if not u_key or not f_key:
            raise ValueError("Invalid user.")
        if u_key == f_key:
            raise ValueError("You cannot add yourself as a friend.")
        with self._lock:
            u_rec = self._get_user_locked(u_key)
            f_rec = self._get_user_locked(f_key)
            if not u_rec:
                raise ValueError("User account not found.")
            if not f_rec:
                raise ValueError(f"Player '{friend_username}' does not exist.")
            now = time.time()
            self._memory_friends.add((u_key, f_key))
            if self._db_path:
                try:
                    with self._connect_db() as conn:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO friends (user_id, friend_id, created_at)
                            VALUES (?, ?, ?)
                            """,
                            (u_key, f_key, now),
                        )
                except sqlite3.Error:
                    pass
            return True

    def remove_friend(self, username, friend_username):
        """Remove friend_username from username's friends list."""
        u_key = (username or "").strip().lower()
        f_key = (friend_username or "").strip().lower()
        if not u_key or not f_key:
            return False
        with self._lock:
            self._memory_friends.discard((u_key, f_key))
            if self._db_path:
                try:
                    with self._connect_db() as conn:
                        conn.execute(
                            "DELETE FROM friends WHERE user_id = ? AND friend_id = ?",
                            (u_key, f_key),
                        )
                except sqlite3.Error:
                    pass
            return True

    def is_friend(self, username, friend_username):
        """Check if username and friend_username are friends."""
        u_key = (username or "").strip().lower()
        f_key = (friend_username or "").strip().lower()
        if not u_key or not f_key:
            return False
        with self._lock:
            if (u_key, f_key) in self._memory_friends:
                return True
            if self._db_path:
                try:
                    with self._connect_db() as conn:
                        row = conn.execute(
                            "SELECT 1 FROM friends WHERE user_id = ? AND friend_id = ?",
                            (u_key, f_key),
                        ).fetchone()
                        if row:
                            self._memory_friends.add((u_key, f_key))
                            return True
                except sqlite3.Error:
                    pass
            return False

    def get_friends(self, username):
        """Return list of friend profiles for username with online status."""
        u_key = (username or "").strip().lower()
        if not u_key:
            return []
        friend_keys = set()
        with self._lock:
            for u, f in self._memory_friends:
                if u == u_key:
                    friend_keys.add(f)
            if self._db_path:
                try:
                    with self._connect_db() as conn:
                        rows = conn.execute("SELECT friend_id FROM friends WHERE user_id = ?", (u_key,)).fetchall()
                        for row in rows:
                            fk = row["friend_id"]
                            friend_keys.add(fk)
                            self._memory_friends.add((u_key, fk))
                except sqlite3.Error:
                    pass
        friends = []
        for fk in friend_keys:
            prof = self.get_profile(fk)
            if prof:
                online = self.is_online(fk)
                friends.append({**prof, "is_online": online})
        # Sort online friends first, then by rating descending, then display name
        friends.sort(key=lambda f: (0 if f["is_online"] else 1, -f["rating"], f["display_name"].lower()))
        return friends

    @staticmethod
    def guest():
        return {
            "username": "Guest",
            "display_name": "Guest",
            "avatar": "K",
            "role": "guest",
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "games": 0,
            "rating": INITIAL_RATING,
            "rank": "Unranked Guest",
            "lichess": None,
            "chesscom": None,
        }
