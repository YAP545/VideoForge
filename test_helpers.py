"""Shared test helpers for VideoForge integration tests.

All test suites must use TestServer as a context manager.
It starts a fresh uvicorn process on an isolated port with a
throwaway SQLite database (test_videoforge.db), then deletes
the database when the context exits.

Your real videoforge.db is never touched.
"""
import http.cookiejar
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent
PYTHON = str(sys.executable)


def _free_port():
    """Pick an OS-assigned free TCP port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestServer:
    """Context manager that starts a dedicated uvicorn server for tests.

    Usage::

        with TestServer(env={"APP_ENV": "production"}) as srv:
            client = srv.client()
            status, body, url = client.get("/")
    """

    def __init__(self, env: dict = None):
        self.port = _free_port()
        self.db_path = BASE_DIR / "test_videoforge.db"
        self.proc = None
        extra = env or {}
        self.env = {
            **os.environ,
            "DATABASE_URL": f"sqlite:///{self.db_path}",
            "PROVIDER": "mock",
            "APP_ENV": "development",
            **extra,
        }

    def __enter__(self):
        # Remove stale test DB if it exists
        if self.db_path.exists():
            self.db_path.unlink()

        self.proc = subprocess.Popen(
            [PYTHON, "-m", "uvicorn", "app:app", "--port", str(self.port), "--log-level", "warning"],
            cwd=str(BASE_DIR),
            env=self.env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait until the server is responding (up to 15s)
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/info", timeout=1)
                return self
            except Exception:
                time.sleep(0.3)

        self.proc.kill()
        raise RuntimeError(f"Test server on port {self.port} did not start within 15s")

    def __exit__(self, *_):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            # Give Windows a moment to release file handles
            time.sleep(0.5)
        # Delete the isolated test DB (retry on Windows file-lock)
        for attempt in range(10):
            try:
                if self.db_path.exists():
                    self.db_path.unlink()
                    print(f"[cleanup] Deleted {self.db_path.name}")
                break
            except PermissionError:
                if attempt < 9:
                    time.sleep(0.5)
                else:
                    print(f"[cleanup] WARNING: could not delete {self.db_path.name} (file locked)")

    def client(self):
        return TestClient(self.port)

    def base_url(self):
        return f"http://127.0.0.1:{self.port}"


class TestClient:
    """Thin HTTP client with cookie jar, CSRF helpers, and JSON POST support."""

    def __init__(self, port: int = 8000):
        self.base = f"http://127.0.0.1:{port}"
        self.cj = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cj)
        )

    def get(self, path: str):
        req = urllib.request.Request(self.base + path)
        try:
            resp = self.opener.open(req)
            return resp.getcode(), resp.read().decode("utf-8", errors="replace"), resp.geturl()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", errors="replace"), e.geturl()

    def post(self, path: str, data: dict):
        encoded = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=encoded)
        try:
            resp = self.opener.open(req)
            return resp.getcode(), resp.read().decode("utf-8", errors="replace"), resp.geturl()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", errors="replace"), e.geturl()

    def post_json(self, path: str, payload: dict):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.base + path, data=data, headers={"Content-Type": "application/json"}
        )
        for c in self.cj:
            req.add_header("Cookie", f"{c.name}={c.value}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace"), None
        except Exception as exc:
            return 500, str(exc), None

    def get_csrf(self, html: str) -> str:
        m = re.search(r'name=["\']csrf_token["\']\s+value=["\']([^"\']+)["\']', html)
        if not m:
            m = re.search(r'value=["\']([^"\']+)["\']\s+name=["\']csrf_token["\']', html)
        return m.group(1) if m else ""


def make_admin_and_verify(port: int, email: str, password: str) -> "TestClient":
    """Sign up the FIRST user on a fresh test server, mark email verified in DB,
    and return a logged-in TestClient.

    The test server must have been started with a fresh (empty) DB.
    Uses a direct SQLAlchemy connection to the test DB file so the real DB
    (videoforge.db) is never touched.
    """
    import sqlalchemy
    from sqlalchemy.orm import sessionmaker

    test_db_path = BASE_DIR / "test_videoforge.db"
    test_engine = sqlalchemy.create_engine(
        f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}
    )
    TestSession = sessionmaker(bind=test_engine)

    client = TestClient(port)
    _, html, _ = client.get("/signup")
    csrf = client.get_csrf(html)
    client.post("/signup", {"email": email, "password": password, "csrf_token": csrf})

    # Manually verify email directly in the test DB
    # Import User model class only (not the global engine/session)
    from models import User as UserModel
    db = TestSession()
    try:
        u = db.query(UserModel).filter(UserModel.email == email).first()
        assert u is not None, f"User {email} not found after signup"
        assert u.is_admin, "First user on fresh DB must be admin"
        u.is_verified = True
        db.commit()
    finally:
        db.close()
    test_engine.dispose()

    # Re-login so the session is verified
    _, login_html, _ = client.get("/login")
    csrf_l = client.get_csrf(login_html)
    client.post("/login", {"email": email, "password": password, "csrf_token": csrf_l})
    return client
