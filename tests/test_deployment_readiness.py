import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure test environment
os.environ["APP_ENV"] = "development"

from fastapi.testclient import TestClient
from app import app, VIDEO_DIR
import pipeline

client = TestClient(app)

print(f"[TEST 1] VIDEO_DIR verification:")
print(f"  app.VIDEO_DIR: {VIDEO_DIR}")
print(f"  pipeline.VIDEO_DIR: {pipeline.VIDEO_DIR}")
assert VIDEO_DIR == pipeline.VIDEO_DIR, "VIDEO_DIR mismatch between app and pipeline!"
assert VIDEO_DIR.exists(), "VIDEO_DIR was not created!"
print("  -> PASS: VIDEO_DIR is identical and exists in both modules.")

print("\n[TEST 2] Health endpoint GET /health:")
resp = client.get("/health")
print(f"  Status: {resp.status_code}, Body: {resp.json()}")
assert resp.status_code == 200
assert resp.json()["status"] == "healthy"
print("  -> PASS: Health check returns HTTP 200 with JSON payload.")

print("\n[TEST 3] SEO GET /robots.txt:")
resp = client.get("/robots.txt")
print(f"  Status: {resp.status_code}, Content-Type: {resp.headers.get('content-type')}")
assert resp.status_code == 200
assert "Disallow: /videos/" in resp.text
assert "Disallow: /dashboard" in resp.text
assert "Allow: /pricing" in resp.text
assert "Sitemap:" in resp.text
print("  -> PASS: robots.txt protects private and video paths, allows public paths.")

print("\n[TEST 4] SEO GET /sitemap.xml:")
resp = client.get("/sitemap.xml")
print(f"  Status: {resp.status_code}, Content-Type: {resp.headers.get('content-type')}")
assert resp.status_code == 200
assert "application/xml" in resp.headers.get("content-type")
assert "<urlset" in resp.text
assert "<loc>" in resp.text
assert "/pricing</loc>" in resp.text
print("  -> PASS: sitemap.xml returns valid XML sitemap.")

print("\n[TEST 5] Landing page GET /:")
resp = client.get("/")
print(f"  Status: {resp.status_code}")
assert resp.status_code == 200
assert "favicon.svg" in resp.text
assert "og:title" in resp.text
assert "twitter:card" in resp.text
assert "WebApplication" in resp.text
print("  -> PASS: Landing page renders with Favicon, OpenGraph, and Schema.org metadata.")

print("\n[TEST 6] Auth routes GET /login and GET /signup:")
resp_login = client.get("/login")
assert resp_login.status_code == 200
resp_signup = client.get("/signup")
assert resp_signup.status_code == 200
print("  -> PASS: /login and /signup return HTTP 200.")

print("\n[TEST 7] Public policy pages:")
for page in ["/pricing", "/terms", "/privacy", "/content-policy"]:
    r = client.get(page)
    assert r.status_code == 200, f"{page} returned {r.status_code}"
print("  -> PASS: /pricing, /terms, /privacy, and /content-policy return HTTP 200.")

print("\nAll integration verification tests passed successfully!")
