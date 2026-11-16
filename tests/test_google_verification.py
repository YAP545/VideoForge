import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_homepage_google_site_verification():
    resp = client.get("/")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

    expected_tag = '<meta name="google-site-verification" content="85nD9IyozQ9j9NQnEnTNYeX_W2Te6vKuhXDCE3MVoXc" />'
    html = resp.text

    # 1. Assert exact tag is present in HTML
    assert expected_tag in html, "Exact verification tag not found in homepage HTML!"

    # 2. Assert count is exactly 1
    count = html.count(expected_tag)
    assert count == 1, f"Expected tag exactly once, found {count} times"

    # 3. Assert tag is inside <head> and before </head>
    head_start = html.find("<head>")
    head_end = html.find("</head>")
    tag_pos = html.find(expected_tag)

    assert head_start != -1 and head_end != -1, "<head> tags not found in HTML"
    assert head_start < tag_pos < head_end, "Tag is not inside <head> section!"

    print("VERIFICATION TEST PASSED: Google Search Console verification meta tag is present inside <head> exactly once.")

if __name__ == "__main__":
    test_homepage_google_site_verification()
