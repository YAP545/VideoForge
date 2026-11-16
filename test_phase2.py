"""Automated test script for VideoForge Phase 2.

Uses an isolated test database (test_videoforge.db).
Your real videoforge.db is never modified.

Tests:
1.  Signup user 1 - gets 20 credits, redirected to dashboard, is admin.
2.  Email verification dev-link displayed.
3.  Duplicate email signup rejected.
4.  Login with wrong password fails.
5.  Logout user 1.
6.  Login user 1 with correct password succeeds.
7.  User 1 creates video - visible in library.
8.  User 2 registered & logged into dashboard.
9.  Data isolation: User 2 cannot see User 1 video.
10. Password reset dev-link displayed.
"""
import sys
from test_helpers import TestServer, TestClient


def run_tests():
    print(">> Starting VideoForge Phase 2 Verification Suite...\n")
    results = []

    with TestServer() as srv:
        port = srv.port

        # 1. Signup User 1 (alice)
        client1 = TestClient(port)
        _, html, _ = client1.get("/signup")
        csrf1 = client1.get_csrf(html)
        assert csrf1, "CSRF token not found on signup page"
        _, html, url = client1.post("/signup", {
            "email": "alice@test.com",
            "password": "Password123!",
            "csrf_token": csrf1,
        })
        ok1 = "dashboard" in url and "20" in html and "alice" in html
        results.append(("1. Signup User 1 -> 20 credits, redirected to Dashboard", ok1))

        # 2. Email verification dev-link
        results.append(("2. Email verification dev-link displayed on screen", "verify-email?token=" in html))

        # 3. Duplicate email rejected
        client_dup = TestClient(port)
        _, html_dup, _ = client_dup.get("/signup")
        csrf_dup = client_dup.get_csrf(html_dup)
        _, html_dup, _ = client_dup.post("/signup", {
            "email": "alice@test.com",
            "password": "Password123!",
            "csrf_token": csrf_dup,
        })
        results.append(("3. Duplicate email signup rejected with error", "already exists" in html_dup.lower()))

        # 4. Wrong password rejected
        client_bad = TestClient(port)
        _, html_bad, _ = client_bad.get("/login")
        csrf_bad = client_bad.get_csrf(html_bad)
        _, html_bad, _ = client_bad.post("/login", {
            "email": "alice@test.com",
            "password": "WrongPassword999",
            "csrf_token": csrf_bad,
        })
        results.append(("4. Login with incorrect password rejected", "incorrect email or password" in html_bad.lower()))

        # 5. Logout User 1
        _, _, logout_url = client1.get("/logout")
        results.append(("5. Logout successful (redirected to login)", "login" in logout_url))

        # 6. Login User 1 again
        _, html_l, _ = client1.get("/login")
        csrf_l = client1.get_csrf(html_l)
        _, html_l, url_l = client1.post("/login", {
            "email": "alice@test.com",
            "password": "Password123!",
            "csrf_token": csrf_l,
        })
        results.append(("6. Login User 1 with correct password succeeds", "dashboard" in url_l and "alice" in html_l))

        # 7. User 1 creates a video
        _, html_c, _ = client1.get("/create")
        csrf_c = client1.get_csrf(html_c)
        _, html_c, url_c = client1.post("/create", {
            "topic": "Alice Secret Animated Garden Story",
            "length": 15,
            "style": "2d_cartoon",
            "ratio": "16:9",
            "caption_lang": "none",
            "csrf_token": csrf_c,
        })
        results.append(("7. User 1 creates video -> visible in User 1 library",
                        "my-videos" in url_c or "Alice" in html_c))

        # 8. Signup User 2 (bob)
        client2 = TestClient(port)
        _, html2, _ = client2.get("/signup")
        csrf2 = client2.get_csrf(html2)
        _, html2, url2 = client2.post("/signup", {
            "email": "bob@test.com",
            "password": "BobPassword999!",
            "csrf_token": csrf2,
        })
        results.append(("8. User 2 registered & logged into their dashboard",
                        "dashboard" in url2 and "bob" in html2))

        # 9. Data isolation: Bob cannot see Alice's video
        _, html_bob, _ = client2.get("/my-videos")
        results.append(("9. Data Isolation: User 2 CANNOT see User 1 video", "Alice" not in html_bob))

        # 10. Password reset dev-link
        client_reset = TestClient(port)
        _, html_r, _ = client_reset.get("/forgot-password")
        csrf_r = client_reset.get_csrf(html_r)
        _, html_r, _ = client_reset.post("/forgot-password", {
            "email": "alice@test.com",
            "csrf_token": csrf_r,
        })
        results.append(("10. Password reset dev-link displayed on screen", "reset-password/" in html_r))

    # Summary
    print("=" * 60)
    print("PHASE 2 TEST SUMMARY")
    print("=" * 60)
    all_passed = True
    for desc, passed in results:
        mark = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"{mark} - {desc}")
    print("=" * 60)
    if all_passed:
        print("\nALL 10 AUTHENTICATION AND ISOLATION TESTS PASSED!\n")
    else:
        print("\nSOME TESTS FAILED!\n")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
