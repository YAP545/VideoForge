import httpx, time
from urllib.parse import quote

p = "stylized 3D animated movie character, small black kitten holding a glowing staff, snowy forest, wide shot, bright cheerful colors"
tests = [(512, 512), (768, 432), (1024, 576), (1280, 720)]
for w, h in tests:
    t = time.time()
    try:
        r = httpx.get("https://image.pollinations.ai/prompt/" + quote(p) + f"?width={w}&height={h}&nologo=true&seed=7", timeout=120, follow_redirects=True)
        body = r.text[:300] if "json" in r.headers.get("content-type", "") else ""
        print(w, h, r.status_code, len(r.content), round(time.time() - t, 1), "s", body, flush=True)
    except Exception as e:
        print(w, h, "ERROR", type(e).__name__, str(e)[:100], flush=True)
    time.sleep(16)