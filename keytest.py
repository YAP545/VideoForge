import httpx, time, os
from urllib.parse import quote
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("POLLINATIONS_API_KEY", "")
print("key set:", bool(key))
p = "stylized 3D animated movie character, small black kitten holding a glowing staff, snowy forest, wide shot, bright cheerful colors"
for i in range(1, 4):
    t = time.time()
    try:
        r = httpx.get("https://gen.pollinations.ai/image/" + quote(p), params={"width": 1280, "height": 720, "seed": i, "key": key}, timeout=120, follow_redirects=True)
        body = r.text[:200] if "json" in r.headers.get("content-type", "") else ""
        print(i, r.status_code, r.headers.get("content-type"), len(r.content), round(time.time() - t, 1), "s", body, flush=True)
    except Exception as e:
        print(i, "ERROR", type(e).__name__, str(e)[:100], flush=True)