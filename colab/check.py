"""Check a vLLM server from the laptop: auth works, the model answers, and what the tunnel adds to latency.

    VLLM_API_KEY=... python colab/check.py https://<tunnel-url>

The tunnel's round trip lands on every client-side number, so measure TTFT and TPOT inside the VM.
This only tells you the server is up and how far away it is.
"""

import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

URL = sys.argv[1].rstrip("/")
KEY = os.environ["VLLM_API_KEY"]


def call(path, body=None, key=KEY):
    headers = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "1"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body else None
    return urllib.request.urlopen(urllib.request.Request(URL + path, data, headers), timeout=120)


def main():
    rtts = []
    for _ in range(10):
        t = time.perf_counter()
        call("/health").read()
        rtts.append((time.perf_counter() - t) * 1000)
    print(f"health round trip: median {statistics.median(rtts):.0f} ms, max {max(rtts):.0f} ms")

    try:
        call("/v1/models", key=None)
        sys.exit("FAIL: /v1/models answered without an API key")
    except urllib.error.HTTPError as e:
        assert e.code == 401, e.code
    print("no key: 401, as it should be")

    model = json.load(call("/v1/models"))["data"][0]["id"]
    print(f"model: {model}")

    body = {"model": model, "max_tokens": 32, "temperature": 0, "stream": True,
            "messages": [{"role": "user", "content": "Count from 1 to 10. /no_think"}]}
    t0 = time.perf_counter()
    stamps, text = [], ""
    for line in call("/v1/chat/completions", body):
        if line.startswith(b"data: {"):
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            if delta:
                stamps.append(time.perf_counter())
                text += delta
    gaps = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
    print(f"reply: {text.strip()!r}")
    print(f"through this path: first token {(stamps[0] - t0) * 1000:.0f} ms, "
          f"median gap {statistics.median(gaps):.1f} ms over {len(stamps)} chunks")
    print("OK")


main()
