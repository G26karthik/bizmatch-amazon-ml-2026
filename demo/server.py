"""Local demo server: serves index.html, the latest run results, and re-runs the real pipeline on the demo slice.

.venv-demo\\Scripts\\python demo\\server.py     ->  http://127.0.0.1:8765
"""
import json
import os
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEMO = ROOT / "local_data" / "demo"
RESULTS = DEMO / "results.json"
CODE = {"metric": ROOT / "src/ber/metric.py", "decide": ROOT / "src/ber/decide.py"}
PORT = 8765
sys.path.insert(0, str(ROOT / "src"))
from ber.normalize import addr_tokens, name_core, name_skel  # noqa: E402

run = {"proc": None, "lines": [], "code": None}
lock = threading.Lock()


def start_run() -> bool:
    with lock:
        if run["proc"] and run["proc"].poll() is None:
            return False
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
               "NUMBA_CACHE_DIR": str(DEMO / "numba_cache")}
        cmd = [sys.executable, str(HERE / "run_pipeline.py"), "--data", str(DEMO / "data"),
               "--work", str(DEMO / "work_live"), "--results", str(RESULTS)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                             text=True, encoding="utf-8", errors="replace")
        run.update(proc=p, lines=[], code=None)

    def pump():
        for line in p.stdout:
            run["lines"].append(line.rstrip())
        run["code"] = p.wait()

    threading.Thread(target=pump, daemon=True).start()
    return True


class Handler(BaseHTTPRequestHandler):
    def send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, status=200):
        self.send(json.dumps(obj).encode(), "application/json", status)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            self.send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/api/results":
            if RESULTS.exists():
                self.send(RESULTS.read_bytes(), "application/json")
            else:
                self.json({"error": "no run yet"}, 404)
        elif u.path == "/api/status":
            i = int(q.get("from", ["0"])[0])
            p = run["proc"]
            self.json({"running": bool(p and p.poll() is None), "code": run["code"],
                       "lines": run["lines"][i:], "n": len(run["lines"])})
        elif u.path == "/api/normalize":
            name, addr = q.get("name", [""])[0][:200], q.get("addr", [""])[0][:300]
            self.json({"core": name_core(name), "skel": name_skel(name), "addr": addr_tokens(addr)})
        elif u.path == "/api/code" and q.get("f", [""])[0] in CODE:
            self.send(CODE[q["f"][0]].read_bytes(), "text/plain; charset=utf-8")
        else:
            self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        if urlparse(self.path).path == "/api/run":
            self.json({"started": start_run()})
        else:
            self.send(b"not found", "text/plain", 404)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"demo running at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    srv.serve_forever()
