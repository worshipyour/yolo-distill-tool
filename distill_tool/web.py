"""Loopback-only UI backed by real CLI training jobs."""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .catalog import MODELS, method_catalog
from .jobs import JobManager, build_recipe, read_json, tail


def hardware():
    code = """import json, sys
try:
 import torch, torchvision
 print(json.dumps({'ready': True, 'python': sys.version.split()[0], 'torch': torch.__version__, 'gpus': [{'id': str(i), 'name': torch.cuda.get_device_name(i), 'memory_gb': round(torch.cuda.get_device_properties(i).total_memory/1024**3,1)} for i in range(torch.cuda.device_count())]}))
except Exception as exc:
 print(json.dumps({'ready': False, 'python': sys.version.split()[0], 'error': str(exc), 'gpus': []}))
"""
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
                                **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"ready": False, "error": str(exc), "gpus": [], "python": sys.version.split()[0]}


def make_server(port, state):
    manager = JobManager(state)
    token = secrets.token_urlsafe(32)
    assets = Path(__file__).parent / "ui_static"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, value, status=200, content_type="application/json; charset=utf-8"):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def authorized(self, api=False):
            allowed = {"127.0.0.1:" + str(self.server.server_port), "localhost:" + str(self.server.server_port)}
            if self.headers.get("Host") not in allowed:
                self.send({"error": "Invalid host"}, 403)
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + h for h in allowed}:
                self.send({"error": "Invalid origin"}, 403)
                return False
            if api and not secrets.compare_digest(self.headers.get("X-Distill-Token", ""), token):
                self.send({"error": "请从工具启动地址打开界面"}, 403)
                return False
            return True

        def do_GET(self):
            url = urlparse(self.path)
            if not self.authorized(api=url.path.startswith("/api/")):
                return
            try:
                query = parse_qs(url.query)
                if url.path in ("/", "/index.html"):
                    page = (assets / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", token)
                    self.send(page.encode("utf-8"), content_type="text/html; charset=utf-8")
                elif url.path in ("/app.js", "/style.css"):
                    self.send((assets / url.path[1:]).read_bytes(), content_type="text/javascript; charset=utf-8" if url.path.endswith(".js") else "text/css; charset=utf-8")
                elif url.path == "/api/catalog":
                    self.send({"models": MODELS, "methods": method_catalog(), "default_output": os.environ.get("YOLO_DISTILL_OUTPUT", str(Path.cwd() / "runs")), "default_device": os.environ.get("YOLO_DISTILL_DEVICE", "0"), "engine": str(manager.engine), "home": str(Path.home())})
                elif url.path == "/api/hardware":
                    self.send(hardware())
                elif url.path == "/api/jobs":
                    self.send(manager.list())
                elif url.path == "/api/job":
                    self.send(manager.describe(query["id"][0], logs=True))
                elif url.path == "/api/report":
                    directory = manager.directory(query["id"][0])
                    cfg = read_json(directory / "resolved.json")
                    self.send(tail(Path(cfg["workspace"]) / "REPORT.md", limit=100000).encode("utf-8"), content_type="text/plain; charset=utf-8")
                elif url.path == "/api/browse":
                    directory = Path(query.get("path", [str(Path.home())])[0]).expanduser().absolute()
                    if not directory.is_dir():
                        directory = directory.parent
                    entries = []
                    for entry in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                        if entry.name.startswith("."):
                            continue
                        if entry.is_dir() or entry.suffix.lower() in (".yaml", ".yml", ".pt"):
                            entries.append({"name": entry.name, "path": str(entry), "directory": entry.is_dir()})
                    self.send({"path": str(directory), "parent": str(directory.parent), "entries": entries[:1000]})
                else:
                    self.send({"error": "Not found"}, 404)
            except (ValueError, KeyError, OSError) as exc:
                self.send({"error": str(exc)}, 400)

        def do_POST(self):
            if not self.authorized(api=True):
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 262144:
                    raise ValueError("请求长度无效")
                values = json.loads(self.rfile.read(size))
                if not isinstance(values, dict):
                    raise ValueError("请求格式无效")
                if self.path == "/api/start":
                    self.send(manager.start(values), 201)
                elif self.path == "/api/resume":
                    self.send(manager.resume(values["id"]))
                elif self.path == "/api/stop":
                    self.send(manager.stop(values["id"]))
                elif self.path == "/api/plan":
                    from .pipeline import plan
                    cfg = build_recipe(values, str(manager.state / "preview-output"), values.get("engine") or manager.engine)
                    result = plan(cfg)
                    self.send({"stages": [row["name"] for row in result["stages"]], "network_hw": result["network_nchw"][-2:], "workflow": cfg["workflow"]})
                else:
                    self.send({"error": "Not found"}, 404)
            except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
                self.send({"error": str(exc)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.manager = manager
    return server


def serve(port=8765, state="~/.yolo-distill", open_browser=True):
    server = make_server(port, state)
    url = "http://127.0.0.1:" + str(server.server_port)
    print("YOLO Distill Tool: " + url, flush=True)
    print("关闭浏览器不会停止已提交的训练；运行记录保存在 " + str(server.manager.state), flush=True)
    if open_browser:
        threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
