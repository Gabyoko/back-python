# server.py
from __future__ import annotations
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import os, threading
import main_cli as planner

BASE_DIR = os.path.abspath(os.getcwd())

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

DISTRICTS = planner.DISTRICTS
SLUG_MAP = {planner.slug(n): n for n in DISTRICTS}

def resolve_district(name_or_slug: str) -> tuple[str, str]:
    s = planner.slug(name_or_slug)
    if s in SLUG_MAP:
        return SLUG_MAP[s], s
    for n in DISTRICTS:
        if n.lower() == name_or_slug.lower():
            return n, planner.slug(n)
    raise ValueError("Distrito inválido")

def run_async(district_name: str, k: int):
    try:
        print(f"------------ [Thread] ------------")
        print(f"[Thread] Iniciando cálculo para: {district_name}, k={k}")
        planner.run_for_district(district_name, k)
        print(f"[Thread] Cálculo TERMINADO para: {district_name}")
        print(f"----------------------------------")
    except Exception as e:
        # Si el cálculo falla, lo veremos en el log
        print(f"------------ [Thread] ------------")
        print(f"[Thread] ERROR FATAL en run_async: {e}")
        print(f"----------------------------------")
        import traceback
        traceback.print_exc() # Imprime el error completo

@app.get("/")
def root():
    return send_from_directory(BASE_DIR, "index.html")

@app.get("/favicon.ico")
def favicon():
    path = os.path.join(BASE_DIR, "favicon.ico")
    if os.path.exists(path):
        return send_from_directory(BASE_DIR, "favicon.ico")
    return ("", 204)

@app.get("/files/<path:subpath>")
def files(subpath: str):
    full = os.path.join(BASE_DIR, subpath)
    if not os.path.isfile(full):
        abort(404)
    return send_from_directory(BASE_DIR, subpath)

@app.post("/api/run")
def api_run():
    data = request.get_json(force=True) or {}
    district_in = data.get("district", "")
    k = int(data.get("k", 15))
    if k < 15 or k > 80:
        return jsonify({"ok": False, "error": "k fuera de rango [15..80]"}), 400
    try:
        district_name, dslug = resolve_district(district_in)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    t = threading.Thread(target=run_async, args=(district_name, k), daemon=True)
    t.start()

    out_dir = os.path.join(planner.RUN_BASE, dslug)
    patrols = [f"patrol_{i}.html" for i in range(1, k + 1)]

    return jsonify({
        "ok": True,
        "district": dslug,
        "k": k,
        "dir": f"{planner.RUN_BASE}/{dslug}",
        "files": {
            "combined": f"{planner.RUN_BASE}/{dslug}/combined.html",
            "patrols": patrols
        }
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
