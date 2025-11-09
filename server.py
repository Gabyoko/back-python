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
    planner.run_for_district(district_name, k)

@app.get("/")
def root():
    return send_from_directory(BASE_DIR, "index.html")

@app.get("/favicon.ico")
def favicon():
    path = os.path.join(BASE_DIR, "favicon.ico")
    if os.path.exists(path):
        return send_from_directory(BASE_DIR, "favicon.ico")
    return ("", 204)

# --- INICIO DE LA CORRECCIÓN ---
@app.get("/files/<path:subpath>")
def files(subpath: str):
    # 1. Definimos el directorio donde REALMENTE están los archivos.
    files_dir = os.path.join(BASE_DIR, "out_runs")
    
    # 2. Comprobamos si el archivo existe dentro de "out_runs"
    # full ahora será /.../back-python/out_runs/chorrillos/patrol_1.html
    full = os.path.join(files_dir, subpath)
    if not os.path.isfile(full):
        abort(404) # No encontrado

    # 3. Servimos el archivo desde el directorio "out_runs"
    return send_from_directory(files_dir, subpath)
# --- FIN DE LA CORRECCIÓN ---

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
