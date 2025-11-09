# build_graphs.py
import os, re, unicodedata
import pandas as pd
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt

# ----- Config -----
CSV_FILE = "DATASET_Denuncias_Policiales_Enero 2018 a Agosto 2025.csv"
OUTPUT_DIR = "out_graphs"
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Distritos (nombres exactamente como OSM los reconoce)
DISTRICTS = [
    "Chorrillos",
    "San Juan de Miraflores",
    "Villa El Salvador",
    "Villa María del Triunfo",
]

# OSMnx
ox.settings.use_cache = True
ox.settings.log_console = False

# Visual
CMAP = {
    "Chorrillos": "#1f77b4",
    "San Juan de Miraflores": "#ff7f0e",
    "Villa El Salvador": "#2ca02c",
    "Villa María del Triunfo": "#d62728",
}

# ----- Utilidades -----
def _normalize(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip().upper()

def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^\w\s-]", "", s.lower())
    return re.sub(r"[\s-]+", "_", s).strip("_")

def read_crime_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el CSV: {path}")
    df = pd.read_csv(path, dtype=str)
    if "cantidad" not in df.columns: raise KeyError("Falta columna 'cantidad'")
    if "DIST_HECHO" not in df.columns: raise KeyError("Falta columna 'DIST_HECHO'")

    qty = (
        df["cantidad"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.extract(r"(\d+(?:\.\d+)?)", expand=False)
        .astype(float)
        .fillna(0.0)
    )
    df["cantidad"] = qty
    df["DIST_NORM"] = df["DIST_HECHO"].apply(_normalize)
    agg = df.groupby("DIST_NORM", dropna=False)["cantidad"].sum().reset_index()
    agg.rename(columns={"cantidad": "crime_count"}, inplace=True)
    return df, agg, dict(zip(agg["DIST_NORM"], agg["crime_count"]))

def build_district_graph(dname: str, crime_dict: dict):
    # 1) Polígono del distrito (Lima, Perú)
    query = f"{dname}, Lima, Peru"
    gdf = ox.geocode_to_gdf(query)
    if gdf.empty:
        raise RuntimeError(f"Geocodificación falló para: {dname}")

    # 2) Red vial
    G = ox.graph_from_polygon(
        gdf.iloc[0].geometry,
        network_type="drive",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
        custom_filter=None,
    )

    # 3) Longitudes y pesos
    G = ox.distance.add_edge_lengths(G)
    dist_norm = _normalize(dname)
    crime_total = float(crime_dict.get(dist_norm, 0.0))

    # Distribuir delito por longitud relativa
    total_len = sum(d.get("length", 0.0) for _, _, d in G.edges(data=True))
    total_len = total_len if total_len > 0 else 1.0

    for u, v, k, d in G.edges(keys=True, data=True):
        L = float(d.get("length", 0.0))
        crime_share = (L / total_len) * crime_total
        d["crime_share"] = crime_share
        d["risk_weight"] = L * (1.0 + (crime_total / (total_len / 1000.0 + 1e-9)))
        d["travel_cost"] = L
        d["district"] = dname

    for n in G.nodes():
        G.nodes[n]["district"] = dname
        G.nodes[n]["node_weight"] = (crime_total / max(1.0, total_len/1000.0))

    return G, crime_total

def save_graph_and_preview(G, dname: str, crime_total: float):
    slug = _slug(dname)
    gpath = os.path.join(GRAPH_DIR, f"{slug}.graphml")
    ipath = os.path.join(IMG_DIR, f"{slug}.png")

    ox.save_graphml(G, gpath)

    node_sizes = []
    for _, data in G.nodes(data=True):
        w = float(data.get("node_weight", 1.0))
        node_sizes.append(np.clip(w, 0.5, 12.0))

    fig, ax = ox.plot_graph(
        G,
        node_size=node_sizes,
        node_color=CMAP.get(dname, "#888"),
        edge_color="#222",
        edge_linewidth=0.25,
        bgcolor="white",
        show=False, close=False, figsize=(7,7)
    )
    ax.set_title(f"{dname} — delitos totales: {int(crime_total)}", fontsize=10)
    plt.savefig(ipath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return gpath, ipath

def main():
    df, agg, cdict = read_crime_csv(CSV_FILE)
    summary = []
    for d in DISTRICTS:
        print(f"[+] Construyendo grafo: {d}")
        G, crime_total = build_district_graph(d, cdict)
        gpath, ipath = save_graph_and_preview(G, d, crime_total)
        summary.append({
            "district": d, "crime_total": crime_total,
            "graphml": gpath, "preview_png": ipath,
            "nodes": G.number_of_nodes(), "edges": G.number_of_edges()
        })
    pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, "resumen_distritos.csv"), index=False, encoding="utf-8")
    print("[OK] Grafos por distrito generados en:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
