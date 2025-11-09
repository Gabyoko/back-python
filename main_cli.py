# main_cli.py
import os, sys, re, unicodedata, random, math, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from string import Template

GRAPH_BASE = "out_graphs/graphs"
RUN_BASE   = "out_runs"

DISTRICTS = [
    "Chorrillos",
    "San Juan de Miraflores",
    "Villa El Salvador",
    "Villa María del Triunfo",
]

random.seed(42)

def palette(n: int):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]

def matplotlib_color_to_hex(c):
    import matplotlib.colors as mcolors
    return mcolors.to_hex(c)

class NodeAccessor:
    def __init__(self, store: dict):
        self._store = store
    def __getitem__(self, n): return self._store[n]
    def __iter__(self): return iter(self._store.keys())
    def items(self): return self._store.items()
    def keys(self): return self._store.keys()
    def __contains__(self, n): return n in self._store

class SimpleGraph:
    def __init__(self):
        self._nodes = {}
        self._adj   = defaultdict(dict)
    @property
    def nodes(self): return NodeAccessor(self._nodes)
    def add_node(self, n, **attrs): self._nodes.setdefault(n, {}).update(attrs)
    def add_edge(self, u, v, **attrs):
        if u == v:
            self._adj[u][v] = attrs; return
        if v in self._adj[u]:
            w_old = float(self._adj[u][v].get("weight", 1.0))
            w_new = float(attrs.get("weight", 1.0))
            if w_new < w_old:
                self._adj[u][v] = dict(attrs); self._adj[v][u] = dict(attrs)
        else:
            self._adj[u][v] = dict(attrs); self._adj[v][u] = dict(attrs)
    def edges(self, data=False):
        seen = set()
        for u in self._adj:
            for v, d in self._adj[u].items():
                if (v, u) in seen: continue
                seen.add((u, v))
                yield (u, v, d) if data else (u, v)
    def has_edge(self, u, v): return v in self._adj.get(u, {})
    def __getitem__(self, u): return self._adj[u]
    def neighbors(self, u): return self._adj[u].keys()
    def degree(self):
        for n in self._nodes: yield (n, len(self._adj.get(n, {})))
    def number_of_nodes(self): return len(self._nodes)
    def number_of_edges(self): return sum(1 for _ in self.edges())
    def subgraph(self, node_set):
        H = SimpleGraph()
        for n in node_set:
            if n in self._nodes: H.add_node(n, **self._nodes[n])
        for u, v, d in self.edges(data=True):
            if u in node_set and v in node_set: H.add_edge(u, v, **d)
        return H
    def copy(self):
        H = SimpleGraph()
        for n, d in self._nodes.items(): H.add_node(n, **d.copy())
        for u, v, d in self.edges(data=True): H.add_edge(u, v, **d.copy())
        return H

class MultiGraph:
    def __init__(self):
        self._nodes = {}
        self._adj = defaultdict(lambda: defaultdict(list))
    @property
    def nodes(self): return NodeAccessor(self._nodes)
    def add_node(self, n, **attrs):
        self._nodes.setdefault(n, {}).update(attrs); _ = self._adj[n]
    def add_edge(self, u, v, **attrs):
        self.add_node(u, **self._nodes.get(u, {})); self.add_node(v, **self._nodes.get(v, {}))
        self._adj[u][v].append(dict(attrs)); self._adj[v][u].append(dict(attrs))
    def degree(self):
        for n in self._nodes: yield (n, sum(len(lst) for lst in self._adj[n].values()))
    def edges_iter(self, u): return self._adj[u].items()
    def has_edges(self, u, v): return len(self._adj[u][v]) > 0
    def pop_edge(self, u, v):
        if self._adj[u][v]:
            self._adj[u][v].pop(); self._adj[v][u].pop()
    def number_of_edges(self): return sum(sum(len(lst) for lst in self._adj[u].values()) for u in self._adj)//2

class MultiDiGraphLite:
    def __init__(self):
        self._nodes = {}
        self._edges = []
    @property
    def nodes(self): return NodeAccessor(self._nodes)
    def add_node(self, n, **attrs): self._nodes.setdefault(n, {}).update(attrs)
    def add_edge(self, u, v, **attrs): self._edges.append((u, v, dict(attrs)))
    def edges(self, data=False):
        for u, v, d in self._edges: yield (u, v, d) if data else (u, v)
    def number_of_nodes(self): return len(self._nodes)
    def number_of_edges(self): return len(self._edges)
    def subgraph(self, node_set):
        H = MultiDiGraphLite()
        for n in node_set:
            if n in self._nodes: H.add_node(n, **self._nodes[n])
        for u, v, d in self._edges:
            if u in node_set and v in node_set: H.add_edge(u, v, **d)
        return H
    def to_undirected_simple(self):
        Gu = SimpleGraph()
        for n, d in self._nodes.items(): Gu.add_node(n, **d)
        crime_acc = defaultdict(float)
        for u, v, data in self._edges:
            w = float(data.get("travel_cost", data.get("length", 1.0)))
            Gu.add_edge(u, v, weight=w)
            cs = float(data.get("crime_share", 0.0))
            a, b = (u, v) if u <= v else (v, u)
            crime_acc[(a, b)] += cs
        for (a, b), c in crime_acc.items():
            if Gu.has_edge(a, b):
                Gu[a][b]["crime_sum"] = Gu[a][b].get("crime_sum", 0.0) + c
                Gu[b][a]["crime_sum"] = Gu[a][b]["crime_sum"]
        return Gu
    def neighbors_undirected(self, u):
        nbrs = set()
        for a, b, _ in self._edges:
            if a == u: nbrs.add(b)
            if b == u: nbrs.add(a)
        return nbrs

def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^\w\s-]", "", s.lower())
    return re.sub(r"[\s-]+", "_", s).strip("_")

def load_graph(dname: str) -> MultiDiGraphLite:
    path = os.path.join(GRAPH_BASE, f"{slug(dname)}.graphml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el grafo para {dname}. Ejecuta primero build_graphs.py")
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    key_map = {}
    for k in root.findall("g:key", ns):
        key_map[k.attrib["id"]] = k.attrib.get("attr.name", "")
    def data_dict(elem):
        dd = {}
        for d in elem.findall("g:data", ns):
            name = key_map.get(d.attrib["key"], d.attrib["key"])
            dd[name] = d.text if d.text is not None else ""
        return dd
    G = MultiDiGraphLite()
    for n in root.findall(".//g:node", ns):
        nid = n.attrib["id"]
        attrs = data_dict(n)
        if "x" in attrs:
            try: attrs["x"] = float(attrs["x"])
            except: attrs["x"] = 0.0
        if "y" in attrs:
            try: attrs["y"] = float(attrs["y"])
            except: attrs["y"] = 0.0
        G.add_node(nid, **attrs)
    for e in root.findall(".//g:edge", ns):
        u = e.attrib["source"]; v = e.attrib["target"]
        attrs = data_dict(e)
        for k in ("length","travel_cost","crime_share","crime_sum"):
            if k in attrs:
                try: attrs[k] = float(attrs[k])
                except: pass
        G.add_edge(u, v, **attrs)
    return G

def node_arrays(G: MultiDiGraphLite):
    ids = np.array([n for n, _ in G.nodes.items()])
    xs  = np.array([G.nodes[n].get("x", 0.0) for n in ids], dtype=float)
    ys  = np.array([G.nodes[n].get("y", 0.0) for n in ids], dtype=float)
    return ids, xs, ys

def nearest_node_by_xy(G: MultiDiGraphLite, x, y):
    best, bd = None, 1e300
    for n, d in G.nodes.items():
        dx = d.get("x", 0.0) - x
        dy = d.get("y", 0.0) - y
        dist2 = dx*dx + dy*dy
        if dist2 < bd:
            bd, best = dist2, n
    return best

def node_crime_score(G) -> dict:
    score = defaultdict(float)
    if isinstance(G, MultiDiGraphLite):
        it = G.edges(data=True)
    else:
        it = ((u, v, G[u][v]) for u, v in G.edges())
    for u, v, data in it:
        cs = float(data.get("crime_share", data.get("crime_sum", 0.0)))
        score[u] += cs; score[v] += cs
    eps = 1e-6
    for n in (G.nodes.keys() if isinstance(G.nodes, NodeAccessor) else G.nodes()):
        score[n] = score.get(n, 0.0) + eps
    return score

def connected_components_simple(Gu: SimpleGraph):
    seen = set()
    for s in Gu.nodes:
        if s in seen: continue
        comp = []
        q = deque([s]); seen.add(s)
        while q:
            u = q.popleft(); comp.append(u)
            for v in Gu.neighbors(u):
                if v not in seen:
                    seen.add(v); q.append(v)
        yield set(comp)

def largest_component_nodes(Gu: SimpleGraph):
    comps = list(connected_components_simple(Gu))
    if not comps: return set()
    return max(comps, key=len)

def induced_subgraph_largest(G: MultiDiGraphLite, node_set: set) -> MultiDiGraphLite:
    Gi = G.subgraph(node_set)
    if Gi.number_of_nodes() == 0: return Gi
    tmp = SimpleGraph()
    for n, d in Gi.nodes.items(): tmp.add_node(n, **d)
    for u, v, _ in Gi.edges(data=True): tmp.add_edge(u, v, weight=1.0)
    biggest = largest_component_nodes(tmp)
    return Gi.subgraph(biggest)

def kmeans_partition_nodes(G: MultiDiGraphLite, k: int) -> dict:
    from sklearn.cluster import KMeans
    ids, xs, ys = node_arrays(G)
    weights_map = node_crime_score(G)
    w = np.array([weights_map.get(n, 1.0) for n in ids], dtype=float)
    k = max(1, min(k, len(ids)))
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(np.vstack([xs, ys]).T, sample_weight=w)
    return {n: int(lbl) for n, lbl in zip(ids, labels)}

def graph_to_undirected(Gi: MultiDiGraphLite) -> SimpleGraph:
    H = SimpleGraph()
    for n, d in Gi.nodes.items(): H.add_node(n, **d)
    crime_acc = defaultdict(float)
    for u, v, data in Gi.edges(data=True):
        w = float(data.get("travel_cost", data.get("length", 1.0)))
        H.add_edge(u, v, weight=w)
        a, b = (u, v) if u <= v else (v, u)
        crime_acc[(a, b)] += float(data.get("crime_share", 0.0))
    for (a, b), c in crime_acc.items():
        if H.has_edge(a, b):
            H[a][b]["crime_sum"] = H[a][b].get("crime_sum", 0.0) + c
            H[b][a]["crime_sum"] = H[a][b]["crime_sum"]
    return H

class UFDS:
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.rank   = {n: 0 for n in nodes}
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra; self.rank[ra] += 1
        return True

def bfs_neighbors(Gu: SimpleGraph, s):
    seen = {s}; q = deque([s]); order = []
    while q:
        u = q.popleft(); order.append(u)
        for v in Gu.neighbors(u):
            if v not in seen:
                seen.add(v); q.append(v)
    return order

def dfs_neighbors(Gu: SimpleGraph, s):
    seen = {s}; st = [s]; order = []
    while st:
        u = st.pop(); order.append(u)
        for v in Gu.neighbors(u):
            if v not in seen:
                seen.add(v); st.append(v)
    return order

def dijkstra_len(Gu: SimpleGraph, s, t) -> float:
    if s not in Gu.nodes or t not in Gu.nodes: return float("inf")
    dist = {n: float("inf") for n in Gu.nodes}; dist[s] = 0.0
    visited = set()
    import heapq
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        if u == t: return d
        for v in Gu.neighbors(u):
            w = float(Gu[u][v].get("weight", 1.0))
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd; heapq.heappush(pq, (nd, v))
    return float("inf")

def dijkstra_path(Gu: SimpleGraph, s, t):
    if s not in Gu.nodes or t not in Gu.nodes: return [s]
    dist = {n: float("inf") for n in Gu.nodes}; prev = {}; dist[s] = 0.0
    import heapq
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]: continue
        if u == t: break
        for v in Gu.neighbors(u):
            w = float(Gu[u][v].get("weight", 1.0))
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
    if t not in prev and s != t: return [s]
    path = [t]
    while path[-1] != s: path.append(prev[path[-1]])
    path.reverse(); return path

def odd_nodes(Gu: SimpleGraph): return [n for n, deg in Gu.degree() if deg % 2 == 1]

def greedy_min_matching_odd(Gu: SimpleGraph, odd_list: list, beam: int = 4) -> list[tuple]:
    coords = {n: (Gu.nodes[n]["x"], Gu.nodes[n]["y"]) for n in odd_list}
    unused = set(odd_list); pairs = []
    while len(unused) >= 2:
        a = min(unused); unused.remove(a)
        ax, ay = coords[a]
        cand = sorted(unused, key=lambda b: (coords[b][0]-ax)**2 + (coords[b][1]-ay)**2)[:beam]
        best_b, best_len = None, float("inf")
        for b in cand:
            L = dijkstra_len(Gu, a, b)
            if L < best_len: best_len = L; best_b = b
        if best_b is None:
            best_b = min(unused); best_len = dijkstra_len(Gu, a, best_b)
        if best_b in unused: unused.remove(best_b)
        pairs.append((a, best_b))
    return pairs

def make_eulerian_multigraph(Gu: SimpleGraph) -> MultiGraph:
    M = MultiGraph()
    for n, d in Gu.nodes.items(): M.add_node(n, **d)
    for u, v, data in Gu.edges(data=True):
        M.add_node(u, **Gu.nodes[u]); M.add_node(v, **Gu.nodes[v]); M.add_edge(u, v, **data)
    odd = odd_nodes(Gu)
    if odd:
        pairs = greedy_min_matching_odd(Gu, odd, beam=6)
        for a, b in pairs:
            path = dijkstra_path(Gu, a, b)
            for u, v in zip(path, path[1:]): M.add_edge(u, v, **Gu[u][v])
    odds = [n for n, deg in M.degree() if deg % 2 == 1]
    while len(odds) >= 2:
        a, b = odds.pop(), odds.pop(); M.add_edge(a, b, weight=0.0)
    return M

def euler_route(M: MultiGraph) -> list:
    start = None
    for n, deg in M.degree():
        if deg > 0: start = n; break
    if start is None:
        if len(list(M.nodes)) == 0: return []
        return [next(iter(M.nodes))]
    stack = [start]; circuit = []; cur = start
    while stack:
        if any(len(lst) > 0 for _, lst in M._adj[cur].items()):
            stack.append(cur)
            for v, lst in M._adj[cur].items():
                if lst: M.pop_edge(cur, v); cur = v; break
        else:
            circuit.append(cur); cur = stack.pop()
    circuit.reverse(); return circuit

def choose_start_by_hotspot(G: MultiDiGraphLite | SimpleGraph, nodes_subset: set) -> str:
    score = node_crime_score(G)
    xs = {n: (G.nodes[n]["x"] if isinstance(G, (MultiDiGraphLite, SimpleGraph)) else 0.0) for n in nodes_subset}
    ys = {n: (G.nodes[n]["y"] if isinstance(G, (MultiDiGraphLite, SimpleGraph)) else 0.0) for n in nodes_subset}
    ws = {n: score.get(n, 1e-6) for n in nodes_subset}
    cx = sum(xs[n]*ws[n] for n in nodes_subset) / max(1e-9, sum(ws.values()))
    cy = sum(ys[n]*ws[n] for n in nodes_subset) / max(1e-9, sum(ws.values()))
    s0 = nearest_node_by_xy(G, cx, cy)
    if s0 in nodes_subset: return s0
    if isinstance(G, MultiDiGraphLite):
        seed = next(iter(nodes_subset)); visited = {seed}; stack = [seed]; best, bestd = seed, 1e30
        while stack:
            u = stack.pop()
            d2 = (G.nodes[u]["x"]-cx)**2 + (G.nodes[u]["y"]-cy)**2
            if d2 < bestd: bestd, best = d2, u
            for v in G.neighbors_undirected(u):
                if v in nodes_subset and v not in visited: visited.add(v); stack.append(v)
        return best
    else:
        seed = next(iter(nodes_subset)); visited = {seed}; stack=[seed]; best, bestd = seed, 1e30
        while stack:
            u = stack.pop()
            d2 = (G.nodes[u]["x"]-cx)**2 + (G.nodes[u]["y"]-cy)**2
            if d2 < bestd: bestd, best = d2, u
            for v in G.neighbors(u):
                if v in nodes_subset and v not in visited: visited.add(v); stack.append(v)
        return best

def rotate_path_to_start(path: list, start) -> list:
    if not path: return path
    i = path.index(start) if start in path else 0
    return path[i:] + path[:i]

# ---------- Render HTML (per patrulla y combinado) ----------
def _viewbox(G):
    xs = [d.get("x", 0.0) for _, d in G.nodes.items()]
    ys = [d.get("y", 0.0) for _, d in G.nodes.items()]
    if not xs or not ys: return (0, 0, 100, 100)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    pad_x = (maxx - minx) * 0.03 if maxx > minx else 1.0
    pad_y = (maxy - miny) * 0.03 if maxy > miny else 1.0
    return (minx - pad_x, miny - pad_y, (maxx - minx) + 2*pad_x, (maxy - miny) + 2*pad_y)

def _controls_js():
    return """
(function(){
  const svg = document.getElementById('svg');
  let vb = svg.viewBox.baseVal;
  const init = {x: vb.x, y: vb.y, w: vb.width, h: vb.height};
  function setVB(x,y,w,h){vb.x=x; vb.y=y; vb.width=w; vb.height=h;}
  function zoom(cx,cy,scale){
    const nx = cx - (cx - vb.x) / scale;
    const ny = cy - (cy - vb.y) / scale;
    const nw = vb.width / scale;
    const nh = vb.height / scale;
    setVB(nx, ny, nw, nh);
  }
  svg.addEventListener('wheel', e=>{
    e.preventDefault();
    const s = (e.deltaY<0)?1.15:1/1.15;
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const ctm = svg.getScreenCTM().inverse(); const p = pt.matrixTransform(ctm);
    zoom(p.x, p.y, s);
  }, {passive:false});
  let dragging=false, sx=0, sy=0, startX=0, startY=0;
  svg.addEventListener('pointerdown', e=>{dragging=true; svg.setPointerCapture(e.pointerId); sx=e.clientX; sy=e.clientY; startX=vb.x; startY=vb.y; svg.style.cursor='grabbing';});
  svg.addEventListener('pointermove', e=>{if(!dragging) return; const dx=e.clientX-sx, dy=e.clientY-sy; const scaleX = vb.width/svg.clientWidth; const scaleY = vb.height/svg.clientHeight; setVB(startX - dx*scaleX, startY - dy*scaleY, vb.width, vb.height); });
  svg.addEventListener('pointerup', e=>{dragging=false; svg.releasePointerCapture(e.pointerId); svg.style.cursor='grab';});
  document.getElementById('zoomIn').onclick=()=>zoom(vb.x+vb.width/2, vb.y+vb.height/2, 1.2);
  document.getElementById('zoomOut').onclick=()=>zoom(vb.x+vb.width/2, vb.y+vb.height/2, 1/1.2);
  document.getElementById('reset').onclick=()=>setVB(init.x,init.y,init.w,init.h);
})();
"""

def write_route_html(G, path, color_hex, out_path, title):
    vb_minx, vb_miny, vb_w, vb_h = _viewbox(G)

    lines_bg = []
    for u, v, _ in G.edges(data=True):
        xu, yu = G.nodes[u].get("x", 0.0), G.nodes[u].get("y", 0.0)
        xv, yv = G.nodes[v].get("x", 0.0), G.nodes[v].get("y", 0.0)
        lines_bg.append(f"<line x1='{xu}' y1='{yu}' x2='{xv}' y2='{yv}' stroke='#d0d0d0' stroke-opacity='0.75' stroke-width='0.8' vector-effect='non-scaling-stroke' shape-rendering='crispEdges' />")

    route_poly = ""
    if len(path) >= 2:
        pts = []
        for n in path:
            nd = G.nodes[n]; pts.append(f"{nd['x']},{nd['y']}")
        route_poly = f"<polyline fill='none' stroke='{color_hex}' stroke-opacity='0.95' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round' vector-effect='non-scaling-stroke' points=\"{' '.join(pts)}\" />"

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>{title}</title>
<style>
body{{margin:0;background:#ffffff;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial}}
.header{{position:sticky;top:0;background:#fff;padding:12px 16px;border-bottom:1px solid #eee;font-weight:600}}
.wrap{{display:flex;justify-content:center;align-items:center;height:calc(100vh - 60px);padding:8px}}
.stage{{width:100%;max-width:1200px;height:100%;border:1px solid #eee;box-shadow:0 2px 10px rgba(0,0,0,.04);border-radius:8px;overflow:hidden;position:relative;background:#fff}}
.canvas{{width:100%;height:100%;touch-action:none;cursor:grab}}
.controls{{position:absolute;right:12px;top:12px;display:flex;gap:6px}}
.controls button{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:6px 10px;cursor:pointer}}
.legend{{position:absolute;left:12px;top:12px;background:#fff;border:1px solid #eee;border-radius:8px;padding:6px 10px;font-size:12px;color:#333}}
</style></head>
<body>
<div class="header">{title}</div>
<div class="wrap">
  <div class="stage">
    <div class="controls">
      <button id="zoomIn">+</button>
      <button id="zoomOut">−</button>
      <button id="reset">Reset</button>
    </div>
    <svg id="svg" class="canvas" viewBox="{vb_minx} {vb_miny} {vb_w} {vb_h}" xmlns="http://www.w3.org/2000/svg">
      <g>{"".join(lines_bg)}</g>
      <g>{route_poly}</g>
    </svg>
  </div>
</div>
<script>{_controls_js()}</script>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

def write_combined_html(G, routes, out_path, title):
    vb_minx, vb_miny, vb_w, vb_h = _viewbox(G)
    colors = [matplotlib_color_to_hex(c) for c in palette(len(routes))]

    lines_bg = []
    for u, v, _ in G.edges(data=True):
        xu, yu = G.nodes[u].get("x", 0.0), G.nodes[u].get("y", 0.0)
        xv, yv = G.nodes[v].get("x", 0.0), G.nodes[v].get("y", 0.0)
        lines_bg.append(f"<line x1='{xu}' y1='{yu}' x2='{xv}' y2='{yv}' stroke='#d0d0d0' stroke-opacity='0.75' stroke-width='0.8' vector-effect='non-scaling-stroke' shape-rendering='crispEdges' />")

    polylines = []
    legend_items = []
    for i, path in enumerate(routes, start=1):
        color_hex = colors[i-1]
        if len(path) >= 2:
            pts = []
            for n in path:
                nd = G.nodes[n]; pts.append(f"{nd['x']},{nd['y']}")
            polylines.append(f"<polyline fill='none' stroke='{color_hex}' stroke-opacity='0.95' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round' vector-effect='non-scaling-stroke' points=\"{' '.join(pts)}\" />")
        legend_items.append(f"<div><span style='display:inline-block;width:12px;height:3px;background:{color_hex};margin-right:6px;vertical-align:middle'></span>Patrulla {i}</div>")

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>{title}</title>
<style>
body{{margin:0;background:#ffffff;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial}}
.header{{position:sticky;top:0;background:#fff;padding:12px 16px;border-bottom:1px solid #eee;font-weight:600}}
.wrap{{display:flex;justify-content:center;align-items:center;height:calc(100vh - 60px);padding:8px}}
.stage{{width:100%;max-width:1200px;height:100%;border:1px solid #eee;box-shadow:0 2px 10px rgba(0,0,0,.04);border-radius:8px;overflow:hidden;position:relative;background:#fff}}
.canvas{{width:100%;height:100%;touch-action:none;cursor:grab}}
.controls{{position:absolute;right:12px;top:12px;display:flex;gap:6px}}
.controls button{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:6px 10px;cursor:pointer}}
.legend{{position:absolute;left:12px;top:12px;background:#fff;border:1px solid #eee;border-radius:8px;padding:6px 10px;font-size:12px;color:#333;max-width:200px}}
.legend h4{{margin:0 0 6px 0;font-size:12px}}
</style></head>
<body>
<div class="header">{title}</div>
<div class="wrap">
  <div class="stage">
    <div class="controls">
      <button id="zoomIn">+</button>
      <button id="zoomOut">−</button>
      <button id="reset">Reset</button>
    </div>
    <div class="legend">
      <h4>Rutas</h4>
      {"".join(legend_items)}
    </div>
    <svg id="svg" class="canvas" viewBox="{vb_minx} {vb_miny} {vb_w} {vb_h}" xmlns="http://www.w3.org/2000/svg">
      <g>{"".join(lines_bg)}</g>
      <g>{"".join(polylines)}</g>
    </svg>
  </div>
</div>
<script>{_controls_js()}</script>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

# ---------- Planificación ----------
def auto_plan_and_route(G: MultiDiGraphLite, k: int) -> list[list]:
    assign = kmeans_partition_nodes(G, k)
    clusters = [[] for _ in range(k)]
    for n, lbl in assign.items(): clusters[lbl].append(n)
    routes = []
    for i in range(k):
        Gi = induced_subgraph_largest(G, set(clusters[i]))
        if Gi.number_of_edges() == 0 or Gi.number_of_nodes() < 2:
            routes.append(list(Gi.nodes.keys())[:1]); continue
        Gu = graph_to_undirected(Gi)
        seq_total = []
        for comp in connected_components_simple(Gu):
            Gc = Gu.subgraph(comp).copy()
            M  = make_eulerian_multigraph(Gc)
            seq = euler_route(M)
            s0  = choose_start_by_hotspot(Gi, set(comp))
            seq = rotate_path_to_start(seq, s0)
            if seq_total and seq and seq_total[-1] != seq[0]: seq_total.append(seq[0])
            seq_total.extend(seq)
        routes.append(seq_total)
    def route_cost(seq):
        if len(seq) < 2: return 0.0
        Gu_all = graph_to_undirected(G)
        tot = 0.0
        for a, b in zip(seq, seq[1:]):
            if Gu_all.has_edge(a, b): tot += float(Gu_all[a][b].get("weight", 1.0))
        return tot
    while len(routes) > k:
        costs = sorted([(i, route_cost(r)) for i, r in enumerate(routes)], key=lambda x: x[1])
        i1, i2 = costs[0][0], costs[1][0]
        r1, r2 = routes[i1], routes[i2]
        merged = r1 + ([r2[0]] if r1 and r2 and r1[-1] != r2[0] else []) + r2
        routes = [r for j, r in enumerate(routes) if j not in (i1, i2)] + [merged]
    while len(routes) < k: routes.append([])
    return routes[:k]

# ---------- Entradas CLI / Backend ----------
def choose(prompt, options):
    print(prompt)
    for i, opt in enumerate(options, start=1): print(f"  {i}. {opt}")
    n = input("Número: ").strip()
    try:
        idx = int(n)
        if 1 <= idx <= len(options): return options[idx-1]
    except: pass
    print("Entrada inválida."); sys.exit(1)

def run_for_district(district: str, k: int):
    G = load_graph(district)
    os.makedirs(RUN_BASE, exist_ok=True)
    out_dir = os.path.join(RUN_BASE, slug(district))
    os.makedirs(out_dir, exist_ok=True)
    k = max(1, min(150, int(k)))
    routes = auto_plan_and_route(G, k)
    title = f"{district} — Cobertura rápida (k={k})"
    # Per-patrulla
    colors = [matplotlib_color_to_hex(c) for c in palette(len(routes))]
    for i, path in enumerate(routes, start=1):
        write_route_html(G, path, colors[i-1], os.path.join(out_dir, f"patrol_{i}.html"), f"Patrulla {i} — {title}")
    # Combinado
    write_combined_html(G, routes, os.path.join(out_dir, "combined.html"), title)

def main():
    print("===============================================")
    print("  Sistema de Patrullaje (Cobertura Automática)")
    print("===============================================")
    input("Presiona ENTER para continuar…\n")
    d = choose("Elige un distrito:", DISTRICTS)
    k = input("Cantidad de patrullas (1–150): ").strip()
    try: k = int(k)
    except: print("Valor inválido."); sys.exit(1)
    run_for_district(d, k)

if __name__ == "__main__":
    main()
