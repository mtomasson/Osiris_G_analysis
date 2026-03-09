# Osiris_knowledge_G_plot.py
# Single Colab cell: deterministic Omega-style graph view from G_2026-03-03c.json
# Emits:
#   - G_metrics.csv
#   - G_view.png
#   - G_view.svg
#
# Notes
# -----
# 1) This cell does NOT mutate G. It derives a deterministic Omega-style surrogate from graph structure only.
# 2) Axis family B is respected as a documented deterministic transform:
#       a1 = entity_anchor
#       a2 = relation_or_process_signal
#       a3 = eventization_signal
#       a4 = temporal_anchor
#       epsilon_i = 2*a_i - 1
# 3) Visualization is absolute-radial around a deterministic origin node and origin time.
# 4) SVG text is preserved as text (svg.fonttype = "none").

import json
import math
import hashlib
from datetime import datetime, timezone
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib import patheffects as pe

# networkx is standard in Colab; if missing, uncomment the next two lines:
# !pip -q install networkx
import networkx as nx

# ----------------------------
# User-configurable parameters
# ----------------------------
GRAPH_PATH = "G_2026-03-03c.json"   # change if needed
OUT_CSV = "G_metrics.csv"
OUT_PNG = "G_view.png"
OUT_SVG = "G_view.svg"

# Optional hard override for absolute origin node. Leave None for deterministic auto-choice.
ORIGIN_NODE_ID = None

# Optional hard override for absolute origin time. Leave None for deterministic auto-choice.
# Example: ORIGIN_TIME = "2026-03-01T00:00:00+00:00"
ORIGIN_TIME = None

# Rendering controls
FIGSIZE = (18, 18)
PNG_DPI = 800
EDGE_ALPHA = 0.055
MAX_LABELS = 140
LABEL_MIN_SIZE = 5.5
LABEL_MAX_SIZE = 12.0

# ----------------------------
# Matplotlib style
# ----------------------------
mpl.rcParams["figure.facecolor"] = "#0b0f1a"
mpl.rcParams["axes.facecolor"] = "#0b0f1a"
mpl.rcParams["savefig.facecolor"] = "#0b0f1a"
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["text.color"] = "white"

# ----------------------------
# Helpers
# ----------------------------
def stable_hash_int(s: str, nbytes: int = 8) -> int:
    h = hashlib.sha256(str(s).encode("utf-8")).digest()
    return int.from_bytes(h[:nbytes], "big", signed=False)

def stable_hash_unit(s: str) -> float:
    # deterministic float in [0,1)
    return stable_hash_int(s, nbytes=8) / float(2**64)

def parse_ts(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    try:
        ts = pd.to_datetime(x, utc=True, errors="coerce")
        return ts
    except Exception:
        return pd.NaT

def safe_list(x):
    return x if isinstance(x, list) else []

def slug_pretty(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("_", " ").replace("-", " ").strip()
    if not s:
        return s
    return " ".join(w.capitalize() if not w.isupper() else w for w in s.split())

def node_label(row):
    for key in ("label", "name", "title"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    desc = row.get("description")
    if isinstance(desc, str) and desc.strip():
        short = desc.strip().split(".")[0][:90].strip()
        if short:
            return short
    return slug_pretty(row.get("id", ""))[:90]

def metadata_to_str(md):
    if isinstance(md, dict):
        return json.dumps(md, sort_keys=True, ensure_ascii=True)
    if isinstance(md, list):
        try:
            return json.dumps(md, sort_keys=True, ensure_ascii=True)
        except Exception:
            return str(md)
    return "" if md is None else str(md)

def type_bucket(t):
    if t is None:
        return "unknown"
    t = str(t)
    return t

# Type colors tuned for a dark-background clustered look.
TYPE_COLORS = {
    "document": "#a6d8ff",
    "person": "#ff6b7d",
    "role": "#ffd166",
    "office": "#06d6a0",
    "org_unit": "#4cc9f0",
    "event": "#f72585",
    "concept": "#bdb2ff",
    "process": "#ff9f1c",
    "process_step": "#f15bb5",
    "policy": "#8ecae6",
    "diagnosis_group": "#e76f51",
    "finance_profile": "#ffb703",
    "financial_intensity_class": "#fee440",
    "cost_anchor": "#ffd166",
    "physician_scope": "#fb8500",
    "diagnosis_mix_snapshot": "#c77dff",
    "care_setting": "#80ed99",
    "expected_process": "#adb5bd",
    "expected_process_step": "#ced4da",
    "gap": "#6c757d",
    "unknown": "#9aa0a6",
}

RELATION_COLORS = {
    "references": (0.65, 0.85, 1.00, EDGE_ALPHA),
    "involves":   (1.00, 0.45, 0.55, EDGE_ALPHA),
    "related_to": (0.85, 0.70, 1.00, EDGE_ALPHA),
    "defines":    (0.40, 0.95, 0.75, EDGE_ALPHA),
    "requires_step": (1.00, 0.70, 0.25, EDGE_ALPHA),
    "requires_notice": (1.00, 0.80, 0.35, EDGE_ALPHA),
    "assigns_role": (1.00, 0.88, 0.35, EDGE_ALPHA),
    "overseen_by": (0.45, 0.90, 0.75, EDGE_ALPHA),
    "reports_to": (0.55, 0.82, 1.00, EDGE_ALPHA),
    "reported_to": (0.55, 0.82, 1.00, EDGE_ALPHA),
    "applies_to": (0.75, 0.75, 1.00, EDGE_ALPHA),
    "part_of": (0.55, 0.95, 0.75, EDGE_ALPHA),
}

SECTOR_ORDER = [
    "document", "org_unit", "office", "role", "person",
    "policy", "process", "process_step", "event", "concept",
    "diagnosis_group", "finance_profile", "financial_intensity_class",
    "cost_anchor", "physician_scope", "diagnosis_mix_snapshot",
    "care_setting", "expected_process", "expected_process_step",
    "gap", "unknown"
]
SECTOR_INDEX = {t: i for i, t in enumerate(SECTOR_ORDER)}

def get_type_color(t):
    return TYPE_COLORS.get(t, TYPE_COLORS["unknown"])

def get_relation_color(r):
    return RELATION_COLORS.get(r, (0.80, 0.80, 0.85, EDGE_ALPHA * 0.8))

# ----------------------------
# Load graph artifact
# ----------------------------
with open(GRAPH_PATH, "r", encoding="utf-8") as f:
    G_artifact = json.load(f)

assert isinstance(G_artifact, dict), "Top-level JSON must be an object."
nodes_raw = safe_list(G_artifact.get("nodes"))
edges_raw = safe_list(G_artifact.get("edges"))

if not nodes_raw or not edges_raw:
    raise ValueError("Expected top-level 'nodes' and 'edges' lists in graph artifact.")

# Canonical ordering consistent with contract spirit.
nodes_raw = sorted(nodes_raw, key=lambda x: str(x.get("id", "")))
edges_raw = sorted(
    edges_raw,
    key=lambda x: (
        str(x.get("source", "")),
        str(x.get("target", "")),
        str(x.get("relation", "")),
        str(x.get("timestamp", "")),
    ),
)

nodes_df = pd.DataFrame(nodes_raw).copy()
edges_df = pd.DataFrame(edges_raw).copy()

# Ensure expected columns exist
for col in ["id", "type", "timestamp", "description", "label", "metadata"]:
    if col not in nodes_df.columns:
        nodes_df[col] = None

for col in ["source", "target", "relation", "timestamp"]:
    if col not in edges_df.columns:
        edges_df[col] = None

nodes_df["id"] = nodes_df["id"].astype(str)
nodes_df["type"] = nodes_df["type"].fillna("unknown").astype(str)
nodes_df["timestamp_parsed"] = nodes_df["timestamp"].apply(parse_ts)
nodes_df["label_final"] = nodes_df.apply(lambda r: node_label(r.to_dict()), axis=1)
nodes_df["metadata_str"] = nodes_df["metadata"].apply(metadata_to_str)

# Filter edges to known endpoints only
node_id_set = set(nodes_df["id"])
edges_df = edges_df[
    edges_df["source"].astype(str).isin(node_id_set)
    & edges_df["target"].astype(str).isin(node_id_set)
].copy()

edges_df["source"] = edges_df["source"].astype(str)
edges_df["target"] = edges_df["target"].astype(str)
edges_df["relation"] = edges_df["relation"].fillna("unknown").astype(str)
edges_df["timestamp_parsed"] = edges_df["timestamp"].apply(parse_ts)

# ----------------------------
# Build graph
# ----------------------------
G = nx.Graph()
for row in nodes_df.itertuples(index=False):
    G.add_node(row.id, node_type=row.type)

for row in edges_df.itertuples(index=False):
    G.add_edge(row.source, row.target, relation=row.relation)

# Components
components = list(nx.connected_components(G))
comp_index = {}
comp_size = {}
for i, comp in enumerate(sorted(components, key=lambda c: (-len(c), sorted(c)[0]))):
    for nid in comp:
        comp_index[nid] = i
        comp_size[nid] = len(comp)

nodes_df["component_id"] = nodes_df["id"].map(comp_index)
nodes_df["component_size"] = nodes_df["id"].map(comp_size)

# Degree metrics
deg = dict(G.degree())
nodes_df["degree"] = nodes_df["id"].map(deg).fillna(0).astype(int)

deg_cent = nx.degree_centrality(G) if len(G) > 1 else {n: 0.0 for n in G.nodes()}
nodes_df["degree_centrality"] = nodes_df["id"].map(deg_cent).fillna(0.0)

# Betweenness only if graph size is manageable enough for Colab
if len(G) <= 5000:
    btw = nx.betweenness_centrality(G, normalized=True)
else:
    k = min(400, max(50, len(G) // 20))
    btw = nx.betweenness_centrality(G, k=k, normalized=True, seed=0)
nodes_df["betweenness"] = nodes_df["id"].map(btw).fillna(0.0)

# Relation incidence per node
incident_relations = defaultdict(set)
for row in edges_df.itertuples(index=False):
    incident_relations[row.source].add(row.relation)
    incident_relations[row.target].add(row.relation)
nodes_df["incident_relations"] = nodes_df["id"].map(lambda x: sorted(incident_relations.get(x, set())))
nodes_df["incident_relations_str"] = nodes_df["incident_relations"].map(lambda xs: "|".join(xs))

# ----------------------------
# Deterministic absolute origin
# ----------------------------
# Node origin: highest degree, then highest betweenness, then id asc
if ORIGIN_NODE_ID is not None:
    if ORIGIN_NODE_ID not in G:
        raise ValueError(f"ORIGIN_NODE_ID={ORIGIN_NODE_ID} not present in graph.")
    origin_node = ORIGIN_NODE_ID
else:
    candidates = sorted(
        nodes_df["id"].tolist(),
        key=lambda nid: (-deg.get(nid, 0), -btw.get(nid, 0.0), nid)
    )
    origin_node = candidates[0]

# Time origin: explicit override, else median node timestamp if available, else median edge timestamp, else now
if ORIGIN_TIME is not None:
    origin_time = pd.to_datetime(ORIGIN_TIME, utc=True)
else:
    all_ts = pd.concat([nodes_df["timestamp_parsed"], edges_df["timestamp_parsed"]], ignore_index=True)
    all_ts = all_ts.dropna()
    if len(all_ts):
        origin_time = all_ts.sort_values().iloc[len(all_ts) // 2]
    else:
        origin_time = pd.Timestamp.now(tz="UTC")

# Shortest-path shells from origin in undirected G
spl = nx.single_source_shortest_path_length(G, origin_node)
nodes_df["shell"] = nodes_df["id"].map(lambda x: spl.get(x, np.nan))
max_shell = int(np.nanmax(nodes_df["shell"])) if np.isfinite(nodes_df["shell"]).any() else 0

# Fill isolated / unreachable just in case
fill_shell = max_shell + 1 if max_shell >= 0 else 1
nodes_df["shell"] = nodes_df["shell"].fillna(fill_shell).astype(int)

# ----------------------------
# Axis-family-B deterministic surrogate
# ----------------------------
# Contract family B:
# a1 entity_anchor
# a2 relation_or_process_signal
# a3 eventization_signal
# a4 temporal_anchor
#
# We document a pure graph-based surrogate:
# a1: 1 for non-meta/non-gap concrete node classes; 0 for concept/gap/expected/unknown
# a2: 1 if process/policy/process_step or incident to structurally operative relations
# a3: 1 if event/event-like or strongly involvement-linked
# a4: 1 if timestamp >= origin_time, else 0; if missing use hash parity around origin

METAISH = {"concept", "gap", "expected_process", "expected_process_step", "unknown"}
PROCESS_SIGNAL_RELS = {
    "requires_step", "requires_notice", "defines", "assigns_role",
    "reports_to", "reported_to", "overseen_by", "applies_to", "part_of"
}
EVENT_SIGNAL_RELS = {"involves"}

def compute_a1(row):
    return 0 if row["type"] in METAISH else 1

def compute_a2(row):
    if row["type"] in {"process", "process_step", "policy", "role", "office", "org_unit"}:
        return 1
    rels = set(row["incident_relations"])
    if rels & PROCESS_SIGNAL_RELS:
        return 1
    return 1 if row["degree"] >= 3 else 0

def compute_a3(row):
    if row["type"] == "event":
        return 1
    rels = set(row["incident_relations"])
    if "involves" in rels:
        return 1
    s = (row["id"] + " " + str(row.get("description", ""))).lower()
    if any(tok in s for tok in ["event", "meeting", "hearing", "review", "incident", "rca"]):
        return 1
    return 1 if row["type"] in {"process_step"} and row["degree"] >= 2 else 0

def compute_a4(row):
    ts = row["timestamp_parsed"]
    if pd.notna(ts):
        return int(ts >= origin_time)
    return stable_hash_int(row["id"], nbytes=2) % 2

nodes_df["a1_entity_anchor"] = nodes_df.apply(compute_a1, axis=1).astype(int)
nodes_df["a2_relation_or_process_signal"] = nodes_df.apply(compute_a2, axis=1).astype(int)
nodes_df["a3_eventization_signal"] = nodes_df.apply(compute_a3, axis=1).astype(int)
nodes_df["a4_temporal_anchor"] = nodes_df.apply(compute_a4, axis=1).astype(int)

for i, src in enumerate(
    ["a1_entity_anchor", "a2_relation_or_process_signal", "a3_eventization_signal", "a4_temporal_anchor"],
    start=1
):
    nodes_df[f"eps{i}"] = 2 * nodes_df[src] - 1

nodes_df["sigma_bits"] = nodes_df[
    ["a1_entity_anchor", "a2_relation_or_process_signal", "a3_eventization_signal", "a4_temporal_anchor"]
].astype(str).agg("".join, axis=1)

nodes_df["omega_key"] = nodes_df[
    ["eps1", "eps2", "eps3", "eps4"]
].astype(int).astype(str).agg(",".join, axis=1)

# 16 vertex classes
nodes_df["omega_index"] = nodes_df["sigma_bits"].map(lambda s: int(s, 2))

# ----------------------------
# Deterministic x,y,z,w embedding
# ----------------------------
# z and w are explicit exported coordinates.
# x,y are radial chart coordinates derived from:
#   - absolute origin node (shell distance from origin)
#   - absolute origin time (temporal anchor / signed w)
#   - node type sector
#   - deterministic hash tiebreak within sector
#   - light Omega influence without force-layout randomness

# Normalize centrality within component
nodes_df["component_local_degree_rank"] = 0.0
for cid, sub in nodes_df.groupby("component_id"):
    vals = sub["degree_centrality"].to_numpy()
    if len(vals) <= 1 or np.allclose(vals.max(), vals.min()):
        rank = np.zeros(len(sub))
    else:
        rank = (vals - vals.min()) / (vals.max() - vals.min())
    nodes_df.loc[sub.index, "component_local_degree_rank"] = rank

# Type sectors
nodes_df["sector_index"] = nodes_df["type"].map(lambda t: SECTOR_INDEX.get(t, SECTOR_INDEX["unknown"]))
n_sectors = len(SECTOR_ORDER)
nodes_df["sector_center"] = 2 * np.pi * (nodes_df["sector_index"] / n_sectors)

# Type-specific sector width; leaves room between groups
sector_width = 2 * np.pi / n_sectors * 0.72

# Shell-based absolute radius, compressed but still readable
hash_jitter = nodes_df["id"].map(stable_hash_unit)
nodes_df["hash_unit"] = hash_jitter

# z coordinate: signed event axis times log degree shell accent
nodes_df["z"] = nodes_df["eps3"] * (1.0 + np.log1p(nodes_df["degree"]).astype(float))

# w coordinate: signed temporal displacement around origin_time
def time_to_w(ts):
    if pd.isna(ts):
        return np.nan
    dt_days = (ts - origin_time).total_seconds() / 86400.0
    # bounded smooth scale
    return float(np.tanh(dt_days / 30.0))

nodes_df["w"] = nodes_df["timestamp_parsed"].map(time_to_w)
missing_w = nodes_df["w"].isna()
nodes_df.loc[missing_w, "w"] = (nodes_df.loc[missing_w, "eps4"] * (0.15 + 0.35 * nodes_df.loc[missing_w, "hash_unit"]))

# Radius:
#   shell = primary absolute structure from origin node
#   local centrality pulls hubs inward within shell
#   omega_index adds deterministic micro-stratification
nodes_df["r"] = (
    1.25
    + nodes_df["shell"].astype(float)
    + 0.38 * (1.0 - nodes_df["component_local_degree_rank"])
    + 0.08 * (nodes_df["omega_index"] / 15.0)
    + 0.06 * nodes_df["hash_unit"]
)

# Angle:
#   sector center by type
#   hash offset within sector
#   small signed Omega and w perturbation
angle_offset = (nodes_df["hash_unit"] - 0.5) * sector_width
omega_perturb = ((nodes_df["eps1"] + 2*nodes_df["eps2"] - nodes_df["eps3"] + 0.5*nodes_df["eps4"]) / 10.0) * (sector_width / 2.5)
time_perturb = nodes_df["w"] * (sector_width / 6.0)
nodes_df["theta"] = nodes_df["sector_center"] + angle_offset + omega_perturb + time_perturb

nodes_df["x"] = nodes_df["r"] * np.cos(nodes_df["theta"])
nodes_df["y"] = nodes_df["r"] * np.sin(nodes_df["theta"])

# ----------------------------
# Metrics table
# ----------------------------
# Community surrogate: grouped by (component, type, omega_index)
nodes_df["community_id"] = (
    nodes_df["component_id"].astype(str)
    + "::" + nodes_df["type"].astype(str)
    + "::" + nodes_df["omega_index"].astype(str)
)

# Simple local relation entropy
def entropy_from_counts(counts):
    arr = np.asarray([c for c in counts if c > 0], dtype=float)
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log2(p)).sum())

rel_counter = defaultdict(Counter)
for row in edges_df.itertuples(index=False):
    rel_counter[row.source][row.relation] += 1
    rel_counter[row.target][row.relation] += 1

nodes_df["relation_entropy"] = nodes_df["id"].map(lambda nid: entropy_from_counts(rel_counter.get(nid, Counter()).values()))

# Transport proxy: fraction of neighbors in different type sectors/components
nbrs = {nid: list(G.neighbors(nid)) for nid in G.nodes()}
node_type_map = dict(zip(nodes_df["id"], nodes_df["type"]))
node_comp_map = dict(zip(nodes_df["id"], nodes_df["component_id"]))

def transport_score(nid):
    ns = nbrs.get(nid, [])
    if not ns:
        return 0.0
    diffs = 0
    for m in ns:
        if (node_type_map.get(m) != node_type_map.get(nid)) or (node_comp_map.get(m) != node_comp_map.get(nid)):
            diffs += 1
    return diffs / len(ns)

nodes_df["transport_score"] = nodes_df["id"].map(transport_score)

# Defect proxy: mismatch between concrete type and meta-heavy omega bits
defect = []
for row in nodes_df.itertuples(index=False):
    d = 0.0
    if row.type in {"person", "role", "office", "org_unit", "document"} and row.a1_entity_anchor == 0:
        d += 1.0
    if row.type in {"event"} and row.a3_eventization_signal == 0:
        d += 1.0
    if row.type in {"process", "process_step", "policy"} and row.a2_relation_or_process_signal == 0:
        d += 1.0
    if row.degree == 0:
        d += 0.5
    defect.append(d)
nodes_df["defect_score"] = defect

# Label score for display prioritization
nodes_df["label_score"] = (
    1.8 * nodes_df["degree_centrality"]
    + 1.2 * nodes_df["betweenness"]
    + 0.35 * np.log1p(nodes_df["degree"])
    + 0.15 * nodes_df["relation_entropy"]
)

# Node size
nodes_df["node_size"] = (
    6.0
    + 1800.0 * nodes_df["degree_centrality"]
    + 700.0 * nodes_df["betweenness"]
    + 18.0 * np.log1p(nodes_df["degree"])
)
nodes_df["node_size"] = nodes_df["node_size"].clip(8, 2200)

# Font size
if nodes_df["label_score"].max() > nodes_df["label_score"].min():
    score_norm = (nodes_df["label_score"] - nodes_df["label_score"].min()) / (nodes_df["label_score"].max() - nodes_df["label_score"].min())
else:
    score_norm = pd.Series(np.zeros(len(nodes_df)), index=nodes_df.index)
nodes_df["font_size"] = LABEL_MIN_SIZE + (LABEL_MAX_SIZE - LABEL_MIN_SIZE) * score_norm

# Determine displayed labels: top nodes globally + protect origin + one per large sector/component when possible
nodes_df = nodes_df.sort_values(["label_score", "degree", "id"], ascending=[False, False, True]).reset_index(drop=True)
nodes_df["show_label"] = False
label_indices = set(nodes_df.head(MAX_LABELS).index.tolist())

# Ensure origin labeled
origin_idx = nodes_df.index[nodes_df["id"] == origin_node]
if len(origin_idx):
    label_indices.add(int(origin_idx[0]))

# Add a few representatives per type/component
reps = (
    nodes_df.sort_values(["type", "component_id", "label_score", "id"], ascending=[True, True, False, True])
    .groupby(["type", "component_id"], as_index=False)
    .head(1)
    .index
    .tolist()
)
for idx in reps[:MAX_LABELS]:
    label_indices.add(int(idx))

label_indices = sorted(list(label_indices))[:MAX_LABELS]
nodes_df.loc[label_indices, "show_label"] = True

# Colors
nodes_df["node_color_hex"] = nodes_df["type"].map(get_type_color)

def hex_to_rgba(hex_color, alpha=0.95):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (1, 1, 1, alpha)
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, alpha)

nodes_df["node_rgba"] = nodes_df["node_color_hex"].map(lambda c: hex_to_rgba(c, alpha=0.92))

node_xy = nodes_df.set_index("id")[["x", "y"]].to_dict("index")
node_ix = {nid: i for i, nid in enumerate(nodes_df["id"])}

edges_df["src_idx"] = edges_df["source"].map(node_ix)
edges_df["dst_idx"] = edges_df["target"].map(node_ix)
edges_df["edge_rgba"] = edges_df["relation"].map(get_relation_color)

# Slight width emphasis for operative relations
def edge_width(rel):
    if rel in {"reports_to", "part_of", "requires_step", "defines"}:
        return 0.35
    if rel in {"involves", "assigns_role"}:
        return 0.28
    return 0.18

edges_df["edge_width"] = edges_df["relation"].map(edge_width)

# Export shared metric table
export_cols = [
    "id", "type", "label_final", "description", "timestamp",
    "component_id", "component_size", "community_id",
    "degree", "degree_centrality", "betweenness",
    "relation_entropy", "transport_score", "defect_score",
    "a1_entity_anchor", "a2_relation_or_process_signal",
    "a3_eventization_signal", "a4_temporal_anchor",
    "eps1", "eps2", "eps3", "eps4",
    "sigma_bits", "omega_key", "omega_index",
    "sector_index", "shell",
    "x", "y", "z", "w", "r", "theta",
    "node_size", "font_size", "label_score", "show_label",
    "incident_relations_str", "metadata_str"
]
for col in export_cols:
    if col not in nodes_df.columns:
        nodes_df[col] = None

metrics_df = nodes_df[export_cols].copy()
metrics_df.to_csv(OUT_CSV, index=False)

# ----------------------------
# Renderer
# ----------------------------
def render_osiris_graph(nodes_df, edges_df, out_png, out_svg):
    xy = nodes_df[["x", "y"]].to_numpy()

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=600)
    ax.set_aspect("equal")
    ax.axis("off")

    # Shell rings
    rvals = np.linalg.norm(xy, axis=1)
    shell_quantiles = sorted(set(np.quantile(rvals, [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]).round(3).tolist()))
    for q in shell_quantiles:
        ax.add_patch(
            Circle((0, 0), q, fill=False, lw=0.35, ec=(1, 1, 1, 0.07), zorder=0)
        )

    # Sector spokes
    max_r = max(rvals.max(), 1.0) * 1.03
    for t in SECTOR_ORDER:
        theta = 2 * np.pi * (SECTOR_INDEX[t] / n_sectors)
        x2, y2 = max_r * np.cos(theta), max_r * np.sin(theta)
        ax.plot([0, x2], [0, y2], lw=0.22, color=(1, 1, 1, 0.05), zorder=0)

    # Edges
    segs = np.stack([xy[edges_df.src_idx.values], xy[edges_df.dst_idx.values]], axis=1)
    edge_layer = LineCollection(
        segs,
        colors=edges_df["edge_rgba"].to_list(),
        linewidths=edges_df["edge_width"].to_numpy(),
        zorder=1,
    )
    # Uncomment for lighter SVG files:
    # edge_layer.set_rasterized(True)
    ax.add_collection(edge_layer)

    # Nodes
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=nodes_df["node_size"].to_numpy(),
        c=nodes_df["node_rgba"].to_list(),
        linewidths=0,
        zorder=2,
    )

    # Origin marker
    origin_row = nodes_df[nodes_df["id"] == origin_node].iloc[0]
    ax.scatter([0], [0], s=160, c=[(1.0, 1.0, 1.0, 0.92)], linewidths=0, zorder=4)
    ax.text(
        0, 0, "⊙",
        color="white", fontsize=12, ha="center", va="center", zorder=5,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="#0b0f1a")]
    )

    # Labels
    labeled = nodes_df[nodes_df["show_label"]].copy()
    for row in labeled.itertuples(index=False):
        # slight radial offset to reduce overlap
        rr = math.hypot(row.x, row.y)
        ux, uy = (row.x / rr, row.y / rr) if rr > 0 else (0.0, 1.0)
        dx, dy = 0.04 * ux, 0.04 * uy
        t = ax.text(
            row.x + dx, row.y + dy, row.label_final,
            color="white", fontsize=float(row.font_size),
            ha="center", va="center", zorder=3
        )
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="#0b0f1a")])

    # Bounds
    pad = max_r * 0.08 + 0.5
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)

    # Save
    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

render_osiris_graph(nodes_df, edges_df, OUT_PNG, OUT_SVG)

# ----------------------------
# Console summary
# ----------------------------
summary = {
    "artifact_type": G_artifact.get("artifact_type"),
    "nodes": int(len(nodes_df)),
    "edges": int(len(edges_df)),
    "node_types": dict(sorted(Counter(nodes_df["type"]).items(), key=lambda kv: (-kv[1], kv[0]))),
    "relations": dict(sorted(Counter(edges_df["relation"]).items(), key=lambda kv: (-kv[1], kv[0]))),
    "components": int(nodes_df["component_id"].nunique()),
    "origin_node": origin_node,
    "origin_time": str(origin_time),
    "outputs": [OUT_CSV, OUT_PNG, OUT_SVG],
}
print(json.dumps(summary, indent=2, ensure_ascii=False))

# Optional preview in Colab
try:
    from IPython.display import display, Image, SVG
    display(metrics_df.head(20))
    display(Image(filename=OUT_PNG))
except Exception:
    pass