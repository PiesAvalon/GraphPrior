from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from dataclasses import dataclass

import networkx as nx
import numpy as np

from ._legacy_types import NodeSpec, TestCase


@dataclass
class GraphData:
    case_id: str
    project: str
    graph: nx.DiGraph
    op_counter: Counter[tuple[str, int, int]]
    osc_codes: dict[int, set[tuple]]
    usc_codes: dict[int, set[tuple]]
    plc_set: set[int]
    coc_set: set[tuple[str, str]]
    wl_features: Counter[str]


@dataclass
class PatternSpaces:
    omega_op: set[tuple[str, int, int]]
    omega_o: dict[int, set[tuple]]
    omega_u: dict[int, set[tuple]]
    nmax_op: dict[tuple[str, int, int], int]
    omega_pl: set[int]
    omega_co: set[tuple[str, str]]


def _shape_signature(shape) -> str:
    if shape is None:
        return "na"
    if isinstance(shape, (list, tuple)):
        parts: list[str] = []
        for dim in shape:
            if isinstance(dim, (list, tuple)):
                parts.append(f"[{_shape_signature(dim)}]")
            elif dim is None:
                parts.append("?")
            else:
                parts.append(str(dim))
        return "x".join(parts)
    return str(shape)


def _dtype_signature(dtype) -> str:
    if dtype is None:
        return "na"
    return str(dtype).replace("torch.", "").replace("tf.", "")


def node_label(node: NodeSpec) -> str:
    dtype = _dtype_signature(node.attrs.get("dtype"))
    input_shape = _shape_signature(node.attrs.get("input_shape"))
    output_shape = _shape_signature(node.attrs.get("output_shape"))
    if node.op == "conv2d":
        out_channels = int(node.attrs["out_channels"])
        kernel = tuple(node.attrs["kernel_size"])
        stride = tuple(node.attrs["stride"])
        padding = node.attrs.get("padding", "valid")
        groups = int(node.attrs.get("groups", 1))
        return (
            f"conv2d:o{out_channels}:k{kernel[0]}x{kernel[1]}:s{stride[0]}:"
            f"p{padding}:g{groups}:dt{dtype}:in{input_shape}:out{output_shape}"
        )
    if node.op == "linear":
        out_dim = int(node.attrs["out_dim"])
        return f"linear:o{out_dim}:dt{dtype}:in{input_shape}:out{output_shape}"
    if node.op == "batchnorm":
        return f"batchnorm:c{int(node.attrs['num_features'])}:dt{dtype}:out{output_shape}"
    if node.op == "maxpool2d":
        kernel = tuple(node.attrs["kernel_size"])
        stride = tuple(node.attrs["stride"])
        padding = node.attrs.get("padding", 0)
        return f"maxpool2d:k{kernel[0]}x{kernel[1]}:s{stride[0]}:p{padding}:dt{dtype}:out{output_shape}"
    if node.op == "global_avg_pool":
        return f"global_avg_pool:dt{dtype}:out{output_shape}"
    if node.op == "flatten":
        return f"flatten:dt{dtype}:out{output_shape}"
    if node.attrs:
        return f"{node.op}:dt{dtype}:in{input_shape}:out{output_shape}"
    return node.op


def build_graph(case: TestCase) -> nx.DiGraph:
    g = nx.DiGraph()
    for node in case.nodes:
        g.add_node(node.node_id, label=node_label(node), op=node.op)
    for node in case.nodes:
        for src in node.inputs:
            if src >= 0:
                g.add_edge(src, node.node_id)
    if not nx.is_directed_acyclic_graph(g):
        # Mutated tests should still be analyzable; drop back-edges if any exist.
        g = nx.DiGraph([(u, v) for u, v in g.edges() if u < v])
        for node in case.nodes:
            if node.node_id not in g:
                g.add_node(node.node_id, label=node_label(node), op=node.op)
    return g


def _op_counter(g: nx.DiGraph) -> Counter[tuple[str, int, int]]:
    counter: Counter[tuple[str, int, int]] = Counter()
    for n in g.nodes():
        label = str(g.nodes[n]["label"])
        sig = (label, int(g.in_degree(n)), int(g.out_degree(n)))
        counter[sig] += 1
    return counter


def _connected_induced_subgraphs(g: nx.DiGraph, k: int, max_subgraphs: int | None = None) -> list[nx.DiGraph]:
    nodes = list(g.nodes())
    res: list[nx.DiGraph] = []
    for combo in itertools.combinations(nodes, k):
        sg = g.subgraph(combo).copy()
        if k == 1 or nx.is_connected(sg.to_undirected()):
            res.append(sg)
            if max_subgraphs is not None and len(res) >= max_subgraphs:
                break
    return res


def _stable_case_seed(base_seed: int, case_id: str) -> int:
    digest = hashlib.blake2b(f"{base_seed}:{case_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _sample_neighbor_pairs(
    neighbors: list[int],
    sample_size: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    d = len(neighbors)
    total_pairs = d * (d - 1) // 2
    if total_pairs == 0 or sample_size <= 0:
        return []

    if sample_size >= total_pairs:
        return [(neighbors[i], neighbors[j]) for i in range(d) for j in range(i + 1, d)]

    sampled_index_pairs: set[tuple[int, int]] = set()
    attempts = 0
    # Keep retries bounded to avoid long tails on high-collision sampling.
    max_attempts = max(64, sample_size * 20)
    while len(sampled_index_pairs) < sample_size and attempts < max_attempts:
        i, j = rng.choice(d, size=2, replace=False)
        if i > j:
            i, j = j, i
        sampled_index_pairs.add((int(i), int(j)))
        attempts += 1

    if len(sampled_index_pairs) < sample_size:
        for i in range(d):
            for j in range(i + 1, d):
                sampled_index_pairs.add((i, j))
                if len(sampled_index_pairs) >= sample_size:
                    break
            if len(sampled_index_pairs) >= sample_size:
                break

    return [(neighbors[i], neighbors[j]) for i, j in sorted(sampled_index_pairs)]


def _sample_connected_triplets(
    g: nx.DiGraph,
    budget_per_node: int,
    rng: np.random.Generator,
    max_triplets: int | None = None,
) -> list[tuple[int, int, int]]:
    if budget_per_node <= 0:
        return []

    ug = g.to_undirected(as_view=True)
    triplets: set[tuple[int, int, int]] = set()
    for center in sorted(ug.nodes()):
        neighbors = sorted(int(n) for n in ug.neighbors(center))
        if len(neighbors) < 2:
            continue

        total_pairs = len(neighbors) * (len(neighbors) - 1) // 2
        local_budget = min(budget_per_node, total_pairs)
        for u, w in _sample_neighbor_pairs(neighbors, local_budget, rng):
            combo = tuple(sorted((int(center), int(u), int(w))))
            triplets.add(combo)
            if max_triplets is not None and len(triplets) >= max_triplets:
                return sorted(triplets)
    return sorted(triplets)


def _ordered_code(sg: nx.DiGraph) -> tuple:
    labels = {n: (str(sg.nodes[n]["label"]), int(sg.in_degree(n)), int(sg.out_degree(n))) for n in sg.nodes()}
    if sg.number_of_nodes() == 1:
        n = next(iter(sg.nodes()))
        return (labels[n], (0,))

    encodings: list[tuple] = []
    try:
        topo_orders = list(nx.all_topological_sorts(sg))
    except Exception:
        topo_orders = [sorted(sg.nodes())]

    for order in topo_orders:
        sigs = tuple(labels[n] for n in order)
        adj_bits: list[int] = []
        for i in range(len(order)):
            for j in range(len(order)):
                adj_bits.append(1 if sg.has_edge(order[i], order[j]) else 0)
        encodings.append((sigs, tuple(adj_bits)))
    return min(encodings)


def _unordered_code(sg: nx.DiGraph) -> tuple:
    labels = {n: (str(sg.nodes[n]["label"]), int(sg.in_degree(n)), int(sg.out_degree(n))) for n in sg.nodes()}
    nodes = list(sg.nodes())
    if len(nodes) == 1:
        n = nodes[0]
        return (labels[n], (0,))

    best: tuple | None = None
    for order in itertools.permutations(nodes):
        sigs = tuple(labels[n] for n in order)
        adj_bits: list[int] = []
        for i in range(len(order)):
            for j in range(len(order)):
                adj_bits.append(1 if sg.has_edge(order[i], order[j]) else 0)
        cand = (sigs, tuple(adj_bits))
        if best is None or cand < best:
            best = cand
    assert best is not None
    return best


def _path_length_set(g: nx.DiGraph, max_paths: int = 256) -> set[int]:
    sources = [n for n in g.nodes() if g.in_degree(n) == 0]
    sinks = [n for n in g.nodes() if g.out_degree(n) == 0]
    lengths: set[int] = set()
    total_paths = 0
    for s in sources:
        for t in sinks:
            if s == t:
                lengths.add(0)
                continue
            try:
                for p in nx.all_simple_paths(g, source=s, target=t):
                    total_paths += 1
                    lengths.add(len(p) - 1)
                    if total_paths >= max_paths:
                        return lengths
            except nx.NetworkXNoPath:
                continue
    return lengths


def _coc_set(g: nx.DiGraph, radius: int = 2) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for u in g.nodes():
        uop = str(g.nodes[u]["op"])
        queue: list[tuple[int, int]] = [(u, 0)]
        visited: set[int] = {u}
        while queue:
            cur, dist = queue.pop(0)
            if dist >= radius:
                continue
            for nxt in g.successors(cur):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
                if 1 <= dist + 1 <= radius:
                    vop = str(g.nodes[nxt]["op"])
                    pairs.add((uop, vop))
    return pairs


def wl_features(g: nx.DiGraph, h: int = 2) -> Counter[str]:
    labels: dict[int, str] = {n: str(g.nodes[n]["label"]) for n in g.nodes()}
    feat: Counter[str] = Counter(labels.values())
    for _ in range(h):
        new_labels: dict[int, str] = {}
        for n in g.nodes():
            pred = sorted(labels[p] for p in g.predecessors(n))
            succ = sorted(labels[s] for s in g.successors(n))
            key = f"{labels[n]}|p:{'|'.join(pred)}|s:{'|'.join(succ)}"
            new_labels[n] = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        labels = new_labels
        feat.update(labels.values())
    return feat


def simhash(features: Counter[str], bits: int = 64) -> int:
    acc = [0.0] * bits
    for token, value in features.items():
        val = float(value)
        for j in range(bits):
            digest = hashlib.blake2b(f"{token}:{j}".encode("utf-8"), digest_size=8).digest()
            bit = digest[-1] & 1
            acc[j] += val if bit else -val

    sig = 0
    for j, score in enumerate(acc):
        if score >= 0:
            sig |= 1 << j
    return sig


def build_graph_data(
    cases: list[TestCase],
    wl_h: int = 2,
    k3_sample_budget_per_node: int = 16,
    k3_sample_seed: int = 2026,
    k3_max_triplets: int | None = None,
) -> list[GraphData]:
    data: list[GraphData] = []
    for case in cases:
        g = build_graph(case)
        op_counter = _op_counter(g)
        osc = {1: set(), 2: set(), 3: set()}
        usc = {1: set(), 2: set(), 3: set()}

        # k=1 connected induced subgraphs are individual nodes.
        for node in g.nodes():
            sg = g.subgraph((node,)).copy()
            osc[1].add(_ordered_code(sg))
            usc[1].add(_unordered_code(sg))

        # k=2 connected induced subgraphs are exactly undirected edges.
        for u, v in g.to_undirected(as_view=True).edges():
            sg = g.subgraph((u, v)).copy()
            osc[2].add(_ordered_code(sg))
            usc[2].add(_unordered_code(sg))

        # k=3 connected induced subgraphs are estimated via wedge sampling.
        rng = np.random.default_rng(_stable_case_seed(k3_sample_seed, case.case_id))
        for triplet in _sample_connected_triplets(
            g,
            budget_per_node=k3_sample_budget_per_node,
            rng=rng,
            max_triplets=k3_max_triplets,
        ):
            sg = g.subgraph(triplet).copy()
            osc[3].add(_ordered_code(sg))
            usc[3].add(_unordered_code(sg))

        data.append(
            GraphData(
                case_id=case.case_id,
                project=case.project,
                graph=g,
                op_counter=op_counter,
                osc_codes=osc,
                usc_codes=usc,
                plc_set=_path_length_set(g),
                coc_set=_coc_set(g, radius=2),
                wl_features=wl_features(g, h=wl_h),
            )
        )
    return data


def build_pattern_spaces(graph_data: list[GraphData]) -> PatternSpaces:
    omega_op: set[tuple[str, int, int]] = set()
    omega_o: dict[int, set[tuple]] = {1: set(), 2: set(), 3: set()}
    omega_u: dict[int, set[tuple]] = {1: set(), 2: set(), 3: set()}
    omega_pl: set[int] = set()
    omega_co: set[tuple[str, str]] = set()
    nmax_op: dict[tuple[str, int, int], int] = {}

    for gd in graph_data:
        omega_op.update(gd.op_counter.keys())
        for sig, cnt in gd.op_counter.items():
            nmax_op[sig] = max(nmax_op.get(sig, 0), int(cnt))
        for k in (1, 2, 3):
            omega_o[k].update(gd.osc_codes[k])
            omega_u[k].update(gd.usc_codes[k])
        omega_pl.update(gd.plc_set)
        omega_co.update(gd.coc_set)

    return PatternSpaces(
        omega_op=omega_op,
        omega_o=omega_o,
        omega_u=omega_u,
        nmax_op=nmax_op,
        omega_pl=omega_pl,
        omega_co=omega_co,
    )


def _safe_ratio(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(numer / denom)


def compute_coverage_scores(
    graph_data: list[GraphData],
    spaces: PatternSpaces,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    scores: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}
    eta = {1: 1.0, 2: 1.0, 3: 1.0}
    zeta = {1: 1.0, 2: 1.0, 3: 1.0}
    beta = 0.75

    for gd in graph_data:
        op_present = set(gd.op_counter.keys())
        opc = _safe_ratio(len(op_present), len(spaces.omega_op))

        osc_k = {}
        usc_k = {}
        for k in (1, 2, 3):
            osc_k[k] = _safe_ratio(len(gd.osc_codes[k]), len(spaces.omega_o[k]))
            usc_k[k] = _safe_ratio(len(gd.usc_codes[k]), len(spaces.omega_u[k]))

        osc = _safe_ratio(sum(eta[k] * osc_k[k] for k in (1, 2, 3)), sum(eta.values()))
        usc = _safe_ratio(sum(zeta[k] * usc_k[k] for k in (1, 2, 3)), sum(zeta.values()))

        nrc_num = 0.0
        nrc_den = 0.0
        for pattern in spaces.omega_op:
            n = min(int(gd.op_counter.get(pattern, 0)), int(spaces.nmax_op.get(pattern, 1)))
            nmax = int(spaces.nmax_op.get(pattern, 1))
            gain = 1.0 - math.exp(-beta * n)
            gain_max = 1.0 - math.exp(-beta * nmax)
            nrc_num += gain
            nrc_den += gain_max
        nrc = _safe_ratio(nrc_num, nrc_den)

        plc = _safe_ratio(len(gd.plc_set), len(spaces.omega_pl))
        coc = _safe_ratio(len(gd.coc_set), len(spaces.omega_co))

        c_operator = (opc + nrc) / 2.0
        c_subgraph = (osc + usc + coc) / 3.0
        c_global = plc
        ocs = (c_operator + c_subgraph + c_global) / 3.0

        scores[gd.case_id] = float(ocs)
        metrics[gd.case_id] = {
            "opc": float(opc),
            "osc": float(osc),
            "usc": float(usc),
            "nrc": float(nrc),
            "plc": float(plc),
            "coc": float(coc),
            "ocs": float(ocs),
        }
    return scores, metrics


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def lsh_cluster(
    case_ids: list[str],
    signatures: dict[str, int],
    bits: int = 64,
    bands: int = 8,
    min_band_collisions: int = 2,
) -> list[list[str]]:
    if bits % bands != 0:
        raise ValueError("bits must be divisible by bands")

    rows = bits // bands
    uf = UnionFind(case_ids)
    buckets: dict[tuple[int, int], list[str]] = {}
    pair_hits: dict[tuple[str, str], int] = {}
    mask = (1 << rows) - 1

    for cid in case_ids:
        sig = signatures[cid]
        for band in range(bands):
            key_bits = (sig >> (band * rows)) & mask
            key = (band, key_bits)
            buckets.setdefault(key, []).append(cid)

    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a = members[i]
                b = members[j]
                pair = (a, b) if a < b else (b, a)
                pair_hits[pair] = pair_hits.get(pair, 0) + 1

    for (a, b), hit in pair_hits.items():
        if hit >= min_band_collisions:
            uf.union(a, b)

    groups: dict[str, list[str]] = {}
    for cid in case_ids:
        root = uf.find(cid)
        groups.setdefault(root, []).append(cid)

    clusters = [sorted(v) for v in groups.values()]
    clusters.sort(key=lambda x: (len(x), x[0]), reverse=True)
    return clusters


def graphprior_analysis(
    cases: list[TestCase],
    wl_h: int = 2,
    simhash_bits: int = 64,
    lsh_bands: int = 8,
    lsh_min_collisions: int = 2,
    k3_sample_budget_per_node: int = 16,
    k3_sample_seed: int = 2026,
    k3_max_triplets: int | None = None,
) -> tuple[
    list[GraphData],
    dict[str, float],
    dict[str, dict[str, float]],
    list[list[str]],
]:
    graph_data = build_graph_data(
        cases,
        wl_h=wl_h,
        k3_sample_budget_per_node=k3_sample_budget_per_node,
        k3_sample_seed=k3_sample_seed,
        k3_max_triplets=k3_max_triplets,
    )
    spaces = build_pattern_spaces(graph_data)
    scores, metrics = compute_coverage_scores(graph_data, spaces)
    signatures = {gd.case_id: simhash(gd.wl_features, bits=simhash_bits) for gd in graph_data}
    clusters = lsh_cluster(
        [c.case_id for c in cases],
        signatures,
        bits=simhash_bits,
        bands=lsh_bands,
        min_band_collisions=lsh_min_collisions,
    )
    return graph_data, scores, metrics, clusters
