from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class OrderMetrics:
    name: str
    apfd: float
    apfdc: float
    bugs_total: int
    bugs_at_10: int
    bugs_at_20: int
    bugs_at_30: int
    bugs_at_50: int


def graphprior_order(
    case_ids: list[str],
    scores: dict[str, float],
    clusters: list[list[str]],
    bug_flags: dict[str, bool],
) -> list[str]:
    cluster_of: dict[str, int] = {}
    for ci, cluster in enumerate(clusters):
        for cid in cluster:
            cluster_of[cid] = ci

    reps: list[str] = []
    members: dict[int, list[str]] = {}
    for ci, cluster in enumerate(clusters):
        rep = max(cluster, key=lambda c: (scores.get(c, 0.0), c))
        reps.append(rep)
        members[ci] = [c for c in cluster if c != rep]

    reps.sort(key=lambda c: (scores.get(c, 0.0), c), reverse=True)
    order: list[str] = []
    deferred: list[str] = []
    visited: set[str] = set()

    for rep in reps:
        if rep in visited:
            continue
        visited.add(rep)
        order.append(rep)
        ci = cluster_of[rep]
        ranked_members = sorted(members[ci], key=lambda c: (scores.get(c, 0.0), c), reverse=True)
        if bug_flags.get(rep, False):
            for c in ranked_members:
                if c not in visited:
                    order.append(c)
                    visited.add(c)
        else:
            for c in ranked_members:
                if c not in visited:
                    deferred.append(c)
                    visited.add(c)

    deferred.sort(key=lambda c: (scores.get(c, 0.0), c), reverse=True)
    order.extend(deferred)

    # Safety fallback if some case was missed due to clustering corner-case.
    for cid in case_ids:
        if cid not in visited:
            order.append(cid)
            visited.add(cid)
    return order


def baseline_orders(
    case_ids: list[str],
    scores: dict[str, float],
    clusters: list[list[str]],
    random_seed: int,
    coverage_baseline_scores: dict[str, float] | None = None,
    coverage_token_map: dict[str, set[str]] | None = None,
) -> dict[str, list[str]]:
    rng = random.Random(random_seed)
    original = list(case_ids)
    random_order = list(case_ids)
    rng.shuffle(random_order)
    cov_scores = coverage_baseline_scores if coverage_baseline_scores is not None else scores
    coverage_only = sorted(case_ids, key=lambda c: (cov_scores.get(c, 0.0), c), reverse=True)
    cluster_of: dict[str, int] = {}
    for ci, cluster in enumerate(clusters):
        for cid in cluster:
            cluster_of[cid] = ci

    # Greedy coverage order: maximize unseen coverage tokens first, then unseen cluster, then OPC score.
    # If no explicit tokens are provided, use the cluster id as a coarse coverage proxy.
    token_map: dict[str, set[str]] = {}
    if coverage_token_map is not None:
        for cid in case_ids:
            token_map[cid] = set(coverage_token_map.get(cid, set()))
    else:
        for cid in case_ids:
            token_map[cid] = {f"cluster:{cluster_of.get(cid, -1)}"}

    remaining = set(case_ids)
    seen_tokens: set[str] = set()
    seen_clusters: set[int] = set()
    coverage_greedy: list[str] = []
    while remaining:
        best = max(
            remaining,
            key=lambda cid: (
                len(token_map.get(cid, set()) - seen_tokens),
                1 if cluster_of.get(cid, -1) not in seen_clusters else 0,
                cov_scores.get(cid, 0.0),
                cid,
            ),
        )
        coverage_greedy.append(best)
        seen_tokens.update(token_map.get(best, set()))
        seen_clusters.add(cluster_of.get(best, -1))
        remaining.remove(best)

    # Degenerate case fallback: if greedy objective collapses to coverage-only ranking,
    # make a minimal early diversification swap.
    if coverage_greedy == coverage_only and len(coverage_greedy) >= 2:
        top_cluster = cluster_of.get(coverage_greedy[0], -1)
        swap_idx = None
        for i in range(1, len(coverage_greedy)):
            if cluster_of.get(coverage_greedy[i], -1) != top_cluster:
                swap_idx = i
                break
        if swap_idx is None:
            swap_idx = 1
        coverage_greedy = list(coverage_greedy)
        coverage_greedy[1], coverage_greedy[swap_idx] = coverage_greedy[swap_idx], coverage_greedy[1]

    reps: list[str] = []
    members: list[str] = []
    case_rank = {cid: i for i, cid in enumerate(case_ids)}
    for cluster in clusters:
        rep = sorted(cluster, key=lambda c: (case_rank.get(c, 10**9), c))[0]
        reps.append(rep)
        rest = [c for c in cluster if c != rep]
        rest.sort(key=lambda c: (case_rank.get(c, 10**9), c))
        members.extend(rest)
    reps.sort(key=lambda c: (case_rank.get(c, 10**9), c))
    structure_only = reps + members

    return {
        "original": original,
        "random": random_order,
        "coverage_greedy": coverage_greedy,
        "coverage_only": coverage_only,
        "structure_only": structure_only,
    }


def _apfd(order: list[str], bug_flags: dict[str, bool]) -> float:
    n = len(order)
    failing_positions = [idx + 1 for idx, cid in enumerate(order) if bug_flags.get(cid, False)]
    m = len(failing_positions)
    if n == 0 or m == 0:
        return 0.0
    return float(1.0 - sum(failing_positions) / (n * m) + 1.0 / (2 * n))


def _apfdc(order: list[str], bug_flags: dict[str, bool], costs: dict[str, float]) -> float:
    total_bugs = sum(1 for cid in order if bug_flags.get(cid, False))
    total_cost = sum(costs.get(cid, 1.0) for cid in order)
    if total_bugs == 0 or total_cost <= 0:
        return 0.0

    x = [0.0]
    y = [0.0]
    cum_cost = 0.0
    cum_bug = 0
    for cid in order:
        cum_cost += costs.get(cid, 1.0)
        if bug_flags.get(cid, False):
            cum_bug += 1
        x.append(cum_cost / total_cost)
        y.append(cum_bug / total_bugs)

    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return float(area)


def _bugs_under_budget(order: list[str], bug_flags: dict[str, bool], budget_ratio: float) -> int:
    n = len(order)
    if n == 0:
        return 0
    k = max(1, int(n * budget_ratio))
    return int(sum(1 for cid in order[:k] if bug_flags.get(cid, False)))


def evaluate_order(
    name: str,
    order: list[str],
    bug_flags: dict[str, bool],
    costs: dict[str, float],
) -> OrderMetrics:
    return OrderMetrics(
        name=name,
        apfd=_apfd(order, bug_flags),
        apfdc=_apfdc(order, bug_flags, costs),
        bugs_total=int(sum(1 for cid in order if bug_flags.get(cid, False))),
        bugs_at_10=_bugs_under_budget(order, bug_flags, 0.10),
        bugs_at_20=_bugs_under_budget(order, bug_flags, 0.20),
        bugs_at_30=_bugs_under_budget(order, bug_flags, 0.30),
        bugs_at_50=_bugs_under_budget(order, bug_flags, 0.50),
    )


def bug_concentration(clusters: list[list[str]], bug_flags: dict[str, bool]) -> dict[str, float]:
    cluster_bug_counts = [sum(1 for cid in cluster if bug_flags.get(cid, False)) for cluster in clusters]
    total_bugs = sum(cluster_bug_counts)
    if total_bugs == 0 or not clusters:
        return {"bug_clusters_ratio": 0.0, "top20_cluster_bug_share": 0.0}

    nonzero_clusters = sum(1 for x in cluster_bug_counts if x > 0)
    bug_clusters_ratio = nonzero_clusters / len(clusters)

    pairs = sorted(zip(cluster_bug_counts, [len(c) for c in clusters]), key=lambda x: x[0], reverse=True)
    top_k = max(1, int(0.2 * len(pairs)))
    top_bug = sum(x[0] for x in pairs[:top_k])
    return {
        "bug_clusters_ratio": float(bug_clusters_ratio),
        "top20_cluster_bug_share": float(top_bug / total_bugs),
    }
