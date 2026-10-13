#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class GRNPriorConfig:
    min_functional_ratio: float = 0.6
    max_nonfunctional_paths: int = 1
    direct_weight_low: float = 0.5
    direct_weight_high: float = 1.0
    indirect_weight_low: float = 0.3
    indirect_weight_high: float = 0.8
    indirect_decay: float = 0.7


def _sample_signed_weight(
    rng: np.random.Generator,
    low: float,
    high: float,
) -> float:
    val = float(rng.uniform(low, high))
    sign = float(rng.choice([-1.0, 1.0]))
    return val * sign


def build_role_map_from_existing_sets(
    tf_class1_ids: List[str],
    tf_class2_ids: List[str],
) -> Dict[str, List[str]]:
    direct = [str(x) for x in tf_class1_ids]
    indirect = [str(x) for x in tf_class2_ids]
    return {
        "direct_functional": direct,
        "indirect_functional": indirect,
        "indirect_nonfunctional": [],
        "noise_tf": [],
    }


def build_local_grn_prior(
    tg_id: str,
    role_map: Dict[str, List[str]],
    rng: np.random.Generator,
    config: GRNPriorConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    cfg = config or GRNPriorConfig()

    tg_id = str(tg_id)
    direct = [str(x) for x in role_map.get("direct_functional", [])]
    indirect_func = [str(x) for x in role_map.get("indirect_functional", [])]
    indirect_nonfunc = [str(x) for x in role_map.get("indirect_nonfunctional", [])]
    noise_tf = [str(x) for x in role_map.get("noise_tf", [])]

    nodes: List[Dict] = [{"node_id": tg_id, "node_type": "TG", "role": "target_gene"}]
    edges: List[Dict] = []
    path_counts: Dict[str, int] = {}

    direct_weights: Dict[str, float] = {}
    for tf_id in direct:
        nodes.append({"node_id": tf_id, "node_type": "TF", "role": "direct_functional"})
        w = _sample_signed_weight(rng, cfg.direct_weight_low, cfg.direct_weight_high)
        direct_weights[tf_id] = w
        edges.append(
            {
                "source": tf_id,
                "target": tg_id,
                "edge_type": "direct_tf_to_tg",
                "source_role": "direct_functional",
                "weight": w,
                "effective_weight_to_tg": w,
                "path_length": 1,
                "is_functional": True,
                "is_noise": False,
            }
        )

    n_direct = len(direct)

    for tf_id in indirect_func:
        nodes.append({"node_id": tf_id, "node_type": "TF", "role": "indirect_functional"})
        if n_direct == 0:
            path_counts[tf_id] = 0
            continue

        num_paths = int(round(rng.normal(loc=4.0, scale=1.0)))
        num_paths = min(n_direct, max(1, num_paths))
        chosen_direct = list(rng.choice(direct, size=num_paths, replace=False))
        base_w = _sample_signed_weight(rng, cfg.indirect_weight_low, cfg.indirect_weight_high)
        path_counts[tf_id] = num_paths

        for d_tf in chosen_direct:
            effective = base_w * cfg.indirect_decay * direct_weights.get(d_tf, 1.0)
            edges.append(
                {
                    "source": tf_id,
                    "target": d_tf,
                    "edge_type": "indirect_tf_to_direct_tf",
                    "source_role": "indirect_functional",
                    "weight": base_w,
                    "effective_weight_to_tg": effective,
                    "path_length": 2,
                    "is_functional": True,
                    "is_noise": False,
                }
            )

    for tf_id in indirect_nonfunc:
        nodes.append({"node_id": tf_id, "node_type": "TF", "role": "indirect_nonfunctional"})
        if n_direct == 0:
            path_counts[tf_id] = 0
            continue

        num_paths = int(rng.integers(0, min(cfg.max_nonfunctional_paths, n_direct) + 1))
        path_counts[tf_id] = num_paths
        if num_paths == 0:
            continue
        chosen_direct = list(rng.choice(direct, size=num_paths, replace=False))
        for d_tf in chosen_direct:
            edges.append(
                {
                    "source": tf_id,
                    "target": d_tf,
                    "edge_type": "indirect_nonfunctional_tf_to_direct_tf",
                    "source_role": "indirect_nonfunctional",
                    "weight": 0.0,
                    "effective_weight_to_tg": 0.0,
                    "path_length": 2,
                    "is_functional": False,
                    "is_noise": True,
                }
            )

    for tf_id in noise_tf:
        nodes.append({"node_id": tf_id, "node_type": "TF", "role": "noise_tf"})
        path_counts[tf_id] = 0

    nodes_df = pd.DataFrame(nodes).drop_duplicates(subset=["node_id", "role"])
    edges_df = pd.DataFrame(edges)

    edge_role_count = {
        "direct_functional": int((edges_df["source_role"] == "direct_functional").sum()) if not edges_df.empty else 0,
        "indirect_functional": int((edges_df["source_role"] == "indirect_functional").sum()) if not edges_df.empty else 0,
        "indirect_nonfunctional": int((edges_df["source_role"] == "indirect_nonfunctional").sum()) if not edges_df.empty else 0,
        "noise_tf": int((edges_df["source_role"] == "noise_tf").sum()) if not edges_df.empty else 0,
    }
    unique_role_count = {
        "direct_functional": len(direct),
        "indirect_functional": len(indirect_func),
        "indirect_nonfunctional": len(indirect_nonfunc),
        "noise_tf": len(noise_tf),
    }

    functional_ratio_ok = all(path_counts.get(tf, 0) >= 1 for tf in indirect_func)

    nonfunctional_path_ok = all(0 <= path_counts.get(tf, 0) <= cfg.max_nonfunctional_paths for tf in indirect_nonfunc)

    summary_df = pd.DataFrame(
        [
            {
                "tg_id": tg_id,
                "num_edges_total": int(len(edges_df)),
                "num_edges_direct_functional": edge_role_count["direct_functional"],
                "num_edges_indirect_functional": edge_role_count["indirect_functional"],
                "num_edges_indirect_nonfunctional": edge_role_count["indirect_nonfunctional"],
                "num_edges_noise_tf": edge_role_count["noise_tf"],
                "num_unique_regulators_total": int(len({*direct, *indirect_func, *indirect_nonfunc, *noise_tf})),
                "num_unique_regulators_direct_functional": unique_role_count["direct_functional"],
                "num_unique_regulators_indirect_functional": unique_role_count["indirect_functional"],
                "num_unique_regulators_indirect_nonfunctional": unique_role_count["indirect_nonfunctional"],
                "num_unique_regulators_noise_tf": unique_role_count["noise_tf"],
                "functional_indirect_path_ratio_ok": bool(functional_ratio_ok),
                "nonfunctional_indirect_path_count_ok": bool(nonfunctional_path_ok),
                "all_constraints_ok": bool(functional_ratio_ok and nonfunctional_path_ok),
            }
        ]
    )

    prior_tf_candidates = sorted({*direct, *indirect_func, *indirect_nonfunc, *noise_tf})
    return nodes_df, edges_df, summary_df, prior_tf_candidates
