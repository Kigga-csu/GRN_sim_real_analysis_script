#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg as la
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
GRN_DIR = PROJECT_ROOT / "GRN_data_simulation"
if str(GRN_DIR) not in sys.path:
    sys.path.insert(0, str(GRN_DIR))

import simulation_step1_core as sim  # noqa: E402
from simulation_grn_prior_builder import (  # noqa: E402
    GRNPriorConfig,
    build_local_grn_prior,
    build_role_map_from_existing_sets,
)


def _save_pickle(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def sim_geno(geno, n):
    if not isinstance(geno, np.ndarray) or geno.ndim != 2:
        raise ValueError("geno must be a 2-D NumPy array (SNP x individual)")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    G = geno.copy()
    if np.any(np.isnan(G)):
        for i in range(G.shape[0]):
            row = G[i, :]
            G[i, np.isnan(row)] = np.nanmean(row)

    G = StandardScaler().fit_transform(G)
    if np.any(np.isnan(G)):
        raise ValueError("genotype matrix contains NaN after scaling")

    LD = (G @ G.T) / G.shape[1] + np.eye(G.shape[0]) * 1e-6
    try:
        L = la.cholesky(LD, lower=True)
    except la.LinAlgError:
        raise ValueError("LD matrix is not positive definite; Cholesky failed")

    Z = L @ np.random.normal(size=(L.shape[0], n))
    Z = (Z - np.mean(Z, axis=1, keepdims=True)) / np.std(Z, axis=1, keepdims=True)
    if np.any(np.isnan(Z)):
        raise ValueError("simulated genotype matrix contains NaN")
    return Z


def sim_trait_final(cis_g, trans_g, h2g_expr, prop_pheno_exp, p_snps):
    g_expr = np.asarray(cis_g, dtype=float) + np.asarray(trans_g, dtype=float)
    n = len(g_expr)
    if n == 0:
        raise ValueError("empty g_expr")

    s2_orig = np.var(g_expr, ddof=1)
    if h2g_expr > 0 and s2_orig > 1e-8:
        g = g_expr * np.sqrt(h2g_expr / s2_orig)
    else:
        g = np.zeros(n)

    h2g = h2g_expr / prop_pheno_exp if 0 < prop_pheno_exp <= 1.0 else h2g_expr
    if h2g > 1.0:
        h2g = 1.0
    h2p = max(0.0, h2g - h2g_expr)

    p = np.zeros(n)
    if p_snps is not None and p_snps.shape[1] > 0 and h2p > 0:
        n_p = p_snps.shape[1]
        beta_p = np.random.normal(0, np.sqrt(h2p / n_p), size=n_p)
        p = p_snps @ beta_p

    e = np.random.normal(0, np.sqrt(max(0.0, 1.0 - h2g)), size=n)
    y = g + p + e
    sd = np.std(y, ddof=1)
    return (y - y.mean()) / sd if sd > 1e-8 else np.zeros(n)


def gwas_summary_stats(geno_df, trait):
    G = geno_df.values.astype(float)
    y = np.asarray(trait, dtype=float)
    n = G.shape[0]
    Gc = G - G.mean(0, keepdims=True)
    yc = y - y.mean()
    denom = (Gc ** 2).sum(0)
    denom[denom == 0] = 1.0
    beta = (Gc * yc[:, None]).sum(0) / denom
    resid = yc[:, None] - Gc * beta
    s2 = (resid ** 2).sum(0) / max(n - 2, 1)
    se = np.sqrt(s2 / denom)
    se[se == 0] = np.nan
    z = beta / se
    p = 2.0 * norm.sf(np.abs(z))
    return pd.DataFrame({"beta": beta, "se": se, "z": z, "p": p})


def _sim_geno_df(geno_df, n, sample_cols):
    mat = sim_geno(geno_df[sample_cols].values, n).T
    sim_df = pd.DataFrame(mat, columns=geno_df["snpID"].values)
    meta = geno_df[["CHROM", "POS", "snpID"]].drop_duplicates("snpID").reset_index(drop=True)
    return sim_df, meta


def _dedup_cols(df):
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def _resolve_h2p(params):
    prop = float(getattr(params, "prop_pheno_exp", 1.0))
    h2g_expr = float(params.h_cis) + float(params.h_trans)
    h2p = h2g_expr * (1.0 / prop - 1.0) if 0 < prop < 1.0 else 0.0
    return h2g_expr, max(h2p, 0.0)


def process_param_set(idx, param_set, out_root, n_reps, n_gwas, base_seed,
                      grn_cfg, grn_force_flat_prior=False, write_individual_genotype=False):
    param_dir = Path(out_root) / f"param_set_{idx}"
    param_dir.mkdir(parents=True, exist_ok=True)

    params = sim.SimulationParams(**param_set)
    selection_seed = base_seed + idx * 1000
    gene_data = sim.extract_gene_data(params, str(param_dir), selection_seed=selection_seed)
    if not gene_data or not gene_data.get("tg_id"):
        print(f"[ps{idx}] ERROR: gene data extraction failed; skipping.")
        return

    _save_pickle(param_dir / "gene_data.pkl", gene_data)
    with open(param_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(param_set, f, indent=2, ensure_ascii=False)

    tg_id = gene_data["tg_id"]
    tf_genos = gene_data["tf_genos"]
    tg_geno = gene_data["tg_geno"]
    real_sample_ids = gene_data["sampleID"]
    h2g_expr, h2p = _resolve_h2p(params)
    print(f"[ps{idx}] tg={tg_id} h2g_expr={h2g_expr:.4f} h2p={h2p:.4f} prop={params.prop_pheno_exp}")

    for rep in range(n_reps):
        rep_seed = base_seed + idx * 1000 + rep
        rep_dir = param_dir / f"rep_{rep:04d}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(rep_seed)

        (
            tg_expressions,
            tf_expressions,
            all_tf_ids,
            tf_class1_ids,
            tf_class2_ids,
            _tf_genos,
            _tg_geno,
            tg_4_real_tf,
            cis_weights,
            trans_weights,
            tg_id_ret,
        ) = sim.generate_expression(
            params, str(rep_dir), gene_data, expression_seed=rep_seed,
            direct_trans_pool_geno=None,
        )
        tg_id = str(tg_id_ret)

        real_tf_ids = [str(x) for x in tg_4_real_tf.get(tg_id, [])]
        role_map = build_role_map_from_existing_sets(
            all_tf_ids=[str(x) for x in all_tf_ids],
            tf_class1_ids=[str(x) for x in tf_class1_ids],
            tf_class2_ids=[str(x) for x in tf_class2_ids],
            direct_functional_ids=real_tf_ids,
        )
        if grn_force_flat_prior:
            role_map = {
                "direct_functional": sorted(set(role_map["direct_functional"] + role_map["indirect_functional"])),
                "indirect_functional": [],
                "indirect_nonfunctional": [],
                "noise_tf": role_map["noise_tf"],
            }
        grn_nodes, grn_edges, grn_summary, prior_tf_candidates = build_local_grn_prior(
            tg_id=tg_id, role_map=role_map, rng=rng, config=grn_cfg,
        )
        grn_nodes.to_csv(rep_dir / "grn_nodes.tsv", sep="\t", index=False)
        grn_edges.to_csv(rep_dir / "grn_edges.tsv", sep="\t", index=False)
        grn_summary.to_csv(rep_dir / "grn_summary.tsv", sep="\t", index=False)

        stage1_payload = {
            "rep": rep, "rep_seed": rep_seed,
            "all_tg_expressions": pd.concat(tg_expressions, axis=0, ignore_index=True),
            "all_tf_expressions": pd.concat(tf_expressions, axis=0, ignore_index=True),
            "all_tf_ids": [str(x) for x in all_tf_ids],
            "tf_class1_ids": [str(x) for x in tf_class1_ids],
            "tf_class2_ids": [str(x) for x in tf_class2_ids],
            "tg_4_real_tf": {str(k): [str(v) for v in vals] for k, vals in tg_4_real_tf.items()},
            "cis_weights": cis_weights,
            "trans_weights": trans_weights,
            "tg_id": tg_id,
            "grn_role_map": role_map,
            "prior_tf_candidates": [str(x) for x in prior_tf_candidates],
            "grn_summary": grn_summary.to_dict(orient="records")[0],
        }
        _save_pickle(rep_dir / "sim_payload.pkl", stage1_payload)

        sim_tf_genos, tf_meta_by_id = {}, {}
        for tf_i, gdf in tf_genos.items():
            if gdf is None or gdf.empty:
                continue
            sdf, meta = _sim_geno_df(gdf, n_gwas, real_sample_ids)
            sim_tf_genos[tf_i] = sdf
            tf_meta_by_id[tf_i] = meta
        sim_tg_geno, tg_meta = _sim_geno_df(tg_geno, n_gwas, real_sample_ids)

        gwas_sample_ids = [f"SIM{i:06d}" for i in range(n_gwas)]

        selected_real_tf = {t: sim_tf_genos[t] for t in real_tf_ids if t in sim_tf_genos}
        tg_snp_df = _dedup_cols(sim_tg_geno.copy())
        real_tf_snp_df = _dedup_cols(pd.concat(list(selected_real_tf.values()), axis=1)) \
            if selected_real_tf else pd.DataFrame(index=range(n_gwas))

        tg_expr_snps = tg_snp_df.columns.intersection(cis_weights["snpID"])
        tf_expr_snps = real_tf_snp_df.columns.intersection(trans_weights["snpID"]) \
            if not real_tf_snp_df.empty else pd.Index([])

        sim_expr_cis = np.dot(
            tg_snp_df[tg_expr_snps].values,
            cis_weights.set_index("snpID").loc[tg_expr_snps, "beta"].values,
        )
        if len(tf_expr_snps) > 0:
            sim_expr_trans = np.dot(
                real_tf_snp_df[tf_expr_snps].values,
                trans_weights.set_index("snpID").loc[tf_expr_snps, "beta"].values,
            )
        else:
            sim_expr_trans = np.zeros(n_gwas)

        scan = getattr(params, "scan", "mix_casual")
        if scan.startswith("trans_"):
            sim_expr_cis = np.zeros(n_gwas)
        elif scan.startswith("cis_"):
            sim_expr_trans = np.zeros(n_gwas)

        pleio_parts = [sim_tg_geno]
        pleio_metas = [tg_meta]
        for t in real_tf_ids:
            if t in sim_tf_genos:
                pleio_parts.append(sim_tf_genos[t])
                pleio_metas.append(tf_meta_by_id[t])
        pleio_geno_df = _dedup_cols(pd.concat(pleio_parts, axis=1))
        pleio_meta = pd.concat(pleio_metas, ignore_index=True).drop_duplicates("snpID").reset_index(drop=True)
        pleio_geno = pleio_geno_df.values.astype(float)

        annot = pleio_meta.copy()
        annot["is_expr_cis"] = annot["snpID"].isin(set(tg_expr_snps))
        annot["is_expr_trans"] = annot["snpID"].isin(set(tf_expr_snps))
        annot["is_pleio"] = True
        annot.to_csv(rep_dir / "snp_annotation.tsv", sep="\t", index=False)

        if write_individual_genotype:
            geno_out = pleio_geno_df.T.copy()
            geno_out.columns = gwas_sample_ids
            geno_out.insert(0, "snpID", list(pleio_geno_df.columns))
            geno_out.to_csv(rep_dir / "gwas_genotype.tsv", sep="\t", index=False)
        pd.Series(gwas_sample_ids, name="sample_id").to_csv(rep_dir / "gwas_samples.tsv", sep="\t", index=False)

        trait = sim_trait_final(
            cis_g=sim_expr_cis, trans_g=sim_expr_trans,
            h2g_expr=h2g_expr, prop_pheno_exp=params.prop_pheno_exp,
            p_snps=pleio_geno,
        )

        pd.DataFrame({"sample_id": gwas_sample_ids, "trait": trait}).to_csv(
            rep_dir / "trait.tsv", sep="\t", index=False)

        stats = gwas_summary_stats(pleio_geno_df, trait)
        stats.insert(0, "snpID", list(pleio_geno_df.columns))
        stats = stats.merge(pleio_meta, on="snpID", how="left")
        stats = stats[["CHROM", "POS", "snpID", "beta", "se", "z", "p"]]
        stats.to_csv(rep_dir / "gwas_summary.tsv", sep="\t", index=False)

        payload = {
            "rep": rep, "rep_seed": rep_seed, "tg_id": tg_id,
            "h2g_expr": h2g_expr, "h2p": h2p, "prop_pheno_exp": params.prop_pheno_exp,
            "n_gwas": n_gwas, "gwas_sample_ids": gwas_sample_ids,
            "real_tf_ids": real_tf_ids,
            "tg_expr_snps": [str(s) for s in tg_expr_snps],
            "tf_expr_snps": [str(s) for s in tf_expr_snps],
            "pleio_snps": [str(s) for s in pleio_geno_df.columns],
            "snp_meta": pleio_meta,
            "trait": trait,
            "gwas_beta": stats["beta"].values,
            "gwas_p": stats["p"].values,
            "cis_weights": cis_weights, "trans_weights": trans_weights,
        }
        _save_pickle(rep_dir / "trait_gwas_payload.pkl", payload)
        print(f"[ps{idx}] rep {rep} done: n_gwas={n_gwas} n_pleio_snps={pleio_geno_df.shape[1]}")


def main():
    ap = argparse.ArgumentParser(description="Stage 2: trait + GWAS data generation "
                                             "(expression + GRN reused from GRN_data_simulation)")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--geno-path", default="/data/lab/wangshixian/data/genotype/1000G/1000G_Phase3_30X/")
    ap.add_argument("--bed-path", default="/data/lab/wangshixian/GRNTWAS_STAR/data/anno_info/gene.bed")
    ap.add_argument("--sample-path", default=str(PROJECT_ROOT / "sample_names" / "EUR.txt"))
    ap.add_argument("--tf-info-path", default="/data/lab/wangshixian/data/GRN/TFlist_human/Homo_sapiens_TF.txt")
    ap.add_argument("--scan", default="mix_pleiotropy", choices=[
        "cis_casual", "trans_casual", "mix_casual", "cis_pleiotropy", "trans_pleiotropy", "mix_pleiotropy"])
    ap.add_argument("--n-reps", type=int, default=5)
    ap.add_argument("--param-limit", type=int, default=0, help="use first N param sets (0 = all)")
    ap.add_argument("--n-gwas", type=int, default=50000, help="samples in the simulated GWAS cohort")
    ap.add_argument("--write-individual-genotype", action="store_true", default=False,
                    help="also dump the per-rep individual-level GWAS genotype table (large at 50k)")
    ap.add_argument("--pleio-prop", type=float, default=0.25, help="prop_pheno_exp for pleiotropy scans")
    ap.add_argument("--windows", type=int, default=None)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--grn-min-functional-ratio", type=float, default=0.6)
    ap.add_argument("--grn-max-nonfunctional-paths", type=int, default=1)
    ap.add_argument("--grn-direct-weight-low", type=float, default=0.5)
    ap.add_argument("--grn-direct-weight-high", type=float, default=1.0)
    ap.add_argument("--grn-indirect-weight-low", type=float, default=0.3)
    ap.add_argument("--grn-indirect-weight-high", type=float, default=0.8)
    ap.add_argument("--grn-indirect-decay", type=float, default=0.7)
    ap.add_argument("--grn-force-flat-prior", action="store_true", default=False)
    args = ap.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sim.GENO_PATH = str(Path(args.geno_path).resolve()) + "/"
    sim.BED_PATH = str(Path(args.bed_path).resolve())
    sim.SAMPLE_PATH = str(Path(args.sample_path).resolve())
    sim.TF_INFO_PATH = str(Path(args.tf_info_path).resolve())

    metadata = {
        "geno_path": sim.GENO_PATH, "bed_path": sim.BED_PATH,
        "sample_path": sim.SAMPLE_PATH, "tf_info_path": sim.TF_INFO_PATH,
        "scan": args.scan, "n_reps": args.n_reps, "n_gwas": args.n_gwas,
        "pleio_prop": args.pleio_prop, "base_seed": args.base_seed,
        "pleiotropy_model": "complete (ALL TG cis-SNPs + ALL real-TF cis-SNPs, i.i.d. direct effects)",
    }
    with open(out_root / "generation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    base_params = sim.base_genetic_params
    param_combinations, _ = sim.generate_simulation_parameters_final(
        base_params=base_params, scan_type=args.scan,
        pleio_proportions=[args.pleio_prop], BASE_OUTPUT=str(out_root),
    )
    if args.param_limit > 0:
        param_combinations = param_combinations[: args.param_limit]

    grn_cfg = GRNPriorConfig(
        min_functional_ratio=args.grn_min_functional_ratio,
        max_nonfunctional_paths=args.grn_max_nonfunctional_paths,
        direct_weight_low=args.grn_direct_weight_low,
        direct_weight_high=args.grn_direct_weight_high,
        indirect_weight_low=args.grn_indirect_weight_low,
        indirect_weight_high=args.grn_indirect_weight_high,
        indirect_decay=args.grn_indirect_decay,
    )

    for idx, param_set in enumerate(param_combinations):
        param_set = dict(param_set)
        if args.windows is not None:
            param_set["windows"] = int(args.windows)
        process_param_set(
            idx=idx, param_set=param_set, out_root=out_root, n_reps=args.n_reps,
            n_gwas=args.n_gwas, base_seed=args.base_seed,
            grn_cfg=grn_cfg, grn_force_flat_prior=args.grn_force_flat_prior,
            write_individual_genotype=args.write_individual_genotype,
        )

    print(f"Stage-2 generation complete. Output root: {out_root}")


if __name__ == "__main__":
    main()
