#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import re
from pathlib import Path

import pandas as pd
import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path.cwd() / ".mplconfig").resolve())

import simulation_step1_core as sim
from simulation_grn_prior_builder import (
    GRNPriorConfig,
    build_local_grn_prior,
    build_role_map_from_existing_sets,
)


def _detect_available_chromosomes(vcf_dir: Path):
    pattern = re.compile(r"chr([0-9]+)\.filtered\.shapeit2-duohmm-phased\.vcf\.gz$")
    chroms = set()
    for p in vcf_dir.glob("*.vcf.gz"):
        m = pattern.search(p.name)
        if not m:
            continue
        tbi = p.with_suffix(p.suffix + ".tbi")
        if tbi.exists():
            chroms.add(m.group(1))
    return sorted(chroms, key=lambda x: int(x))


def _build_filtered_bed(bed_path: Path, output_dir: Path, allowed_chroms):
    df = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "strand", "gene_id", "gene_name", "gene_type"],
    )
    allowed = set(allowed_chroms)

    def _norm(c):
        c = str(c)
        return c[3:] if c.startswith("chr") else c

    kept = df[df["chrom"].map(_norm).isin(allowed)].copy()
    if kept.empty:
        raise RuntimeError("Filtered BED is empty. Check VCF/BED chromosome naming.")

    kept["chrom"] = kept["chrom"].map(lambda c: str(_norm(c)))

    filtered_bed = output_dir / "gene.filtered_by_available_vcf.bed"
    kept.to_csv(filtered_bed, sep="\t", header=False, index=False)
    return filtered_bed, len(df), len(kept)


def _build_tg_direct_trans_pool(gene_data, min_distance_bp: int, window_bp: int):
    tg_id = gene_data["tg_id"]
    anno_data = gene_data["anno_data"]
    sample_id = gene_data["sampleID"]

    tg_row = anno_data[anno_data["gene_id"] == tg_id]
    if tg_row.empty:
        return pd.DataFrame(columns=["CHROM", "POS", "snpID", "REF", "ALT", "p_HWE", "MAF", "missing_rate", *sample_id])

    gene_start = int(tg_row.iloc[0]["start"])
    gene_end = int(tg_row.iloc[0]["end"])
    gene_chrom = str(tg_row.iloc[0]["chrom"])
    if gene_chrom.startswith("chr"):
        gene_chrom = gene_chrom[3:]

    geno_df, _ = sim.extract_genotype_vcf(
        bed_file=sim.BED_PATH,
        geno_path=sim.GENO_PATH,
        gene_ID=tg_id,
        sampleid_path=sim.SAMPLE_PATH,
        windows=window_bp,
    )
    if geno_df is None or geno_df.empty:
        return pd.DataFrame(columns=["CHROM", "POS", "snpID", "REF", "ALT", "p_HWE", "MAF", "missing_rate", *sample_id])

    chrom_norm = geno_df["CHROM"].astype(str).map(lambda x: x[3:] if x.startswith("chr") else x)
    pos = pd.to_numeric(geno_df["POS"], errors="coerce")
    mask_chr = chrom_norm == gene_chrom
    mask_far = (pos <= gene_start - min_distance_bp) | (pos >= gene_end + min_distance_bp)
    out = geno_df[mask_chr & mask_far].copy()
    if out.empty:
        return out
    return out.drop_duplicates(subset=["snpID"], keep="first")


def _save_pickle(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: generate simulation data only")
    parser.add_argument("--output-root", required=True, help="Output root for generated datasets")
    parser.add_argument("--geno-path", required=True, help="Directory containing 1000G chr*.vcf.gz")
    parser.add_argument("--bed-path", required=True, help="gene.bed path")
    parser.add_argument("--sample-path", required=True, help="sample list (one id per line)")
    parser.add_argument("--tf-info-path", required=True, help="TF file path (must include Ensembl column)")
    parser.add_argument("--scan", default="mix_casual", choices=[
        "cis_casual", "trans_casual", "mix_casual", "cis_pleiotropy", "trans_pleiotropy", "mix_pleiotropy"
    ])
    parser.add_argument("--n-reps", type=int, default=5, help="Replications per parameter set")
    parser.add_argument("--param-limit", type=int, default=1, help="Use first N param sets for quick testing")
    parser.add_argument("--windows", type=int, default=None, help="SNP extraction flank window in bp for TF/TG")
    parser.add_argument("--h-cis", type=float, default=None, help="Override h_cis for all parameter sets")
    parser.add_argument("--h-trans", type=float, default=None, help="Override h_trans for all parameter sets")
    parser.add_argument("--n-tf-class1", type=int, default=None, help="Override n_TF_class1 for all parameter sets")
    parser.add_argument("--n-tf-class2", type=int, default=None, help="Override n_TF_class2 for all parameter sets")
    parser.add_argument("--real-tf-class1", type=int, default=None, help="Override real_TF_class1 for all parameter sets")
    parser.add_argument("--real-tf-class2", type=int, default=None, help="Override real_TF_class2 for all parameter sets")
    parser.add_argument("--pleio-prop", type=float, default=0.25, help="prop_pheno_exp for pleiotropy scans")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--auto-filter-bed", action="store_true", default=True,
                        help="Filter BED to chromosomes with downloaded VCFs")
    parser.add_argument("--grn-min-functional-ratio", type=float, default=0.6)
    parser.add_argument("--grn-max-nonfunctional-paths", type=int, default=1)
    parser.add_argument("--grn-direct-weight-low", type=float, default=0.5)
    parser.add_argument("--grn-direct-weight-high", type=float, default=1.0)
    parser.add_argument("--grn-indirect-weight-low", type=float, default=0.3)
    parser.add_argument("--grn-indirect-weight-high", type=float, default=0.8)
    parser.add_argument("--grn-indirect-decay", type=float, default=0.7)
    parser.add_argument("--grn-force-flat-prior", action="store_true", default=False,
                        help="Force no indirect GRN layer by moving indirect functional TFs to direct layer")
    parser.add_argument("--direct-trans-eqtl-count", type=int, default=10,
                        help="Number of TG direct trans-eQTL SNPs (same chr, > distance threshold)")
    parser.add_argument("--direct-trans-min-distance-mb", type=float, default=10.0,
                        help="Minimum TG distance (Mb) for direct trans-eQTL SNP pool")
    parser.add_argument("--direct-trans-window-mb", type=float, default=50.0,
                        help="Search window around TG (Mb) for building direct trans SNP pool")
    parser.add_argument("--h-trans-direct", type=float, default=0.0,
                        help="Variance component of direct trans-eQTL (not TF-mediated)")
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    geno_path = Path(args.geno_path).resolve()
    bed_path = Path(args.bed_path).resolve()
    sample_path = Path(args.sample_path).resolve()
    tf_info_path = Path(args.tf_info_path).resolve()

    sim.GENO_PATH = str(geno_path) + "/"
    sim.BED_PATH = str(bed_path)
    sim.SAMPLE_PATH = str(sample_path)
    sim.TF_INFO_PATH = str(tf_info_path)

    metadata = {
        "geno_path": sim.GENO_PATH,
        "bed_path": sim.BED_PATH,
        "sample_path": sim.SAMPLE_PATH,
        "tf_info_path": sim.TF_INFO_PATH,
        "scan": args.scan,
        "n_reps": args.n_reps,
        "param_limit": args.param_limit,
        "base_seed": args.base_seed,
        "overrides": {
            "windows": args.windows,
            "h_cis": args.h_cis,
            "h_trans": args.h_trans,
            "n_TF_class1": args.n_tf_class1,
            "n_TF_class2": args.n_tf_class2,
            "real_TF_class1": args.real_tf_class1,
            "real_TF_class2": args.real_tf_class2,
        },
        "grn_prior_config": {
            "min_functional_ratio": args.grn_min_functional_ratio,
            "max_nonfunctional_paths": args.grn_max_nonfunctional_paths,
            "direct_weight_low": args.grn_direct_weight_low,
            "direct_weight_high": args.grn_direct_weight_high,
            "indirect_weight_low": args.grn_indirect_weight_low,
            "indirect_weight_high": args.grn_indirect_weight_high,
            "indirect_decay": args.grn_indirect_decay,
            "force_flat_prior": args.grn_force_flat_prior,
        },
        "direct_trans_config": {
            "eqtl_count": args.direct_trans_eqtl_count,
            "min_distance_mb": args.direct_trans_min_distance_mb,
            "window_mb": args.direct_trans_window_mb,
            "h_trans_direct": args.h_trans_direct,
        },
    }

    if args.auto_filter_bed:
        chroms = _detect_available_chromosomes(geno_path)
        if not chroms:
            raise RuntimeError("No valid chr*.vcf.gz + .tbi detected under --geno-path")
        filtered_bed, n_before, n_after = _build_filtered_bed(bed_path, out_root, chroms)
        sim.BED_PATH = str(filtered_bed)
        metadata["available_chromosomes"] = chroms
        metadata["bed_filtered_from"] = n_before
        metadata["bed_filtered_to"] = n_after
        metadata["bed_path_filtered"] = sim.BED_PATH

    base_params = sim.base_genetic_params
    pleio_props = [args.pleio_prop]
    param_combinations, _ = sim.generate_simulation_parameters_final(
        base_params=base_params,
        scan_type=args.scan,
        pleio_proportions=pleio_props,
        BASE_OUTPUT=str(out_root),
    )

    if args.param_limit > 0:
        param_combinations = param_combinations[: args.param_limit]

    metadata["n_param_sets"] = len(param_combinations)
    with open(out_root / "generation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    for idx, param_set in enumerate(param_combinations):
        param_set = dict(param_set)
        if args.windows is not None:
            param_set["windows"] = int(args.windows)
        if args.h_cis is not None:
            param_set["h_cis"] = float(args.h_cis)
        if args.h_trans is not None:
            param_set["h_trans"] = float(args.h_trans)
        if args.n_tf_class1 is not None:
            param_set["n_TF_class1"] = int(args.n_tf_class1)
        if args.n_tf_class2 is not None:
            param_set["n_TF_class2"] = int(args.n_tf_class2)
        if args.real_tf_class1 is not None:
            param_set["real_TF_class1"] = int(args.real_tf_class1)
        if args.real_tf_class2 is not None:
            param_set["real_TF_class2"] = int(args.real_tf_class2)

        param_dir = out_root / f"param_set_{idx}"
        param_dir.mkdir(parents=True, exist_ok=True)

        params = sim.SimulationParams(**param_set)
        params.h_trans_direct = float(args.h_trans_direct)
        params.n_direct_trans_eqtl = int(args.direct_trans_eqtl_count)
        selection_seed = args.base_seed + idx * 1000
        gene_data = sim.extract_gene_data(params, str(param_dir), selection_seed=selection_seed)
        if args.direct_trans_eqtl_count > 0 and args.h_trans_direct > 0:
            direct_trans_pool = _build_tg_direct_trans_pool(
                gene_data=gene_data,
                min_distance_bp=int(args.direct_trans_min_distance_mb * 1_000_000),
                window_bp=int(args.direct_trans_window_mb * 1_000_000),
            )
        else:
            direct_trans_pool = pd.DataFrame()
        gene_data["tg_direct_trans_pool_geno"] = direct_trans_pool

        _save_pickle(param_dir / "gene_data.pkl", gene_data)
        with open(param_dir / "params.json", "w", encoding="utf-8") as f:
            json.dump(param_set, f, indent=2, ensure_ascii=False)

        for rep in range(args.n_reps):
            rep_seed = args.base_seed + idx * 1000 + rep
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
                tg_id,
            ) = sim.generate_expression(
                params,
                str(rep_dir),
                gene_data,
                expression_seed=rep_seed,
                direct_trans_pool_geno=gene_data.get("tg_direct_trans_pool_geno"),
            )

            role_map = build_role_map_from_existing_sets(
                tf_class1_ids=[str(x) for x in tf_class1_ids],
                tf_class2_ids=[str(x) for x in tf_class2_ids],
            )
            if args.grn_force_flat_prior:
                role_map = {
                    "direct_functional": sorted(set(role_map["direct_functional"] + role_map["indirect_functional"])),
                    "indirect_functional": [],
                    "indirect_nonfunctional": [],
                    "noise_tf": role_map["noise_tf"],
                }
            grn_cfg = GRNPriorConfig(
                min_functional_ratio=args.grn_min_functional_ratio,
                max_nonfunctional_paths=args.grn_max_nonfunctional_paths,
                direct_weight_low=args.grn_direct_weight_low,
                direct_weight_high=args.grn_direct_weight_high,
                indirect_weight_low=args.grn_indirect_weight_low,
                indirect_weight_high=args.grn_indirect_weight_high,
                indirect_decay=args.grn_indirect_decay,
            )
            grn_nodes, grn_edges, grn_summary, prior_tf_candidates = build_local_grn_prior(
                tg_id=str(tg_id),
                role_map=role_map,
                rng=rng,
                config=grn_cfg,
            )
            grn_nodes.to_csv(rep_dir / "grn_nodes.tsv", sep="\t", index=False)
            grn_edges.to_csv(rep_dir / "grn_edges.tsv", sep="\t", index=False)
            grn_summary.to_csv(rep_dir / "grn_summary.tsv", sep="\t", index=False)

            payload = {
                "rep": rep,
                "rep_seed": rep_seed,
                "all_tg_expressions": pd.concat(tg_expressions, axis=0, ignore_index=True),
                "all_tf_expressions": pd.concat(tf_expressions, axis=0, ignore_index=True),
                "all_tf_ids": list(all_tf_ids),
                "tf_class1_ids": [str(x) for x in tf_class1_ids],
                "tf_class2_ids": [str(x) for x in tf_class2_ids],
                "tg_4_real_tf": {str(k): [str(v) for v in vals] for k, vals in tg_4_real_tf.items()},
                "cis_weights": cis_weights,
                "trans_weights": trans_weights,
                "tg_id": str(tg_id),
                "grn_role_map": role_map,
                "prior_tf_candidates": [str(x) for x in prior_tf_candidates],
                "grn_summary": grn_summary.to_dict(orient="records")[0],
            }
            _save_pickle(rep_dir / "sim_payload.pkl", payload)

    print(f"Generation complete. Output root: {out_root}")


if __name__ == "__main__":
    main()
