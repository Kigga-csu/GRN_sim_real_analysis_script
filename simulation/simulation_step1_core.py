#!/usr/bin/env python3

import operator
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import GRNutils as tg


base_genetic_params = [
    {"n_TG": 1, "h_cis": 0.0375, "h_trans": 0.0125, "h_tf": 0.1, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.0375, "h_trans": 0.0375, "h_tf": 0.1, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.0375, "h_trans": 0.075, "h_tf": 0.1, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.05, "h_trans": 0.01, "h_tf": 0.1, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.05, "h_trans": 0.05, "h_tf": 0.1, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.05, "h_trans": 0.2, "h_tf": 0.3, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.075, "h_trans": 0.025, "h_tf": 0.1, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.075, "h_trans": 0.075, "h_tf": 0.2, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.075, "h_trans": 0.3, "h_tf": 0.5, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.1, "h_trans": 0.03, "h_tf": 0.2, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.1, "h_trans": 0.1, "h_tf": 0.3, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.1, "h_trans": 0.3, "h_tf": 0.6, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.2, "h_trans": 0.05, "h_tf": 0.2, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.2, "h_trans": 0.1, "h_tf": 0.3, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.2, "h_trans": 0.2, "h_tf": 0.3, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.2, "h_trans": 0.4, "h_tf": 0.8, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.2, "h_trans": 0.6, "h_tf": 0.3, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.3, "h_trans": 0.1, "h_tf": 0.5, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.3, "h_trans": 0.6, "h_tf": 0.9, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.0375, "h_trans": 0.0, "h_tf": 0.1, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
    {"n_TG": 1, "h_cis": 0.0, "h_trans": 0.0375, "h_tf": 0.1, "n_TF_class1": 6, "n_TF_class2": 4, "real_TF_class1": 1, "real_TF_class2": 2},
]


class SimulationParams:
    def __init__(
        self,
        n_TG=1,
        n_TF_class1=6,
        n_TF_class2=4,
        real_TF_class1=1,
        real_TF_class2=2,
        p_eqtl_cis=0.1,
        h_cis=0.2,
        h_trans=0.1,
        h_tf=0.1,
        h_t=0.3,
        windows=5000,
        scan="mix_casual",
        prop_pheno_exp=1.0,
    ):
        self.n_TG = n_TG
        self.n_TF_class1 = n_TF_class1
        self.n_TF_class2 = n_TF_class2
        self.real_TF_class1 = real_TF_class1
        self.real_TF_class2 = real_TF_class2
        self.p_eqtl_cis = p_eqtl_cis
        self.h_cis = h_cis
        self.h_trans = h_trans
        self.h_tf = h_tf
        self.h2 = h_cis + h_trans
        self.h_U_tf = max(1 - self.h_tf, 0) * 0.8
        self.h_U_tg = max(1 - self.h_cis - (self.h_trans / self.h_tf), 0) * 0.8
        self.h_E_tf = max(1 - self.h_tf, 0) * 0.2
        self.h_E_tg = max(1 - self.h_cis - (self.h_trans / self.h_tf), 0) * 0.2
        self.windows = windows
        self.h_t = h_t
        self.scan = scan
        self.prop_pheno_exp = prop_pheno_exp


GENO_PATH = ""
BED_PATH = ""
SAMPLE_PATH = ""
TF_INFO_PATH = ""

MAF_THRESHOLD = 0.01
MISSING_RATE_THRESHOLD = 0.15
HWE_THRESHOLD = 0.05
_STARTUP_CACHE = {}


def generate_simulation_parameters_final(base_params, scan_type, pleio_proportions=None, BASE_OUTPUT="./"):
    final_combinations = []
    is_pleiotropic = "pleiotropy" in scan_type

    if is_pleiotropic:
        if pleio_proportions is None or not pleio_proportions:
            raise ValueError("pleio_proportions must be provided for pleiotropy scans.")
        prop_values = pleio_proportions
    else:
        prop_values = [1.0]

    for base_p in base_params:
        h2g_expr = base_p["h_cis"] + base_p["h_trans"]
        if h2g_expr <= 0:
            warnings.warn(f"Skip invalid base params h_cis+h_trans<=0: {base_p}")
            continue
        for prop in prop_values:
            new_param_set = base_p.copy()
            new_param_set["scan"] = scan_type
            new_param_set["prop_pheno_exp"] = prop
            final_combinations.append(new_param_set)

    return final_combinations, os.path.join(BASE_OUTPUT, scan_type)


def gene_info_4_id(bed_file_path, gene_ids, windows):
    df = pd.read_csv(
        bed_file_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "strand", "gene_id", "gene_name", "gene_type"],
    )
    df_filtered = df[df["gene_id"].isin(gene_ids)]
    if df_filtered.empty:
        raise ValueError(f"Gene ids not found in BED: {gene_ids}")
    chrom = df_filtered["chrom"].astype(str).values
    gene_names = df_filtered["gene_name"].values
    gene_start = df_filtered["start"].values
    gene_end = df_filtered["end"].values
    start = max(int(gene_start[0]) - windows, 0)
    end = int(gene_end[0]) + windows
    return gene_names, start, end, chrom


def _norm_chr(c):
    c = str(c)
    return c[3:] if c.startswith("chr") else c


def extract_genotype_vcf(genofile_type="vcf", data_format="GT", bed_file=None, geno_path=None, gene_ID=None, sampleid_path=None, windows=None):
    target_name, start, end, chrom = gene_info_4_id(bed_file, [gene_ID], windows)
    chrom_key = _norm_chr(chrom[0])
    try:
        chrom_int = int(chrom_key)
        if chrom_int not in range(1, 23):
            print(f"{target_name} not have genotype")
            return None, None
    except ValueError:
        print(f"{target_name} not have genotype")
        return None, None

    genofile_path = f"{geno_path}CCDG_14151_B01_GRM_WGS_2020-08-05_chr{chrom_key}.filtered.shapeit2-duohmm-phased.vcf.gz"
    cache_key = (chrom_key, genofile_type, data_format, genofile_path, sampleid_path)
    cached = _STARTUP_CACHE.get(cache_key)
    if cached is None:
        sampleID, _, geno_info = tg.sampleid_startup_simulation(
            chrm=chrom_key,
            genofile_type=genofile_type,
            data_format=data_format,
            geno_path=genofile_path,
            sampleid_path=sampleid_path,
        )
        _STARTUP_CACHE[cache_key] = (sampleID, geno_info)
    else:
        sampleID, geno_info = cached
    if not isinstance(sampleID, np.ndarray):
        print(f"Failed to get sample IDs for {target_name}")
        return None, None

    df_Geno = tg.read_tabix_revise(str(start), str(end), sampleID, **geno_info)
    if df_Geno is False or df_Geno.empty:
        print(f"No genotype data for {target_name}")
        return None, None
    df_Geno.drop_duplicates(subset="snpID", keep="first", inplace=True)
    df_Geno = tg.handle_missing_wsx(df_Geno, sampleID, MISSING_RATE_THRESHOLD)
    if df_Geno.empty:
        return None, None
    df_Geno = tg.calc_maf(df_Geno, sampleID, MAF_THRESHOLD, op=operator.ge)
    if df_Geno.empty:
        return None, None
    df_Geno = tg.calc_p_hwe(df_Geno, sampleID, HWE_THRESHOLD, op=operator.ge)
    if df_Geno.empty:
        return None, None
    if len(df_Geno) > 100:
        df_Geno = df_Geno.sample(n=100, random_state=42)
    return df_Geno, sampleID


def simulate_low_rank_confounder(n_samples, h_U, k=10):
    if h_U <= 0:
        return np.zeros(n_samples)
    A = np.random.randn(n_samples, k)
    f = np.random.randn(k)
    U = A @ f
    if k > 0:
        return U * np.sqrt(h_U / k)
    return np.zeros(n_samples)


def simulate_gene_expression(snpsMat, n_qtl, p_causal, h_component):
    n_qtls = max(1, int(np.floor(p_causal * n_qtl)))
    c_qtls = np.random.choice(n_qtl, size=n_qtls, replace=False)
    b_qtls = np.zeros(n_qtl)
    b_qtls[c_qtls] = np.random.normal(loc=0, scale=np.sqrt(h_component / n_qtls), size=n_qtls)
    return np.dot(snpsMat, b_qtls), b_qtls


def simulate_TF_expression(tf_gene_id, geno, sampleID, params):
    if geno is None or geno.empty:
        n_samples = len(sampleID)
        gexpr_tot = np.random.normal(loc=0, scale=np.sqrt(params.h_U_tf + params.h_E_tf), size=n_samples)
        weights_df = pd.DataFrame(columns=["CHROM", "POS", "snpID", "REF", "ALT", "beta"])
        return gexpr_tot, weights_df, sampleID, None

    snpsMat = geno.drop(columns=["CHROM", "POS", "snpID", "REF", "ALT", "p_HWE", "MAF", "missing_rate"]).values
    snpsMat = StandardScaler().fit_transform(snpsMat.T)
    n_qtl = geno["snpID"].size
    gexpr_cis, b_cis = simulate_gene_expression(snpsMat, n_qtl, params.p_eqtl_cis, params.h_tf)
    U = simulate_low_rank_confounder(len(sampleID), params.h_U_tf, k=10)
    E = np.random.normal(loc=0, scale=np.sqrt(params.h_E_tf), size=len(sampleID))
    gexpr_tot = gexpr_cis + U + E
    weights_df = pd.DataFrame(
        {
            "CHROM": geno["CHROM"],
            "POS": geno["POS"],
            "snpID": geno["snpID"],
            "REF": geno["REF"],
            "ALT": geno["ALT"],
            "beta": b_cis,
        }
    )
    return gexpr_tot, weights_df[weights_df["beta"] != 0], sampleID, geno


def simulate_TG_expression(tg_gene_id, tg_geno, sampleID, real_tf_ids, tf_expressions, tf_weights, params, direct_trans_pool_geno=None):
    if tg_geno is None or tg_geno.empty:
        return None, None, None, None

    tg_snpsMat = tg_geno.drop(columns=["CHROM", "POS", "snpID", "REF", "ALT", "p_HWE", "MAF", "missing_rate"]).values
    tg_snpsMat = StandardScaler().fit_transform(tg_snpsMat.T)
    n_qtl_cis = tg_geno["snpID"].size
    gexpr_cis, b_cis = simulate_gene_expression(tg_snpsMat, n_qtl_cis, params.p_eqtl_cis, params.h_cis)
    cis_weights = pd.DataFrame(
        {
            "CHROM": tg_geno["CHROM"],
            "POS": tg_geno["POS"],
            "snpID": tg_geno["snpID"],
            "REF": tg_geno["REF"],
            "ALT": tg_geno["ALT"],
            "beta": b_cis,
        }
    )
    cis_weights = cis_weights[cis_weights["beta"] != 0]

    h_trans_direct = float(getattr(params, "h_trans_direct", 0.0) or 0.0)
    h_trans_tf = max(float(params.h_trans) - h_trans_direct, 0.0)

    real_tf_expression_values = []
    for tf_id in real_tf_ids:
        tf_expr_df = next((tf_expr for tf_expr in tf_expressions if tf_expr["TargetID"].iloc[0] == tf_id), None)
        if tf_expr_df is not None:
            tf_expression_for_tg = tg.center(tf_expr_df[sampleID], sampleID).values.flatten()
            real_tf_expression_values.append(tf_expression_for_tg)

    if real_tf_expression_values:
        tf_expression_matrix = np.vstack(real_tf_expression_values).T
        n_tf = tf_expression_matrix.shape[1]
        if params.h_tf > 0 and h_trans_tf > 0:
            b_trans_expr = np.random.normal(loc=0, scale=np.sqrt(h_trans_tf / params.h_tf / n_tf), size=n_tf)
        else:
            b_trans_expr = np.zeros(n_tf)
        gexpr_trans_tf = np.dot(tf_expression_matrix, b_trans_expr)
    else:
        gexpr_trans_tf = np.zeros(len(sampleID))
        b_trans_expr = np.zeros(0)

    n_direct_trans_eqtl = int(getattr(params, "n_direct_trans_eqtl", 0) or 0)
    direct_weight_cols = ["CHROM", "POS", "snpID", "REF", "ALT", "beta"]
    selected_direct_trans_weights = pd.DataFrame(columns=direct_weight_cols)
    gexpr_trans_direct = np.zeros(len(sampleID))
    if n_direct_trans_eqtl > 0 and h_trans_direct > 0 and direct_trans_pool_geno is not None and not direct_trans_pool_geno.empty:
        n_available = direct_trans_pool_geno["snpID"].nunique()
        n_pick = min(n_direct_trans_eqtl, n_available)
        if n_pick > 0:
            picked = direct_trans_pool_geno.sample(n=n_pick, replace=False, random_state=None).copy()
            x_direct = picked.drop(columns=["CHROM", "POS", "snpID", "REF", "ALT", "p_HWE", "MAF", "missing_rate"], errors="ignore").values
            x_direct = StandardScaler().fit_transform(x_direct.T)
            b_direct = np.random.normal(loc=0, scale=np.sqrt(h_trans_direct / n_pick), size=n_pick)
            gexpr_trans_direct = np.dot(x_direct, b_direct)
            selected_direct_trans_weights = pd.DataFrame(
                {
                    "CHROM": picked["CHROM"].values,
                    "POS": picked["POS"].values,
                    "snpID": picked["snpID"].values,
                    "REF": picked["REF"].values,
                    "ALT": picked["ALT"].values,
                    "beta": b_direct,
                }
            )

    U = simulate_low_rank_confounder(len(sampleID), params.h_U_tg, k=10)
    E = np.random.normal(loc=0, scale=np.sqrt(params.h_E_tg), size=len(sampleID))
    gexpr_tot = gexpr_cis + gexpr_trans_tf + gexpr_trans_direct + U + E

    selected_trans_weights_list = []
    for i, tf_id in enumerate(real_tf_ids):
        if tf_id in tf_weights and not tf_weights[tf_id].empty:
            tf_weight_df = tf_weights[tf_id].copy()
            tf_weight_df["beta"] = tf_weight_df["beta"] * b_trans_expr[i]
            selected_trans_weights_list.append(tf_weight_df)
    selected_trans_weights = (
        pd.concat(selected_trans_weights_list, ignore_index=True)
        if selected_trans_weights_list
        else pd.DataFrame(columns=direct_weight_cols)
    )
    if not selected_direct_trans_weights.empty:
        selected_trans_weights = pd.concat([selected_trans_weights, selected_direct_trans_weights], ignore_index=True)
    return gexpr_tot, cis_weights, selected_trans_weights, sampleID, tg_geno


def extract_gene_data(params, OUTPUT_DIR, selection_seed):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    anno_data = pd.read_csv(BED_PATH, sep="\t", header=None, names=["chrom", "start", "end", "strand", "gene_id", "gene_name", "gene_type"])
    tf_data = pd.read_csv(TF_INFO_PATH, sep="\t")
    all_genes = anno_data["gene_id"].tolist()
    tf_candidates_all = tf_data["Ensembl"].tolist()
    tf_candidates = [gene for gene in tf_candidates_all if gene in all_genes]

    tg_list = [gene for gene in all_genes if gene not in tf_candidates]
    tg_candidates_pool = list(tg_list)
    tg_id = None
    tg_geno = None
    sampleID = None
    np.random.seed(selection_seed)
    while tg_candidates_pool:
        current_tg_id = np.random.choice(tg_candidates_pool)
        current_tg_geno, current_sampleID = extract_genotype_vcf(
            bed_file=BED_PATH,
            geno_path=GENO_PATH,
            gene_ID=current_tg_id,
            sampleid_path=SAMPLE_PATH,
            windows=params.windows,
        )
        if current_tg_geno is not None and not current_tg_geno.empty:
            tg_id = current_tg_id
            tg_geno = current_tg_geno
            sampleID = current_sampleID
            break
        tg_candidates_pool.remove(current_tg_id)
    if tg_id is None:
        raise RuntimeError("No TG with valid genotype data found.")

    tf_genos = {}

    def _select_tf_ids(candidates_pool, n_need):
        selected = []
        pool = list(candidates_pool)
        while len(selected) < n_need and pool:
            tf_id = np.random.choice(pool)
            geno, _ = extract_genotype_vcf(
                bed_file=BED_PATH,
                geno_path=GENO_PATH,
                gene_ID=tf_id,
                sampleid_path=SAMPLE_PATH,
                windows=params.windows,
            )
            if geno is not None and not geno.empty:
                selected.append(tf_id)
                tf_genos[tf_id] = geno
            pool.remove(tf_id)
        return selected

    tf_class1_ids_selected = _select_tf_ids(tf_candidates, params.n_TF_class1)
    tf_class2_pool = [gene for gene in tf_candidates if gene not in set(tf_class1_ids_selected)]
    tf_class2_ids_selected = _select_tf_ids(tf_class2_pool, params.n_TF_class2)

    if len(tf_class1_ids_selected) < params.n_TF_class1:
        raise RuntimeError(
            f"Insufficient TF class1 with valid genotype: need {params.n_TF_class1}, got {len(tf_class1_ids_selected)}"
        )
    if len(tf_class2_ids_selected) < params.n_TF_class2:
        raise RuntimeError(
            f"Insufficient TF class2 with valid genotype: need {params.n_TF_class2}, got {len(tf_class2_ids_selected)}"
        )

    tf_class1_ids = np.array(tf_class1_ids_selected)
    tf_class2_ids = np.array(tf_class2_ids_selected)

    np.random.seed(selection_seed)
    n_causal = int(params.real_TF_class1) + int(params.real_TF_class2)
    all_tf_ids = np.concatenate([tf_class1_ids, tf_class2_ids])
    real_tf = np.random.choice(all_tf_ids, size=min(n_causal, len(all_tf_ids)), replace=False)
    tg_4_real_tf = {tg_id: real_tf}

    return {
        "tf_class1_ids": tf_class1_ids,
        "tf_class2_ids": tf_class2_ids,
        "tg_id": tg_id,
        "tf_genos": tf_genos,
        "tg_geno": tg_geno,
        "tg_4_real_tf": tg_4_real_tf,
        "sampleID": sampleID,
        "anno_data": anno_data,
    }


def generate_expression(params, OUTPUT_DIR, gene_data, expression_seed=None, direct_trans_pool_geno=None):
    tf_class1_ids = gene_data["tf_class1_ids"]
    tf_class2_ids = gene_data["tf_class2_ids"]
    tg_id = gene_data["tg_id"]
    tf_genos = gene_data["tf_genos"]
    tg_geno = gene_data["tg_geno"]
    tg_4_real_tf = gene_data["tg_4_real_tf"]
    sampleID = gene_data["sampleID"]
    anno_data = gene_data["anno_data"]

    np.random.seed(expression_seed if expression_seed is not None else None)

    tf_expressions = []
    tf_weights = {}
    all_tf_ids = np.concatenate([tf_class1_ids, tf_class2_ids])
    for tf_id in all_tf_ids:
        geno = tf_genos.get(tf_id, None)
        expr, weights, _, _ = simulate_TF_expression(tf_id, geno, sampleID, params)
        if expr is not None:
            gene_info = anno_data[anno_data["gene_id"] == tf_id].iloc[0]
            expr_df = pd.DataFrame(
                {
                    "CHROM": [gene_info["chrom"]],
                    "GeneStart": [gene_info["start"]],
                    "GeneEnd": [gene_info["end"]],
                    "TargetID": [tf_id],
                    "GeneName": [gene_info["gene_name"]],
                    **{sampleID[i]: [expr[i]] for i in range(len(sampleID))},
                }
            )
            tf_expressions.append(expr_df)
            tf_weights[tf_id] = weights
            weights.to_csv(f"{OUTPUT_DIR}/TF_eQTL_beta_{tf_id}.csv", sep="\t", index=False)

    tg_expressions = []
    real_tf_ids = tg_4_real_tf[tg_id]
    expr, cis_weights, selected_trans_weights, _, _ = simulate_TG_expression(
        tg_id,
        tg_geno,
        sampleID,
        real_tf_ids,
        tf_expressions,
        tf_weights,
        params,
        direct_trans_pool_geno=direct_trans_pool_geno,
    )
    weights = pd.concat([cis_weights, selected_trans_weights], ignore_index=True)
    if expr is not None:
        gene_info = anno_data[anno_data["gene_id"] == tg_id].iloc[0]
        expr_df = pd.DataFrame(
            {
                "CHROM": [gene_info["chrom"]],
                "GeneStart": [gene_info["start"]],
                "GeneEnd": [gene_info["end"]],
                "TargetID": [tg_id],
                "GeneName": [gene_info["gene_name"]],
                **{sampleID[i]: [expr[i]] for i in range(len(sampleID))},
            }
        )
        tg_expressions.append(expr_df)
        weights.to_csv(f"{OUTPUT_DIR}/TG_eQTL_beta_{tg_id}_TFexpr.csv", sep="\t", index=False)

    TG_CHR_DIR = os.path.join(OUTPUT_DIR, f"TG_expression_h_cis_{params.h_cis}_h_trans_{params.h_trans}_hU_{params.h_U_tg}_by_chr")
    TF_CHR_DIR = os.path.join(OUTPUT_DIR, f"TF_expression_h_cis_{params.h_cis}_h_trans_{params.h_trans}_hU_{params.h_U_tf}_by_chr")
    os.makedirs(TG_CHR_DIR, exist_ok=True)
    os.makedirs(TF_CHR_DIR, exist_ok=True)

    if tf_expressions:
        final_tf_expr = pd.concat(tf_expressions, ignore_index=True)
        final_tf_expr.to_csv(f"{OUTPUT_DIR}/TF_expression_h_tf_{params.h_tf}_hU_{params.h_U_tf}.csv", sep="\t", index=False)
        for chrom, group in final_tf_expr.groupby("CHROM"):
            group.to_csv(os.path.join(TF_CHR_DIR, f"TF_expression_{chrom}.csv"), sep="\t", index=False)

    if tg_expressions:
        final_tg_expr = pd.concat(tg_expressions, ignore_index=True)
        final_tg_expr.to_csv(
            f"{OUTPUT_DIR}/TG_expression_h_cis_{params.h_cis}_h_trans_{params.h_trans}_hU_{params.h_U_tg}.csv",
            sep="\t",
            index=False,
        )
        for chrom, group in final_tg_expr.groupby("CHROM"):
            group.to_csv(os.path.join(TG_CHR_DIR, f"TG_expression_{chrom}.csv"), sep="\t", index=False)

    all_tf_ids = [tf["TargetID"].iloc[0] for tf in tf_expressions]
    return (
        tg_expressions,
        tf_expressions,
        all_tf_ids,
        tf_class1_ids,
        tf_class2_ids,
        tf_genos,
        tg_geno,
        tg_4_real_tf,
        cis_weights,
        selected_trans_weights,
        tg_id,
    )
