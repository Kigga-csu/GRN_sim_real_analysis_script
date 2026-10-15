#!/usr/bin/env python3

import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

GRN_RESULT_FILE = "vcf_GRNTWAS_identified_TFs.tsv"
KNOCK_FILE = "diff_expr_gene_all_data_raw.tsv"
BRAIN_TF_LIST = "TF_all_brian_260104.tsv"
BRAIN_GRN_RELATION = "Combination_trrust_HTF_Brain.tsv"
TPM_FILE = "tpm_normalized_gene_peer_vcf.csv"
OUTPUT_DIR = "."

FC_THRESHOLD = 0.2
N_PERMS = 1000


def plot_permutation_histogram(
    null_dist, observed_value, analysis_name, filename, color="skyblue", font_size=20, bin_count=25
):
    null_dist = np.array(null_dist)

    plt.figure(figsize=(12, 9))

    mean_null = np.mean(null_dist)
    fold_enrich = observed_value / mean_null if mean_null > 0 else np.inf

    n_perms = len(null_dist)
    p_count = int(np.sum(null_dist >= observed_value))
    if p_count == 0:
        p_text = f"P < {1 / n_perms:.4f}"
    else:
        p_val = p_count / n_perms
        p_text = f"P = {p_val:.2e}" if p_val < 0.0001 else f"P = {p_val:.3f}"

    plt.hist(
        null_dist, bins=bin_count, color=color, edgecolor="black", alpha=0.7, label="Expected by chance"
    )
    plt.axvline(
        observed_value, color="#D62728", linestyle="--", linewidth=4, label=f"GRNTWAS observed ({observed_value})"
    )

    plt.title(
        f"{analysis_name}\n(Fold enrichment: {fold_enrich:.2f}, {p_text})", fontsize=font_size + 4, pad=20
    )
    plt.ylabel("Number of iterations", fontsize=font_size + 2, labelpad=15)
    plt.xlabel("Number of experimentally supported pairs", fontsize=font_size + 2, labelpad=15)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.legend(
        fontsize=font_size - 2,
        loc="upper center",
        bbox_to_anchor=(0.75, 1.0),
        frameon=True,
        shadow=True,
        fancybox=True,
        borderpad=0.3,
    )

    plt.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename} (fold enrichment {fold_enrich:.2f}, {p_text})")


def run_permutation_test(fc_threshold=FC_THRESHOLD, n_perms=N_PERMS):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_grn_res = pd.read_csv(GRN_RESULT_FILE, sep="\t")
    df_grn_res = df_grn_res[df_grn_res["Identified_TFs"] != "None"].copy()
    observed_pairs = []
    for _, row in df_grn_res.iterrows():
        for tf_info in str(row["Identified_TFs"]).split(","):
            match = re.match(r"(.+?)\((.+?)\)", tf_info.strip())
            if match:
                observed_pairs.append((match.group(1), str(row["TargetID"]).strip()))
    observed_tgs = [p[1] for p in observed_pairs]

    df_brain_tfs = pd.read_csv(BRAIN_TF_LIST, sep="\t", header=None)
    full_brain_tf_names = df_brain_tfs[0].unique().tolist()

    df_ktf = pd.read_csv(
        KNOCK_FILE, sep="\t", usecols=["TF", "Ensembl_ID", "Log2FC"], low_memory=False
    )
    df_ktf["Log2FC"] = pd.to_numeric(df_ktf["Log2FC"], errors="coerce")
    support_set = set(
        zip(
            df_ktf[df_ktf["Log2FC"].abs() > fc_threshold]["TF"],
            df_ktf[df_ktf["Log2FC"].abs() > fc_threshold]["Ensembl_ID"],
        )
    )

    df_static_grn = pd.read_csv(BRAIN_GRN_RELATION, sep="\t", header=None, names=["TF", "TG"])
    static_grn_map = df_static_grn.groupby("TG")["TF"].apply(list).to_dict()

    def count_hits(pairs):
        return sum(1 for tf, tg in pairs if (tf, tg) in support_set)

    real_hits = count_hits(observed_pairs)

    null_1 = [
        count_hits(zip(np.random.choice(full_brain_tf_names, len(observed_pairs)), observed_tgs))
        for _ in range(n_perms)
    ]
    plot_permutation_histogram(
        null_1, real_hits, "Global brain TF pool",
        os.path.join(OUTPUT_DIR, "Plot_1_Global_Pool.pdf"), color="lightsteelblue",
    )

    null_2 = [
        count_hits([(np.random.choice(static_grn_map.get(tg, full_brain_tf_names)), tg) for tg in observed_tgs])
        for _ in range(n_perms)
    ]
    plot_permutation_histogram(
        null_2, real_hits, "Static GRN (TRRUST + hTFtarget)",
        os.path.join(OUTPUT_DIR, "Plot_2_GRN_Constrained.pdf"), color="cadetblue",
    )

    df_tpm = pd.read_csv(TPM_FILE, sep="\t")
    sample_cols = df_tpm.columns[5:]
    tf_expr_matrix = df_tpm[df_tpm["GeneName"].isin(full_brain_tf_names)].set_index("GeneName")[sample_cols]

    tg_coexp_map = {}
    all_coexp_pool = []
    for tg_id in set(observed_tgs):
        tg_data = df_tpm[df_tpm["TargetID"] == tg_id]
        if tg_data.empty:
            continue
        tg_vec = tg_data[sample_cols].values.flatten()
        coexp_tfs = []
        for tf_name, tf_vec in tf_expr_matrix.iterrows():
            r, p = pearsonr(tg_vec, tf_vec)
            if p < 0.05 and abs(r) > 0.1:
                coexp_tfs.append(tf_name)
                all_coexp_pool.append((tf_name, tg_id))
        tg_coexp_map[tg_id] = coexp_tfs

    null_3 = [
        count_hits([all_coexp_pool[i] for i in np.random.choice(len(all_coexp_pool), len(observed_pairs))])
        for _ in range(n_perms)
    ]
    plot_permutation_histogram(
        null_3, real_hits, "Co-expression TF-target pairs",
        os.path.join(OUTPUT_DIR, "Plot_3_Coexp_Pool.pdf"), color="darkseagreen",
    )

    intersection_bg_map = {}
    for tg in observed_tgs:
        grn_tfs = set(static_grn_map.get(tg, []))
        coexp_tfs = set(tg_coexp_map.get(tg, []))
        intersect = list(grn_tfs.intersection(coexp_tfs))
        intersection_bg_map[tg] = intersect if intersect else full_brain_tf_names

    null_4 = [
        count_hits([(np.random.choice(intersection_bg_map[tg]), tg) for tg in observed_tgs])
        for _ in range(n_perms)
    ]
    plot_permutation_histogram(
        null_4, real_hits, "GRN and co-expression intersection",
        os.path.join(OUTPUT_DIR, "Plot_4_Double_Constrained.pdf"), color="indianred",
    )

    print("Permutation test complete.")


if __name__ == "__main__":
    run_permutation_test()
