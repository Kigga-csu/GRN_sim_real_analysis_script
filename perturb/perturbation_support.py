#!/usr/bin/env python3

import re

import pandas as pd

GRN_FILE = "vcf_GRNTWAS_identified_TFs.tsv"
KNOCK_FILE = "diff_expr_gene_all_data_raw.tsv"
OUTPUT_FILE = "GRNTWAS_KnockTF_Log2FC_Validation.tsv"
FC_THRESHOLD = 0.2


def analyze_tf_tg_support(grn_file, knock_file, fc_threshold):
    df_grn = pd.read_csv(grn_file, sep="\t")
    df_grn = df_grn[df_grn["Identified_TFs"].notna() & (df_grn["Identified_TFs"] != "None")].copy()

    df_ktf = pd.read_csv(
        knock_file, sep="\t", usecols=["TF", "Ensembl_ID", "Log2FC", "Sample_ID"], low_memory=False
    )
    df_ktf["Log2FC"] = pd.to_numeric(df_ktf["Log2FC"], errors="coerce")
    df_ktf = df_ktf.dropna(subset=["Log2FC"])

    ktf_tf_pool = set(df_ktf["TF"].unique())

    expanded_pairs = []
    for _, row in df_grn.iterrows():
        tg_id = str(row["TargetID"]).strip()
        tg_name = row["GeneName"]
        for tf_item in str(row["Identified_TFs"]).split(","):
            match = re.match(r"(.+?)\((.+?)\)", tf_item.strip())
            if match:
                expanded_pairs.append(
                    {"TG_Ensembl": tg_id, "TG_Name": tg_name, "TF_Name": match.group(1)}
                )

    df_pairs = pd.DataFrame(expanded_pairs)

    sig_ktf = df_ktf[df_ktf["Log2FC"].abs() > fc_threshold]
    ktf_support_map = sig_ktf.groupby("TF")["Ensembl_ID"].apply(set).to_dict()

    results = []
    for _, pair in df_pairs.iterrows():
        tf = pair["TF_Name"]
        tg = pair["TG_Ensembl"]
        tf_exists = tf in ktf_tf_pool
        is_supported = False
        max_fc = None
        support_sample = None

        if tf_exists and tf in ktf_support_map and tg in ktf_support_map[tf]:
            is_supported = True
            tmp = df_ktf[(df_ktf["TF"] == tf) & (df_ktf["Ensembl_ID"] == tg)]
            idx_max = tmp["Log2FC"].abs().idxmax()
            max_fc = tmp.loc[idx_max, "Log2FC"]
            support_sample = tmp.loc[idx_max, "Sample_ID"]

        results.append(
            {
                "TF": tf,
                "TG_Name": pair["TG_Name"],
                "TG_Ensembl": tg,
                "TF_in_Database": tf_exists,
                "Supported_by_Exp": is_supported,
                "Max_Log2FC": max_fc,
                "Sample_ID": support_sample,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    results_df = analyze_tf_tg_support(GRN_FILE, KNOCK_FILE, FC_THRESHOLD)
    results_df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Total TF-target pairs: {len(results_df)}")
    print(f"Perturbation-supported pairs: {int(results_df['Supported_by_Exp'].sum())}")
    print(f"Wrote {OUTPUT_FILE}")
