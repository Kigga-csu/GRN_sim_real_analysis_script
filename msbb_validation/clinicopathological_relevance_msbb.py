#!/usr/bin/env python3

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

RANKING_FILE = "gene_ranking.tsv"
TRAIT_FILE = "input/BM_36_trait.txt"
OUTPUT_DIR = "results/clinicopathological_relevance"
N_BINS = 50

sns.set_style("darkgrid")


def partition_ranking_correlation_trait(ranking_file, trait_file, output_dir, n_bins=N_BINS):
    ranking = pd.read_csv(ranking_file, sep="\t", header=None, index_col=0)
    trait = pd.read_csv(trait_file, sep="\t", header=0, index_col=0)

    overlap_genes = ranking.index.intersection(trait.index)
    data = ranking.loc[overlap_genes, :].copy()
    data["CERAD"] = trait.loc[overlap_genes, "CERAD"]
    data["Braak Score"] = trait.loc[overlap_genes, "Braak Score"]
    data["CDR"] = trait.loc[overlap_genes, "CDR"]

    m = n_bins
    n = int(len(data) / m) - 1
    columns = ["CERAD", "Braak Score", "CDR"]
    plt.figure(figsize=(10, 2.5))

    for j, item in enumerate(columns, start=1):
        bin_matrix = np.zeros([n, m])
        for k in range(m):
            loc1 = k * n
            loc2 = (k + 1) * n
            bin_matrix[:, k] = list(data.loc[data.index[loc1:loc2], item])[0:n]
        bin_matrix = pd.DataFrame(bin_matrix)
        bin_matrix.columns = list(range(m))
        bin_matrix.index = range(n)

        x = np.array(range(m))
        y = np.array(bin_matrix.mean())

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        label = f"r = {round(r_value, 3)}, P = {p_value:.1e}"
        y_fit = intercept + slope * x

        ax = plt.subplot(1, 3, j)
        ax.plot(x, y_fit, linewidth=1, c="red", label=label)
        ax.scatter(x, y, s=4, c="lightcoral")
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_xticklabels([1, 10, 20, 30, 40, 50])
        ax.set_xlabel("Bins", fontsize=12)
        ax.set_ylabel("PCC with " + item, fontsize=12)
        plt.legend(loc=3, frameon=False)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "trait_50groups_trend.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


if __name__ == "__main__":
    partition_ranking_correlation_trait(RANKING_FILE, TRAIT_FILE, OUTPUT_DIR)
