#!/usr/bin/env python3

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from sklearn.linear_model import LinearRegression

RANKING_FILE = "gene_ranking.tsv"
KNOWN_GENE_FILE = "input/gene.txt"
META_FILE = "input/MSBB_clinical.csv"
EXPR_FILE = "input/AMP-AD_MSBB_MSSM_BM_36_sym.tsv"
OUTPUT_DIR = "results/variance_explained"
N_TOP_GENES = 20

sns.set_style("white")

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 26,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 1.2,
        "legend.frameon": False,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
    }
)


def regression_phenotype(x, y):
    reg = LinearRegression().fit(x, y)
    return reg.score(x, y)


def meta_geneExpression_mapping(file_meta, file_geneExpression, key):
    meta = pd.read_csv(file_meta, header=0, index_col=key, sep=",")
    meta = meta[~meta.index.duplicated(keep="first")]
    geneExp = pd.read_csv(file_geneExpression, header=0, index_col=0, sep="\t")
    overlap_sample = list(set(meta.index.intersection(geneExp.columns)))
    metanew = meta.loc[overlap_sample, :]
    geneExpnew = geneExp.loc[:, overlap_sample]
    return metanew, geneExpnew


def variance_explained_single_gene(meta, geneExp, key, gene):
    meta = meta[~meta.index.duplicated(keep="first")]
    y = meta.loc[:, key].dropna()
    result = 0
    if gene in geneExp.index:
        x = geneExp.loc[gene, :].dropna()
        x = x[~x.index.duplicated(keep="first")]
        indexs = y.index.intersection(x.index)
        y = y[indexs]
        x = x[indexs]
        newy = [[y[item]] for item in y.index]
        newx = [[x[item]] for item in y.index]
        result = regression_phenotype(newx, newy)
    return result


def variance_explained(meta, geneExp, key, genes):
    y = meta.loc[:, key].dropna()
    indexs = geneExp.index.intersection(genes)
    x = geneExp.loc[indexs, :].dropna()
    indexs = x.columns.intersection(y.index)
    y = y[indexs]
    X = x[indexs]
    result = 0
    if len(indexs) > 1:
        result = regression_phenotype(X.transpose(), y)
    if len(indexs) == 1:
        newy = [[y[item]] for item in y.index]
        newx = [[geneExp.loc[indexs[0], item]] for item in y.index]
        result = regression_phenotype(newx, newy)
    return result


def variation_barplot(meta, geneExp, key, genes, APOE4, title, output_dir):
    R2 = {}
    genes_to_plot = genes[:N_TOP_GENES]
    for genei in genes_to_plot:
        R2[genei] = variance_explained_single_gene(meta, geneExp, key, genei)

    R2 = pd.Series(R2).sort_values(ascending=False)
    R2["APOE4"] = APOE4
    R2["Top genes"] = variance_explained(meta, geneExp, key, genes)

    y = list(R2.index)[::-1]
    width = list(R2)[::-1]

    fig, ax = plt.subplots(figsize=(20 / 2.54, 24 / 2.54))
    ax.barh(y=y, width=width, color="#00A087")
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.10))
    for i, v in enumerate(width):
        ax.annotate("{:.1%}".format(v), (v + 0.001, i - 0.2), fontsize=26)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_alpha(1.0)
    ax.tick_params(axis="both", colors="black")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    plot_title = title.replace("MSBB_36_", "")
    plt.title(plot_title, fontsize=32, pad=15)
    plt.xlabel("Percentage variation", fontsize=26, labelpad=10)
    plt.tight_layout()

    save_dir = os.path.join(output_dir, key)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, title + ".pdf")
    plt.savefig(save_path)
    plt.close(fig)


def run_variation_analysis(ranking_file, output_dir):
    with open(KNOWN_GENE_FILE) as f:
        known_gene = [line.strip() for line in f]

    rank = pd.read_csv(ranking_file, sep="\t", header=None, index_col=0)
    known_gene = rank.index.intersection(known_gene)
    rank = rank.drop(index=known_gene, errors="ignore")

    meta, geneExp = meta_geneExpression_mapping(META_FILE, EXPR_FILE, "specimenID")

    genes = geneExp.index.intersection(rank.index[0:N_TOP_GENES])
    genes = list(genes)

    CERAD = meta["CERAD"].replace({1: "normal", 2: "definite AD", 3: "Probable AD", 4: "possible AD"})
    CERAD = CERAD.replace({"normal": 1, "definite AD": 4, "Probable AD": 3, "possible AD": 2})
    meta["CERAD"] = CERAD

    phenotype_keys = ["CDR", "CERAD", "Braak"]
    for key in phenotype_keys:
        y = meta[key].dropna()
        x = meta["apoeGenotype"].dropna()
        meta_sub = meta.loc[y.index, :].copy()
        meta_sub[key] = y
        x = x.replace({22: 0, 23: 0, 33: 0, 24: 1, 34: 1, 44: 2})
        indexs = x.index.intersection(y.index)
        y_sub = [[y[item]] for item in indexs]
        x_sub = [[x[item]] for item in indexs]
        APOE4 = regression_phenotype(x_sub, y_sub)
        title = f"MSBB_36_{key}"
        variation_barplot(meta_sub, geneExp, key, genes, APOE4, title, output_dir)

    print("Variance explained analysis complete.")


if __name__ == "__main__":
    run_variation_analysis(RANKING_FILE, OUTPUT_DIR)
