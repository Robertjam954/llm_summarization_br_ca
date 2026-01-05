#!/usr/bin/env python3
"""
Compute top mutated genes and plot per-subtype mutation prevalence in a grid of bar charts.

Outputs saved to output/gene_mutation_plots/
"""
import os
import argparse
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from col_normalize import normalize_columns


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def build_gene_presence(df, gene_list_col='Hugo_Symbol_list', sep=';'):
    s = df[gene_list_col].fillna('').astype(str)
    per_row = []
    all_items = []
    for v in s:
        items = [it.strip() for it in v.split(sep) if it.strip()]
        per_row.append(items)
        all_items.extend(items)
    counts = Counter(all_items)
    return per_row, counts


def top_genes(counts, topk=12):
    return [g for g, _ in counts.most_common(topk)]


def plot_prevalence_by_subtype(df, per_row, top_genes_list, out_png, subtype_col='SUBTYPE'):
    # create binary matrix for top genes
    mat = pd.DataFrame(0, index=df.index, columns=top_genes_list)
    for i, items in enumerate(per_row):
        for it in items:
            if it in mat.columns:
                mat.at[i, it] = 1

    # group by subtype
    groups = df[subtype_col].fillna('<missing>').unique().tolist()
    groups = sorted(groups)
    ncols = 4
    nrows = (len(top_genes_list) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for idx, gene in enumerate(top_genes_list):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        vals = []
        for g in groups:
            mask = df[subtype_col].fillna('<missing>') == g
            if mask.sum() == 0:
                vals.append(0)
            else:
                vals.append(mat.loc[mask, gene].sum() / mask.sum() * 100)
        ax.bar(groups, vals, color='#7fc97f')
        ax.set_title(gene)
        ax.set_ylim(0, max(10, max(vals) * 1.2))
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.set_ylabel('Percent cases (%)')
    # hide unused axes
    total = nrows * ncols
    for idx in range(len(top_genes_list), total):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis('off')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='output/merged_genie_with_mutations.csv')
    p.add_argument('--outdir', default='output/gene_mutation_plots')
    p.add_argument('--topk', type=int, default=12)
    p.add_argument('--subtype_col', default='SUBTYPE')
    args = p.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.input, dtype=str)
    df = normalize_columns(df)
    per_row, counts = build_gene_presence(df)
    top = top_genes(counts, topk=args.topk)
    # save counts
    pd.DataFrame(counts.most_common()).rename(columns={0: 'gene', 1: 'count'}).to_csv(os.path.join(args.outdir, 'gene_counts.csv'), index=False)
    out_png = os.path.join(args.outdir, f'top_{args.topk}_genes_by_subtype.png')
    plot_prevalence_by_subtype(df, per_row, top, out_png, subtype_col=args.subtype_col)
    print('Wrote', out_png)


if __name__ == '__main__':
    main()
