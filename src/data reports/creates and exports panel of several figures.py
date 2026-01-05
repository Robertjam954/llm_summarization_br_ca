#!/usr/bin/env python3
"""
Assemble a multi-panel results figure from available pngs/plots and data.

The script looks for standard outputs under `output/` and arranges them in a grid,
skipping any panels whose source files are missing. Outputs `output/figures/results_figure.png` and PDF.
"""
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def try_open(img_path, max_size=(800,800)):
    if not img_path or not os.path.exists(img_path):
        return None
    try:
        img = Image.open(img_path).convert('RGB')
        img.thumbnail(max_size)
        return img
    except Exception:
        return None


def main():
    outdir = 'output/figures'
    os.makedirs(outdir, exist_ok=True)

    # candidate panels (user attachments suggests panels: subtype pie/summary, KM OS, KM PFS, mutation prevalence, molecular markers barplots, covariate panels)
    candidates = {
        'covariate_grid': 'output/covariate_proportions/covariate_grid.png',
        'km_os': 'output/risk_models/kaplan_os.png',
        'km_pfs': 'output/risk_models/kaplan_pfs.png',
        'gene_prevalence': 'output/gene_mutation_plots/top_12_genes_by_subtype.png',
        'molecular_markers': 'output/gene_mutation_plots/molecular_markers.png',
        'fine_gray_volcano': 'output/fine_gray_plots/volcano.png'
    }

    images = []
    for k,p in candidates.items():
        img = try_open(p)
        if img is not None:
            images.append((k,img))
        else:
            print('Missing',k,'->',p)

    if not images:
        print('No panels found; please generate plots first under output/. Exiting.')
        return

    # layout: try 2 columns
    cols = 2
    rows = (len(images)+cols-1)//cols
    fig_w = 8*cols
    fig_h = 4*rows
    fig, axs = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axs = axs.flatten() if rows*cols>1 else [axs]

    for ax_idx in range(len(axs)):
        ax = axs[ax_idx]
        ax.axis('off')
        if ax_idx < len(images):
            name, img = images[ax_idx]
            ax.imshow(img)
            ax.set_title(name.replace('_',' ').title())
    out_png = os.path.join(outdir, 'results_figure.png')
    out_pdf = os.path.join(outdir, 'results_figure.pdf')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    print('Wrote', out_png, out_pdf)


if __name__ == '__main__':
    main()

