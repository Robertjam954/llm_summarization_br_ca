#!/usr/bin/env python3
"""
Summarize feature contributions (SHAP-like) for a trained XGBoost AFT survival model.

- Reconstructs feature matrix using the same logic as survival_and_xgb_analysis.py
- Loads the saved model JSON and computes pred_contribs
- Saves per-feature mean|contrib| ranking and per-sample contrib matrix
- Produces a barplot of top-N features by mean|contrib|

Inputs:
  --xlsx: datasets_analysis_dictionary/merged_genie.xlsx
  --model-json: output/survival/xgb/xgb_aft_model.json
  --per-gene-csv: output/fine_gray_python/per_gene_cs_cox_sksurv_bootstrap.csv (top-N genes; optional)

Outputs under output/survival/xgb/ by default:
  - shap_contribs.csv          (N x (F+1) matrix, last column is bias term)
  - shap_feature_ranking.csv   (feature, mean_abs_contrib)
  - shap_topN_bar.png          (bar chart of top features)
"""
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(__file__))
try:
    from col_normalize import normalize_columns
except Exception:
    def normalize_columns(df):
        return df

try:
    import xgboost as xgb
except Exception:
    raise SystemExit('xgboost is required for this script')


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def detect_col(df: pd.DataFrame, candidates):
    lut = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lut:
            return lut[cand.lower()]
    return None


def to_numeric_safe(s):
    return pd.to_numeric(s, errors='coerce')


def load_top_genes(csv_path: str, topn=10):
    if not os.path.exists(csv_path):
        return []
    try:
        pg = pd.read_csv(csv_path)
        if 'n' in pg.columns:
            pg = pg.sort_values('n', ascending=False)
        return pg['gene'].head(topn).dropna().astype(str).tolist() if 'gene' in pg.columns else []
    except Exception:
        return []


def build_gene_indicators(df: pd.DataFrame, top_genes: list):
    # Try to parse HUGO gene list if present
    hugo_col = None
    for c in df.columns:
        if isinstance(c, str) and c.strip().upper().startswith('HUGO'):
            hugo_col = c
            break
    if hugo_col is not None:
        parsed = [set([g.strip() for g in re.split(r'[;|,]', str(v)) if g.strip()]) for v in df[hugo_col].fillna('')]
        for g in top_genes:
            gcol = 'G__' + re.sub(r"\s+","_", g)
            if gcol not in df.columns:
                df[gcol] = [1 if g in s else 0 for s in parsed]
    return df


def prepare_common_covariates(df: pd.DataFrame):
    # Basic numeric covariates with defensive casting
    for col in ['BUFFA_HYPOXIA_SCORE','MUT_COUNT','TBL_LOW','TBL_HIGH','AGE','AJCC_STAGE_NUM','ETHNICITY_BIN']:
        if col not in df.columns:
            df[col] = 0
        df[col] = to_numeric_safe(df[col]).fillna(0)

    # If MUT_COUNT missing, derive from gene indicators
    if 'MUT_COUNT' not in df.columns:
        gene_cols = [c for c in df.columns if isinstance(c,str) and c.startswith('G__')]
        df['MUT_COUNT'] = df[gene_cols].sum(axis=1) if gene_cols else 0

    # dummies for MANTIS_BIN and SUBTYPE
    if 'MANTIS_BIN' in df.columns and df['MANTIS_BIN'].dtype.name == 'category':
        df['MANTIS_BIN'] = df['MANTIS_BIN'].astype(str)
    if 'SUBTYPE' in df.columns and df['SUBTYPE'].dtype.name == 'category':
        df['SUBTYPE'] = df['SUBTYPE'].astype(str)

    mantis = pd.get_dummies(df.get('MANTIS_BIN', 'Unknown'), prefix='MANTIS', drop_first=True)
    subtype = pd.get_dummies(df.get('SUBTYPE', 'Unknown'), prefix='SUBTYPE', drop_first=True)

    cov_base = df[['BUFFA_HYPOXIA_SCORE','MUT_COUNT','TBL_LOW','TBL_HIGH','AGE','AJCC_STAGE_NUM','ETHNICITY_BIN']].copy()
    X_common = pd.concat([cov_base.reset_index(drop=True), mantis.reset_index(drop=True), subtype.reset_index(drop=True)], axis=1)
    return X_common


def main():
    p = argparse.ArgumentParser(description='XGB AFT SHAP-like summary')
    p.add_argument('--xlsx', default=os.path.join('datasets_analysis_dictionary','merged_genie.xlsx'))
    p.add_argument('--model-json', default=os.path.join('output','survival','xgb','xgb_aft_model.json'))
    p.add_argument('--per-gene-csv', default=os.path.join('output','fine_gray_python','per_gene_cs_cox_sksurv_bootstrap.csv'))
    p.add_argument('--topn-genes', type=int, default=10)
    p.add_argument('--outdir', default=os.path.join('output','survival','xgb'))
    p.add_argument('--topn-plot', type=int, default=20)
    args = p.parse_args()

    ensure_dir(args.outdir)

    if not os.path.exists(args.xlsx):
        raise FileNotFoundError(args.xlsx)
    if not os.path.exists(args.model_json):
        raise FileNotFoundError(args.model_json)

    df = pd.read_excel(args.xlsx, engine='openpyxl')
    df = normalize_columns(df)

    # build gene indicators from top genes (optional)
    top_genes = load_top_genes(args.per_gene_csv, topn=args.topn_genes)
    df = build_gene_indicators(df, top_genes)

    # prepare features (must match training logic)
    X_common = prepare_common_covariates(df)
    gene_cols = [c for c in df.columns if isinstance(c,str) and c.startswith('G__')]
    feat_cols = list(X_common.columns) + gene_cols

    X = pd.concat([X_common, df[gene_cols]], axis=1) if gene_cols else X_common
    X = X.fillna(0.0).astype(float)

    # load model and compute pred_contribs
    bst = xgb.Booster()
    bst.load_model(args.model_json)

    dmat = xgb.DMatrix(X.values)
    contrib = bst.predict(dmat, pred_contribs=True)
    # contrib shape: (n_samples, n_features + 1) last is bias term

    # save matrix (with feature names + bias)
    feat_names = list(feat_cols) + ['bias']
    contrib_df = pd.DataFrame(contrib, columns=feat_names)
    contrib_df.to_csv(os.path.join(args.outdir, 'shap_contribs.csv'), index=False)

    # feature ranking by mean|contrib| (exclude bias)
    abs_mean = np.abs(contrib[:, :-1]).mean(axis=0)
    ranking = pd.DataFrame({'feature': feat_cols, 'mean_abs_contrib': abs_mean})
    ranking = ranking.sort_values('mean_abs_contrib', ascending=False)
    ranking.to_csv(os.path.join(args.outdir, 'shap_feature_ranking.csv'), index=False)

    # barplot top-N
    topn = ranking.head(args.topn_plot)
    plt.figure(figsize=(10, max(4, 0.4 * len(topn))))
    plt.barh(topn['feature'][::-1], topn['mean_abs_contrib'][::-1], color='#1f77b4')
    plt.xlabel('Mean |contribution|')
    plt.title('XGB AFT feature importance (SHAP-like)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'shap_topN_bar.png'), dpi=300)
    plt.close()

    print('Wrote SHAP-like outputs to', args.outdir)


if __name__ == '__main__':
    main()
