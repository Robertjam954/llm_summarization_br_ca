import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from col_normalize import normalize_columns

# authoritative path (user-provided)
XLSX = r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\Research\Projects\tcga_gene_alteration_dfs_os_xgb\merged_tcga.xlsx"


def load_df():
    candidates = [XLSX,
                  os.path.join('output','merged_genie.xlsx'),
                  os.path.join('output','merged_genie_harmonized.xlsx'),
                  os.path.join('output','merged_genie_harmonized_fixed.csv'),
                  'merged_genie.xlsx']
    last_err = None
    for p in candidates:
        if not p:
            continue
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_excel(p, engine='openpyxl')
            print(f"Loaded data from: {p}")
            return df
        except Exception as e:
            last_err = e
            # try the next candidate
            continue
    # if we reach here, nothing worked
    raise RuntimeError(f"Unable to read any candidate merged_genie file. Last error: {last_err}")


def missingness(df):
    total = len(df)
    rows = []
    for c in df.columns:
        nmiss = int(df[c].isna().sum())
        pct = nmiss / total * 100
        rows.append({'variable': c, 'n_missing': nmiss, 'pct_missing': pct})
    out = pd.DataFrame(rows).sort_values('pct_missing', ascending=False)
    return out
    
#descriptive analysis

def format_patient_characteristics(df, group_col='DFS_STATUS'):
    # Ensure group exists
    if group_col not in df.columns or df[group_col].isna().all():
        # try PFS event fallback
        if 'PFS_EVENT' in df.columns and df['PFS_EVENT'].notna().any():
            df[group_col] = pd.to_numeric(df['PFS_EVENT'], errors='coerce')
        else:
            df[group_col] = pd.NA

    df[group_col] = pd.to_numeric(df[group_col], errors='coerce')
    groups = sorted(df[group_col].dropna().unique())

    # Variables of interest (only keep those present in df)
    cont_candidates = ['AGE','DFS_MONTHS','PFS_MONTHS','OS_MONTHS','DSS_MONTHS','DAYS_LAST_FOLLOWUP','TMB_NONSYNONYMOUS','ANEUPLOIDY_SCORE','MSI_SENSOR_SCORE','BUFFA_HYPOXIA_SCORE','MANTIS','TBL_SCORE']
    cont_vars = [v for v in cont_candidates if v in df.columns]
    ordinal_vars = []
    nominal_candidates = ['AGE_CAT','SEX','RACE','ETHNICITY','SUBTYPE','CANCER_TYPE','CANCER_TYPE_DETAILED','AJCC_PATHOLOGIC_TUMOR_STAGE','MANTIS_BIN','HISTORY_NEOADJUVANT_TRTYN','SOMATIC_STATUS','SAMPLE_TYPE','TUMOR_TYPE','TUMOR_TISSUE_SITE','TISSUE_SOURCE_SITE','TISSUE_SOURCE_SITE_CODE']
    nominal_vars = [v for v in nominal_candidates if v in df.columns]

    # If no groups found, treat whole dataset as a single 'All' group
    use_all_group = False
    if len(groups) == 0:
        use_all_group = True
        groups = ['All']

    table_rows = []
    for v in cont_vars:
        row = {'variable': v}
        for g in groups:
            if use_all_group:
                s = pd.to_numeric(df[v], errors='coerce').dropna()
            else:
                s = pd.to_numeric(df.loc[df[group_col]==g, v], errors='coerce').dropna()
            if s.empty:
                row[f'group_{g}'] = ''
            else:
                    row[f'group_{g}'] = f"{s.mean():.2f} ± {s.std():.2f} (var {s.var():.2f})"
        table_rows.append(row)

    for v in ordinal_vars:
        row = {'variable': v}
        for g in groups:
            if use_all_group:
                s = pd.to_numeric(df[v], errors='coerce').dropna()
            else:
                s = pd.to_numeric(df.loc[df[group_col]==g, v], errors='coerce').dropna()
            if s.empty:
                row[f'group_{g}'] = ''
            else:
                row[f'group_{g}'] = f"{s.median():.2f} (IQR {s.quantile(0.75)-s.quantile(0.25):.2f})"
        table_rows.append(row)

    for v in nominal_vars:
        row = {'variable': v}
        for g in groups:
            if use_all_group:
                s = df[v].dropna().astype(str)
            else:
                s = df.loc[df[group_col]==g, v].dropna().astype(str)
            if s.empty:
                row[f'group_{g}'] = ''
            else:
                mode = s.mode()
                if mode.empty:
                    row[f'group_{g}'] = ''
                else:
                    m = mode.iloc[0]
                    cnt = int((s==m).sum())
                    pct = cnt / len(s) * 100
                    row[f'group_{g}'] = f"{m} (n={cnt}, {pct:.1f}%)"
        table_rows.append(row)

    table_df = pd.DataFrame(table_rows)
    return table_df


def top_genes_threshold(df, group_col='DFS_STATUS', threshold=5):
    gene_cols = [c for c in df.columns if c.startswith('G__')]
    if not gene_cols:
        print('Top genes: no columns beginning with G__ found; skipping gene prevalence tables.')
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    total = len(df)
    for g in gene_cols:
        n = int(df[g].sum())
        if n < threshold:
            continue
        # prevalence by group
        for label in sorted(df[group_col].dropna().unique()):
            sub = df[df[group_col]==label]
            nsub = int(sub.get(g, pd.Series(0)).sum())
            rows.append({'gene': g.replace('G__',''), 'group': label, 'n_mut': nsub, 'n_total_group': len(sub), 'pct_group': (nsub/len(sub)*100 if len(sub)>0 else 0)})
    out = pd.DataFrame(rows)
    # also compute overall prevalence
    overall = []
    for g in gene_cols:
        n = int(df[g].sum())
        if n >= threshold:
            overall.append({'gene': g.replace('G__',''), 'n_overall': n, 'pct_overall': n/total*100})
    overall_df = pd.DataFrame(overall)
    return out, overall_df


def improved_scatter(df, outfile_png, outfile_pdf=None):
    group_col = 'DFS_STATUS'
    if group_col not in df.columns:
        print('Scatter plot skipped: DFS_STATUS not available in dataset.')
        return

    subtype_col = None
    for cand in ['SUBTYPE', 'CANCER_TYPE_DETAILED', 'CANCER_TYPE', 'TUMOR_TYPE']:
        if cand in df.columns:
            subtype_col = cand
            break
    if subtype_col is None:
        print('Scatter plot skipped: no subtype or cancer-type column available.')
        return

    metric_col = None
    for cand in ['TMB_NONSYNONYMOUS', 'ANEUPLOIDY_SCORE', 'BUFFA_HYPOXIA_SCORE', 'MSI_SENSOR_SCORE', 'MANTIS', 'TBL_SCORE']:
        if cand in df.columns:
            metric_col = cand
            break
    if metric_col is None:
        print('Scatter plot skipped: no numeric burden column available for plotting.')
        return

    plot_df = df[[subtype_col, metric_col, group_col]].copy()
    plot_df[group_col] = pd.to_numeric(plot_df[group_col], errors='coerce')
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors='coerce')
    plot_df = plot_df.dropna(subset=[subtype_col, metric_col, group_col])
    if plot_df.empty:
        print('Scatter plot skipped: insufficient data after filtering.')
        return

    plot_df[subtype_col] = plot_df[subtype_col].astype(str)
    order = plot_df[subtype_col].value_counts().index.tolist()

    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=subtype_col, y=metric_col, data=plot_df, order=order, inner=None, color='.9')
    sns.stripplot(x=subtype_col, y=metric_col, hue=group_col, data=plot_df, order=order, jitter=0.25, dodge=True, alpha=0.7, palette='Set1', size=4)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(subtype_col.replace('_', ' '))
    plt.ylabel(metric_col.replace('_', ' '))
    plt.title(f"{metric_col.replace('_', ' ')} by {subtype_col.replace('_', ' ')}")
    plt.legend(title=group_col)
    plt.tight_layout()
    plt.savefig(outfile_png, dpi=300)
    if outfile_pdf:
        plt.savefig(outfile_pdf)
    plt.close()


def _drop_mask_for_ser(df, ser):
    """Return boolean mask of dropped entries (True = drop) for a Series with df context for grouping.
    Drops NA, blank, and values equal to 'unknown', 'na', 'n/a', 'none' (case-insensitive)."""
    if ser is None:
        return pd.Series([True]*len(df), index=df.index)
    if pd.api.types.is_numeric_dtype(ser):
        return ser.isna()
    s = ser.astype(str).str.strip()
    return s.eq('') | s.isna() | s.str.lower().isin(['unknown','na','n/a','none','nan'])


def table1_by_dfs(df, out_dir=os.path.join('output','descriptive'), group_col='DFS_STATUS'):
    """Table 1: patient characteristics and non-genetic tumor variables by DFS group and Total.
    Drops NA/Unknown/blank per covariate and writes CSV/LaTeX and a dropped-counts CSV."""
    os.makedirs(out_dir, exist_ok=True)
    label_map = {0: 'Disease Free', 1: 'Recurred or Progressed'}

    numeric_summary_vars = ['AGE','DFS_MONTHS','PFS_MONTHS','OS_MONTHS','DSS_MONTHS','DAYS_LAST_FOLLOWUP','TMB_NONSYNONYMOUS','ANEUPLOIDY_SCORE','MSI_SENSOR_SCORE','BUFFA_HYPOXIA_SCORE','MANTIS','TBL_SCORE']
    categorical_summary_vars = ['AGE_CAT','SEX','RACE','ETHNICITY','SUBTYPE','CANCER_TYPE','CANCER_TYPE_DETAILED','AJCC_PATHOLOGIC_TUMOR_STAGE','MANTIS_BIN','HISTORY_NEOADJUVANT_TRTYN','SOMATIC_STATUS','SAMPLE_TYPE','TUMOR_TYPE','TUMOR_TISSUE_SITE','TISSUE_SOURCE_SITE','TISSUE_SOURCE_SITE_CODE']

    # First, collapse to patient-level using merge_key or patient_ID if present
    id_col = None
    for cand in ['merge_key', 'patient_ID', 'patient_id']:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        patient_df = df.copy()
        print('Warning: no merge_key/patient_ID found; Table1 will run on original rows (variant-level).')
    else:
        patient_cols = list(dict.fromkeys(numeric_summary_vars + categorical_summary_vars + [group_col]))

        def _agg_numeric(series):
            s = pd.to_numeric(series, errors='coerce')
            return float(s.mean()) if s.notna().any() else pd.NA

        def _agg_first(series):
            s = series.dropna()
            return s.iloc[0] if not s.empty else pd.NA

        agg_map = {}
        for c in patient_cols:
            if c not in df.columns:
                continue
            if c in numeric_summary_vars:
                agg_map[c] = _agg_numeric
            elif c == group_col:
                agg_map[c] = _agg_first
            else:
                agg_map[c] = _agg_first

        if agg_map:
            patient_df = df.groupby(id_col).agg(agg_map).reset_index()
        else:
            patient_df = df.groupby(id_col).size().reset_index(name='count')

        patient_df[group_col] = pd.to_numeric(patient_df.get(group_col), errors='coerce')
        patient_df[group_col] = patient_df[group_col].where(patient_df[group_col].isin([0, 1]), pd.NA)

    working_df = patient_df
    total_n = len(working_df)

    rows = []
    dropped = []
    for v in numeric_summary_vars + categorical_summary_vars:
        if v not in working_df.columns:
            dropped.append({'variable': v, 'n_dropped_total': total_n, 'n_dropped_group_0': total_n, 'n_dropped_group_1': total_n})
            continue

        ser = working_df[v]
        mask_drop = _drop_mask_for_ser(working_df, ser)
        n_drop_total = int(mask_drop.sum())
        if group_col in working_df.columns:
            n_drop_g0 = int(mask_drop[working_df[group_col] == 0].sum())
            n_drop_g1 = int(mask_drop[working_df[group_col] == 1].sum())
        else:
            n_drop_g0 = 0
            n_drop_g1 = 0
        dropped.append({'variable': v, 'n_dropped_total': n_drop_total, 'n_dropped_group_0': n_drop_g0, 'n_dropped_group_1': n_drop_g1})

        ser_clean = ser[~mask_drop]
        if v in numeric_summary_vars:
            for g in [0, 1]:
                if group_col not in working_df.columns:
                    val = ''
                else:
                    vals = pd.to_numeric(working_df.loc[(working_df[group_col] == g) & (~mask_drop), v], errors='coerce').dropna()
                    if vals.empty:
                        val = ''
                    else:
                        med = vals.median()
                        iqr = vals.quantile(0.75) - vals.quantile(0.25)
                        val = f"{med:.2f} (IQR {iqr:.2f})"
                rows.append({'variable': v, 'level': '', 'group': label_map.get(g), 'value': val})
            vals_all = pd.to_numeric(ser_clean, errors='coerce').dropna()
            if vals_all.empty:
                allval = ''
            else:
                med = vals_all.median()
                iqr = vals_all.quantile(0.75) - vals_all.quantile(0.25)
                allval = f"{med:.2f} (IQR {iqr:.2f})"
            rows.append({'variable': v, 'level': '', 'group': 'Total', 'value': allval})
        else:
            if ser_clean.empty:
                continue
            levels = list(pd.Series(ser_clean.astype(str)).value_counts().index)
            for lev in levels:
                for g in [0, 1]:
                    if group_col not in working_df.columns:
                        cnt = 0
                        pct = 0.0
                    else:
                        sub = working_df.loc[(working_df[group_col] == g) & (~mask_drop), v].dropna().astype(str)
                        cnt = int((sub == str(lev)).sum())
                        pct = cnt / len(sub) * 100 if len(sub) > 0 else 0.0
                    rows.append({'variable': v, 'level': str(lev), 'group': label_map.get(g), 'value': f"{cnt} ({pct:.1f}%)"})
                cnt_all = int((ser_clean.astype(str) == str(lev)).sum())
                pct_all = cnt_all / len(ser_clean) * 100 if len(ser_clean) > 0 else 0.0
                rows.append({'variable': v, 'level': str(lev), 'group': 'Total', 'value': f"{cnt_all} ({pct_all:.1f}%)"})

    out = pd.DataFrame(rows)
    dropped_df = pd.DataFrame(dropped)
    dropped_df.to_csv(os.path.join(out_dir, 'table1_dropped_counts.csv'), index=False)

    if out.empty:
        print('Table1: no non-genetic variables available after dropping NA/Unknown/blank.')
        return out, dropped_df

    pivot = out.pivot_table(index=['variable', 'level'], columns='group', values='value', aggfunc=lambda x: ' | '.join(x.astype(str))).reset_index()
    for col in [label_map[0], label_map[1], 'Total']:
        if col not in pivot.columns:
            pivot[col] = ''
    pivot.to_csv(os.path.join(out_dir, 'table1_by_dfs.csv'), index=False)

    n0 = int(working_df[working_df[group_col] == 0].shape[0]) if group_col in working_df.columns else 0
    n1 = int(working_df[working_df[group_col] == 1].shape[0]) if group_col in working_df.columns else 0
    ntot = int(working_df.shape[0])
    tex_lines = []
    tex_lines.append('\\begin{table}[htb]')
    tex_lines.append('\\centering')
    tex_lines.append('\\small')
    tex_lines.append('\\begin{tabular}{llccc}')
    tex_lines.append('\\hline')
    header = "Variable & Level & Disease Free (n={}) & Recurred/Progressed (n={}) & Total (n={})".format(n0, n1, ntot)
    tex_lines.append(header + ' \\\\')
    for _, r in pivot.iterrows():
        var = r['variable']
        lev = r['level'] if r['level'] and str(r['level']) != 'nan' else ''
        a = r.get(label_map[0], '')
        b = r.get(label_map[1], '')
        t = r.get('Total', '')
        tex_lines.append(f"{var} & {lev} & {a} & {b} & {t} " + ' \\\\')
    tex_lines.append('\\hline')
    tex_lines.append('\\end{tabular}')
    tex_lines.append('\\caption{Patient characteristics by DFS status. Categorical variables are reported as n (\\%), continuous variables as median (IQR).}')
    tex_lines.append('\\end{table}')
    with open(os.path.join(out_dir, 'table1_by_dfs.tex'), 'w') as fh:
        fh.write('\n'.join(tex_lines))

    return pivot, dropped_df


def table2_by_dfs(df, out_dir=os.path.join('output', 'descriptive'), group_col='DFS_STATUS', top_genes=20):
    """Table 2: molecular burden and outcome metrics available in the dataset by DFS group and Total."""
    os.makedirs(out_dir, exist_ok=True)
    label_map = {0: 'Disease Free', 1: 'Recurred or Progressed'}

    numeric_vars = ['TMB_NONSYNONYMOUS', 'ANEUPLOIDY_SCORE', 'MSI_SENSOR_SCORE', 'MANTIS', 'BUFFA_HYPOXIA_SCORE', 'TBL_SCORE', 'DFS_MONTHS', 'PFS_MONTHS', 'OS_MONTHS', 'DSS_MONTHS', 'DAYS_LAST_FOLLOWUP']
    categorical_vars = ['MANTIS_BIN', 'HISTORY_NEOADJUVANT_TRTYN', 'SOMATIC_STATUS', 'SAMPLE_TYPE', 'TUMOR_TYPE', 'TUMOR_TISSUE_SITE', 'TISSUE_SOURCE_SITE', 'TISSUE_SOURCE_SITE_CODE', 'OS_STATUS', 'PFS_STATUS', 'DSS_STATUS']

    df[group_col] = pd.to_numeric(df.get(group_col), errors='coerce')
    df[group_col] = df[group_col].where(df[group_col].isin([0, 1]), pd.NA)

    pid_col = None
    for cand in ['merge_key', 'PATIENT_ID', 'patient_ID', 'patient_id']:
        if cand in df.columns:
            pid_col = cand
            break
    if pid_col is None:
        print('Warning: no patient identifier found; Table2 will run on original rows (variant-level).')
        patient_df = df.copy()
    else:
        patient_cols = list(dict.fromkeys(numeric_vars + categorical_vars + [group_col]))

        def _agg_numeric(series):
            s = pd.to_numeric(series, errors='coerce')
            return float(s.mean()) if s.notna().any() else pd.NA

        def _agg_first(series):
            s = series.dropna()
            return s.iloc[0] if not s.empty else pd.NA

        agg_map = {}
        for c in patient_cols:
            if c not in df.columns:
                continue
            if c in numeric_vars:
                agg_map[c] = _agg_numeric
            elif c == group_col:
                agg_map[c] = _agg_first
            else:
                agg_map[c] = _agg_first

        if agg_map:
            patient_df = df.groupby(pid_col).agg(agg_map).reset_index()
        else:
            patient_df = df.groupby(pid_col).size().reset_index(name='count')

    total_n = len(patient_df)
    rows = []
    dropped = []

    available_numeric = [v for v in numeric_vars if v in patient_df.columns]
    available_categorical = [v for v in categorical_vars if v in patient_df.columns]

    for v in available_numeric + available_categorical:
        ser = patient_df[v]
        mask_drop = _drop_mask_for_ser(patient_df, ser)
        n_drop_total = int(mask_drop.sum())
        if group_col in patient_df.columns:
            n_drop_g0 = int(mask_drop[patient_df[group_col] == 0].sum())
            n_drop_g1 = int(mask_drop[patient_df[group_col] == 1].sum())
        else:
            n_drop_g0 = 0
            n_drop_g1 = 0
        dropped.append({'variable': v, 'n_dropped_total': n_drop_total, 'n_dropped_group_0': n_drop_g0, 'n_dropped_group_1': n_drop_g1})

        ser_clean = ser[~mask_drop]
        if v in available_numeric:
            for g in [0, 1]:
                if group_col not in patient_df.columns:
                    val = ''
                else:
                    vals = pd.to_numeric(patient_df.loc[(patient_df[group_col] == g) & (~mask_drop), v], errors='coerce').dropna()
                    if vals.empty:
                        val = ''
                    else:
                        med = vals.median()
                        iqr = vals.quantile(0.75) - vals.quantile(0.25)
                        val = f"{med:.2f} (IQR {iqr:.2f})"
                rows.append({'variable': v, 'level': '', 'group': label_map.get(g), 'value': val})
            vals_all = pd.to_numeric(ser_clean, errors='coerce').dropna()
            if vals_all.empty:
                allval = ''
            else:
                med = vals_all.median()
                iqr = vals_all.quantile(0.75) - vals_all.quantile(0.25)
                allval = f"{med:.2f} (IQR {iqr:.2f})"
            rows.append({'variable': v, 'level': '', 'group': 'Total', 'value': allval})
        else:
            if ser_clean.empty:
                continue
            levels = list(pd.Series(ser_clean.astype(str)).value_counts().index)
            for lev in levels:
                for g in [0, 1]:
                    if group_col not in patient_df.columns:
                        cnt = 0
                        pct = 0.0
                    else:
                        sub = patient_df.loc[(patient_df[group_col] == g) & (~mask_drop), v].dropna().astype(str)
                        cnt = int((sub == str(lev)).sum())
                        pct = cnt / len(sub) * 100 if len(sub) > 0 else 0.0
                    rows.append({'variable': v, 'level': str(lev), 'group': label_map.get(g), 'value': f"{cnt} ({pct:.1f}%)"})
                cnt_all = int((ser_clean.astype(str) == str(lev)).sum())
                pct_all = cnt_all / len(ser_clean) * 100 if len(ser_clean) > 0 else 0.0
                rows.append({'variable': v, 'level': str(lev), 'group': 'Total', 'value': f"{cnt_all} ({pct_all:.1f}%)"})

    out = pd.DataFrame(rows)
    dropped_df = pd.DataFrame(dropped)
    dropped_df.to_csv(os.path.join(out_dir, 'table2_dropped_counts.csv'), index=False)

    if out.empty:
        print('Table2: no molecular/outcome variables available after dropping NA/Unknown/blank.')
        return out, dropped_df

    pivot = out.pivot_table(index=['variable', 'level'], columns='group', values='value', aggfunc=lambda x: ' | '.join(x.astype(str))).reset_index()
    for col in [label_map[0], label_map[1], 'Total']:
        if col not in pivot.columns:
            pivot[col] = ''
    pivot.to_csv(os.path.join(out_dir, 'table2_by_dfs.csv'), index=False)

    n0 = int(patient_df[patient_df[group_col] == 0].shape[0]) if group_col in patient_df.columns else 0
    n1 = int(patient_df[patient_df[group_col] == 1].shape[0]) if group_col in patient_df.columns else 0
    ntot = int(patient_df.shape[0])
    tex_lines = []
    tex_lines.append('\\begin{table}[htb]')
    tex_lines.append('\\centering')
    tex_lines.append('\\small')
    tex_lines.append('\\begin{tabular}{llccc}')
    tex_lines.append('\\hline')
    header = "Variable & Level & Disease Free (n={}) & Recurred/Progressed (n={}) & Total (n={})".format(n0, n1, ntot)
    tex_lines.append(header + ' \\\\')
    for _, r in pivot.iterrows():
        var = r['variable']
        lev = r['level'] if r['level'] and str(r['level']) != 'nan' else ''
        a = r.get(label_map[0], '')
        b = r.get(label_map[1], '')
        t = r.get('Total', '')
        tex_lines.append(f"{var} & {lev} & {a} & {b} & {t} " + ' \\\\')
    tex_lines.append('\\hline')
    tex_lines.append('\\end{tabular}')
    tex_lines.append('\\caption{Molecular burden and outcome metrics by DFS status. Continuous variables are reported as median (IQR); categorical variables are reported as n (\\%).}')
    tex_lines.append('\\end{table}')
    with open(os.path.join(out_dir, 'table2_by_dfs.tex'), 'w') as fh:
        fh.write('\n'.join(tex_lines))

    return pivot, dropped_df


def main():
    df = load_df()
    df = normalize_columns(df)
    os.makedirs(os.path.join('output','descriptive'), exist_ok=True)
    os.makedirs(os.path.join('output','figures'), exist_ok=True)

    # missingness
    miss = missingness(df)
    miss.to_csv(os.path.join('output','descriptive','missingness.csv'), index=False)

    # formatted patient characteristics
    char = format_patient_characteristics(df, group_col='DFS_STATUS')
    char.to_csv(os.path.join('output','descriptive','patient_characteristics_formatted.csv'), index=False)

    # create Table 1 (non-genetic covariates) and Table 2 (genetic summaries)
    t1, t1_dropped = table1_by_dfs(df)
    t2, t2_dropped = table2_by_dfs(df)
    # keep the older formatted patient characteristics as supplemental output
    # char already written above

    # top genes
    top_groups, overall = top_genes_threshold(df, group_col='DFS_STATUS', threshold=5)
    top_groups.to_csv(os.path.join('output','descriptive','top_genes_by_group_threshold5.csv'), index=False)
    overall.to_csv(os.path.join('output','descriptive','top_genes_overall_threshold5.csv'), index=False)

    # improved scatter/violin plot
    out_png = os.path.join('output','figures','dfs_scatter_violin.png')
    out_pdf = os.path.join('output','figures','dfs_scatter_violin.pdf')
    improved_scatter(df, out_png, out_pdf)

    print('Wrote formatted patient table, missingness CSV, top-gene tables, and improved scatter/violin plots.')

#variant panel patient level plots
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from col_normalize import normalize_columns


def find_col(df, candidates):
    for c in candidates:
        for col in df.columns:
            if isinstance(col, str) and col.strip().upper() == c.strip().upper():
                return col
    for c in candidates:
        for col in df.columns:
            if isinstance(col, str) and c.strip().upper() in col.strip().upper():
                return col
    return None


def main():
    path = os.path.join('datasets_analysis_dictionary','merged_genie.xlsx')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_excel(path, engine='openpyxl')
    df = normalize_columns(df)

    sample_col = find_col(df, ['SAMPLE_ID','Tumor_Sample_Barcode','SAMPLE_BARCODE','PATIENT_ID','PATIENT'])
    gene_col = find_col(df, ['HUGO_SYMBOL','Hugo_Symbol','Hugo_Symbol_list','HUGO_SYMBOL_LIST'])
    vartype_col = find_col(df, ['Variant_Type','VARIANT_TYPE'])
    consequence_col = find_col(df, ['Consequence','CONSEQUENCE'])

    if sample_col is None:
        raise KeyError('sample id column not found')

    # Build per-patient Variant_Type and Consequence sets
    # If mutation-level data has multiple rows per sample, we'll aggregate unique types per patient and then create patient-level proportions

    # patient -> set of variant types
    vt_map = {}
    c_map = {}
    # iterate mutation records
    for _,row in df.iterrows():
        sid = row.get(sample_col)
        if pd.isna(sid):
            continue
        if vartype_col:
            vt = row.get(vartype_col)
            if pd.notna(vt):
                vt_map.setdefault(sid, set()).add(str(vt))
        if consequence_col:
            cs = row.get(consequence_col)
            if pd.notna(cs):
                c_map.setdefault(sid, set()).add(str(cs))

    # patient-level data frame
    patient_grp = df[[sample_col,'DFS_STATUS']].drop_duplicates(subset=[sample_col]).set_index(sample_col)['DFS_STATUS']
    # variant type counts per DFS group
    rows_vt = []
    rows_c = []
    groups = sorted(df['DFS_STATUS'].dropna().unique())
    total_per_group = {g: int((patient_grp==g).sum()) for g in groups}

    # collect all variant types and consequences seen
    all_vt = set()
    all_c = set()
    for sid, s in vt_map.items():
        all_vt.update(s)
    for sid, s in c_map.items():
        all_c.update(s)

    for vt in sorted(all_vt):
        row = {'variant_type': vt}
        for g in groups:
            cnt = 0
            for sid, s in vt_map.items():
                if g == patient_grp.get(sid, None) and vt in s:
                    cnt += 1
            total = total_per_group.get(g, 1)
            row[f'prop_group_{g}'] = cnt/total*100
        rows_vt.append(row)

    for c in sorted(all_c):
        row = {'consequence': c}
        for g in groups:
            cnt = 0
            for sid, s in c_map.items():
                if g == patient_grp.get(sid, None) and c in s:
                    cnt += 1
            total = total_per_group.get(g, 1)
            row[f'prop_group_{g}'] = cnt/total*100
        rows_c.append(row)

    vt_df = pd.DataFrame(rows_vt)
    c_df = pd.DataFrame(rows_c)
    os.makedirs(os.path.join('output','descriptive'), exist_ok=True)
    vt_df.to_csv(os.path.join('output','descriptive','variant_type_prop_by_dfs_patient_level.csv'), index=False)
    c_df.to_csv(os.path.join('output','descriptive','consequence_prop_by_dfs_patient_level.csv'), index=False)

    # simple barplots
    os.makedirs(os.path.join('output','figures'), exist_ok=True)
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    if not vt_df.empty:
        ax = axes[0]
        cols = [c for c in vt_df.columns if c.startswith('prop_group_')]
        x = np.arange(len(vt_df))
        width = 0.35
        for i,g in enumerate(groups):
            ax.bar(x + (i-0.5)*width, vt_df[f'prop_group_{g}'], width, label=f'DFS={g}')
        ax.set_xticks(x)
        ax.set_xticklabels(vt_df['variant_type'], rotation=45, ha='right')
        ax.set_title('Patient-level Variant_Type by DFS')
    else:
        axes[0].text(0.5,0.5,'No data', ha='center')

    if not c_df.empty:
        ax = axes[1]
        cols = [c for c in c_df.columns if c.startswith('prop_group_')]
        x = np.arange(len(c_df))
        width = 0.35
        for i,g in enumerate(groups):
            ax.bar(x + (i-0.5)*width, c_df[f'prop_group_{g}'], width, label=f'DFS={g}')
        ax.set_xticks(x)
        ax.set_xticklabels(c_df['consequence'], rotation=45, ha='right')
        ax.set_title('Patient-level Consequence by DFS')
    else:
        axes[1].text(0.5,0.5,'No data', ha='center')

    plt.tight_layout()
    out = os.path.join('output','figures','variant_panel_patient_level.png')
    plt.savefig(out, dpi=300)
    print('Wrote', out)


if __name__ == '__main__':
    main()

#variant panel grouped by DFS
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from col_normalize import normalize_columns


def find_col(df, candidates):
    for c in candidates:
        for col in df.columns:
            if isinstance(col, str) and col.strip().upper() == c.strip().upper():
                return col
    # fallback: look for partial matches
    for c in candidates:
        for col in df.columns:
            if isinstance(col, str) and c.strip().upper() in col.strip().upper():
                return col
    return None


def patient_gene_presence(df, sample_col, gene_col):
    # Build mapping sample -> set of genes mutated
    dfg = df[[sample_col, gene_col]].dropna()
    # ensure strings
    dfg[gene_col] = dfg[gene_col].astype(str)
    grouped = dfg.groupby(sample_col)[gene_col].agg(lambda s: set([g.strip() for v in s for g in v.split(';') if g.strip()]))
    return grouped


def main():
    path = os.path.join('datasets_analysis_dictionary', 'merged_genie.xlsx')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_excel(path, engine='openpyxl')
    df = normalize_columns(df)

    # find key columns
    sample_col = find_col(df, ['SAMPLE_ID', 'Tumor_Sample_Barcode', 'SAMPLE_BARCODE', 'PATIENT_ID', 'PATIENT'])
    gene_col = find_col(df, ['HUGO_SYMBOL', 'Hugo_Symbol', 'Hugo_Symbol_list', 'HUGO_SYMBOL_LIST'])
    varclass_col = find_col(df, ['Variant_Classification', 'VARIANT_CLASSIFICATION'])
    vartype_col = find_col(df, ['Variant_Type', 'VARIANT_TYPE'])
    consequence_col = find_col(df, ['Consequence', 'CONSEQUENCE'])

    if 'DFS_STATUS' not in df.columns:
        raise KeyError('DFS_STATUS column not found in merged_genie.xlsx; please ensure DFS_STATUS exists')

    # Ensure DFS_STATUS numeric
    df['DFS_STATUS'] = pd.to_numeric(df['DFS_STATUS'], errors='coerce')

    os.makedirs(os.path.join('output','figures'), exist_ok=True)
    os.makedirs(os.path.join('output','descriptive'), exist_ok=True)

    # Panel 1: Hugo_Symbol gene prevalence per patient by DFS group
    if gene_col and sample_col:
        pg = patient_gene_presence(df, sample_col, gene_col)
        # pg: Series indexed by sample -> set of genes
        samp_to_group = df[[sample_col,'DFS_STATUS']].drop_duplicates(subset=[sample_col]).set_index(sample_col)['DFS_STATUS']
        # build per-group patient lists
        groups = sorted(df['DFS_STATUS'].dropna().unique())
        # build gene->counts per group
        gene_counts = {}
        total_per_group = {g: int((samp_to_group==g).sum()) for g in groups}
        # count patients per gene in each group
        for sample, genes in pg.items():
            grp = samp_to_group.get(sample, None)
            if pd.isna(grp):
                continue
            for g in genes:
                gene_counts.setdefault(g, Counter()).update([grp])
    # Convert to DataFrame
        # ensure consistent order: top genes overall
        overall_counts = Counter()
        for g, cnts in gene_counts.items():
            overall_counts[g] = sum(cnts.values())
        top_genes = [g for g,_ in overall_counts.most_common(20)]
        rows = []
        for g in top_genes:
            row = {'gene': g}
            for grp in groups:
                n = gene_counts.get(g, Counter()).get(grp, 0)
                total = total_per_group.get(grp, 1)
                row[f'prop_group_{grp}'] = n / total * 100
            rows.append(row)
        gene_df = pd.DataFrame(rows)
        gene_df.to_csv(os.path.join('output','descriptive','hugo_prevalence_by_dfs.csv'), index=False)
    else:
        gene_df = None

    # Panel 2: Variant_Type proportions per DFS (mutation-level)
    if vartype_col:
        vdf = df[[vartype_col,'DFS_STATUS']].dropna()
        vt_counts = vdf.groupby('DFS_STATUS')[vartype_col].value_counts().unstack(fill_value=0)
        vt_prop = vt_counts.div(vt_counts.sum(axis=1), axis=0) * 100
        vt_prop.to_csv(os.path.join('output','descriptive','variant_type_prop_by_dfs.csv'))
    else:
        vt_prop = None

    # Panel 3: Consequence proportions per DFS (mutation-level)
    if consequence_col:
        cdf = df[[consequence_col,'DFS_STATUS']].dropna()
        c_counts = cdf.groupby('DFS_STATUS')[consequence_col].value_counts().unstack(fill_value=0)
        c_prop = c_counts.div(c_counts.sum(axis=1), axis=0) * 100
        c_prop.to_csv(os.path.join('output','descriptive','consequence_prop_by_dfs.csv'))
    else:
        c_prop = None

    # plotting
    fig, axes = plt.subplots(1,3, figsize=(18,6))

    # Panel 1 plot
    if gene_df is not None and not gene_df.empty:
        ax = axes[0]
        x = np.arange(len(gene_df))
        width = 0.35
        for i,grp in enumerate(groups):
            ax.bar(x + (i-0.5)*width, gene_df[f'prop_group_{grp}'], width, label=f'DFS={grp}')
        ax.set_xticks(x)
        ax.set_xticklabels([g if len(g)<=15 else g[:15]+'...' for g in gene_df['gene']], rotation=45, ha='right')
        ax.set_ylabel('Percent patients with mutation (%)')
        ax.set_title('Top genes: patient prevalence by DFS status')
        ax.legend()
    else:
        axes[0].text(0.5,0.5,'No Hugo_Symbol patient-level data found', ha='center')

    # Panel 2 plot
    ax = axes[1]
    if vt_prop is not None and not vt_prop.empty:
        cols = vt_prop.columns.tolist()
        x = np.arange(len(cols))
        width = 0.35
        for i,grp in enumerate(vt_prop.index.tolist()):
            ax.bar(x + (i-0.5)*width, vt_prop.loc[grp, cols].values, width, label=f'DFS={grp}')
        ax.set_xticks(x)
        ax.set_xticklabels(cols, rotation=45, ha='right')
        ax.set_ylabel('Percent mutation records (%)')
        ax.set_title('Variant type proportions by DFS status')
        ax.legend()
    else:
        ax.text(0.5,0.5,'No Variant_Type data found', ha='center')

    # Panel 3 plot
    ax = axes[2]
    if c_prop is not None and not c_prop.empty:
        cols = c_prop.columns.tolist()
        x = np.arange(len(cols))
        width = 0.35
        for i,grp in enumerate(c_prop.index.tolist()):
            ax.bar(x + (i-0.5)*width, c_prop.loc[grp, cols].values, width, label=f'DFS={grp}')
        ax.set_xticks(x)
        ax.set_xticklabels(cols, rotation=45, ha='right')
        ax.set_ylabel('Percent mutation records (%)')
        ax.set_title('Consequence proportions by DFS status')
        ax.legend()
    else:
        ax.text(0.5,0.5,'No Consequence data found', ha='center')

    plt.tight_layout()
    out_png = os.path.join('output','figures','variant_panel_by_dfs.png')
    plt.savefig(out_png, dpi=300)
    print('Wrote', out_png)


if __name__ == '__main__':
    main()

#descriptive plots
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
# make sure scripts/ is importable when running this file directly
sys.path.insert(0, os.path.dirname(__file__))
from col_normalize import normalize_columns


def mode_with_freq(s):
    if s.dropna().empty:
        return (np.nan, 0)
    mode = s.mode()
    if mode.empty:
        return (np.nan, 0)
    m = mode.iloc[0]
    freq = int((s==m).sum())
    return (m, freq)


def summarize_by_group(df, group_col='DFS_STATUS'):
    groups = sorted(df[group_col].dropna().unique())
    rows = []
    for g in groups:
        sub = df[df[group_col]==g]
        n = len(sub)
        # continuous: AGE
        age_mean = sub['AGE'].dropna().astype(float).mean()
        age_var = sub['AGE'].dropna().astype(float).var()
        # ordinal: AJCC_STAGE_NUM median/IQR
        stage = pd.to_numeric(sub.get('AJCC_STAGE_NUM'), errors='coerce')
        stage_median = stage.median()
        stage_iqr = stage.quantile(0.75) - stage.quantile(0.25)
        # nominal: SUBTYPE, ETHNICITY, RACE, CANCER_TYPE_DETAILED
        subtype_mode, subtype_freq = mode_with_freq(sub.get('SUBTYPE', pd.Series()))
        eth_mode, eth_freq = mode_with_freq(sub.get('ETHNICITY_BIN', pd.Series()))
        race_mode, race_freq = mode_with_freq(sub.get('RACE', pd.Series()))
        cancer_mode, cancer_freq = mode_with_freq(sub.get('CANCER_TYPE_DETAILED', pd.Series()))
        rows.append({
            'DFS_STATUS': g,
            'n': n,
            'AGE_mean': age_mean,
            'AGE_var': age_var,
            'AJCC_STAGE_median': stage_median,
            'AJCC_STAGE_IQR': stage_iqr,
            'SUBTYPE_mode': subtype_mode,
            'SUBTYPE_mode_freq': subtype_freq,
            'ETHNICITY_mode': eth_mode,
            'ETHNICITY_mode_freq': eth_freq,
            'RACE_mode': race_mode,
            'RACE_mode_freq': race_freq,
            'CANCER_TYPE_mode': cancer_mode,
            'CANCER_TYPE_mode_freq': cancer_freq,
        })
    return pd.DataFrame(rows)

def top_genes_by_group(df, group_col='DFS_STATUS', topk=5):
    groups = sorted(df[group_col].dropna().unique())
    rows = []
    gene_cols = [c for c in df.columns if c.startswith('G__')]
    for g in groups:
        sub = df[df[group_col]==g]
        if gene_cols and len(sub)>0:
            preval = sub[gene_cols].sum().sort_values(ascending=False)
            top = preval.head(topk)
            for gene, cnt in top.items():
                rows.append({'DFS_STATUS':g,'gene':gene.replace('G__',''),'n_mut':int(cnt)})
    return pd.DataFrame(rows)

# genomic SV and mutation
def plot_panels(df, top_genes):
    out_dir = os.path.join('output','figures')
    os.makedirs(out_dir, exist_ok=True)
    panels = plt.figure(figsize=(20,7))
    # Panel A: MSI - pick top 2 most common levels
    ax1 = panels.add_subplot(1,3,1)
    # ensure DFS_STATUS is numeric 0/1-ish and MANTIS_BIN is string
    df['DFS_STATUS'] = pd.to_numeric(df.get('DFS_STATUS'), errors='coerce')
    df['MANTIS_BIN'] = df.get('MANTIS_BIN').fillna('Unknown').astype(str)
    msi_tab = pd.crosstab(df['MANTIS_BIN'], df['DFS_STATUS'])
    # pick two most frequent levels
    msi_levels = msi_tab.sum(axis=1).sort_values(ascending=False).head(2).index.tolist()
    if len(msi_levels) == 0:
        ax1.text(0.5,0.5,'No MANTIS_BIN data', ha='center')
    else:
        # normalize per MANTIS level to show proportion of DFS statuses
        msi_prop = msi_tab.div(msi_tab.sum(axis=1).replace({0:np.nan}), axis=0).fillna(0)
        toplot = [lev for lev in msi_levels if lev in msi_prop.index]
        if toplot:
            msi_prop.loc[toplot].plot(kind='bar', stacked=True, ax=ax1, legend=False, colormap='Set2')
    ax1.set_title('MSI levels (proportion DFS by status)')
    ax1.set_xlabel('MANTIS_BIN')
    ax1.set_ylabel('Proportion')
    ax1.set_ylim(0,1)

    # Panel B: TBL high vs low (use TBL_HIGH if present, else quartile by TBL_SCORE)
    ax2 = panels.add_subplot(1,3,2)
    if 'TBL_HIGH' in df.columns:
        df['TBL_HIGH'] = pd.to_numeric(df.get('TBL_HIGH'), errors='coerce').fillna(0).astype(int)
        tbl_tab = pd.crosstab(df['TBL_HIGH'], df['DFS_STATUS'])
        # pick top two levels
        tbl_levels = tbl_tab.sum(axis=1).sort_values(ascending=False).head(2).index.tolist()
        if tbl_levels:
            tbl_prop = tbl_tab.div(tbl_tab.sum(axis=1).replace({0:np.nan}), axis=0).fillna(0)
            toplot = [lev for lev in tbl_levels if lev in tbl_prop.index]
            if toplot:
                tbl_prop.loc[toplot].plot(kind='bar', stacked=True, ax=ax2, legend=False, colormap='Paired')
    else:
        ax2.text(0.5,0.5,'No TBL data',ha='center')
    ax2.set_title('TBL high vs low (proportion DFS)')
    ax2.set_xlabel('TBL_HIGH')
    ax2.set_ylim(0,1)

    # Panel C: Top genes grouped bars of MSI H vs L per gene (normalized proportion of DFS=1)
    ax3 = panels.add_subplot(1,3,3)
    genes = top_genes
    data = []
    for g in genes:
        col = 'G__'+g
        if col not in df.columns:
            data.append([0,0])
            continue
        sub = df[df[col]==1]
        sub['MANTIS_BIN'] = sub.get('MANTIS_BIN').fillna('Unknown').astype(str)
        msi_tab = pd.crosstab(sub['MANTIS_BIN'], sub['DFS_STATUS'])
        prop = msi_tab.div(msi_tab.sum(axis=1).replace({0:np.nan}), axis=0).fillna(0)
        # pick two most common levels among this subset
        levs = msi_tab.sum(axis=1).sort_values(ascending=False).head(2).index.tolist()
        val0 = prop.loc[levs[0],1] if (len(levs)>0 and levs[0] in prop.index and 1 in prop.columns) else 0
        val1 = prop.loc[levs[1],1] if (len(levs)>1 and levs[1] in prop.index and 1 in prop.columns) else 0
        # ensure consistent ordering: high/low if present else levs
        high = val0
        low = val1
        data.append([high, low])
    # ensure data is shape (n_genes,2)
    if len(data)==0:
        data = np.zeros((len(genes),2))
    else:
        data = np.array([ [d[0] if len(d)>0 else 0, d[1] if len(d)>1 else 0] for d in data ])
    x = np.arange(len(genes))
    width = 0.35
    ax3.bar(x - width/2, data[:,0], width, label='MSI Level 1 (prop DFS=1)')
    ax3.bar(x + width/2, data[:,1], width, label='MSI Level 2 (prop DFS=1)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(genes, rotation=45, ha='right')
    ax3.set_ylabel('Proportion DFS=1')
    ax3.set_title('Top genes: MSI H vs L proportion of DFS=1')
    ax3.legend()

    plt.tight_layout()
    out_png = os.path.join(out_dir,'dfs_covariates_panel.png')
    out_pdf = os.path.join(out_dir,'dfs_covariates_panel_polished.pdf')
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    return out_png


def main():
    # Prefer authoritative merged Excel (merged_genie.xlsx) if present
    # user-provided authoritative path
    xlsx = r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\Research\Projects\genomics_brain_mets_genie_bpc\datasets_analysis_dictionary\merged_genie.xlsx"
    csv = os.path.join('output','fine_gray_ready_from_xlsx_fixed.csv')
    if os.path.exists(xlsx):
        print('Reading', xlsx)
        try:
            df = pd.read_excel(xlsx, engine='openpyxl')
        except Exception:
            # fallback to pandas default
            df = pd.read_excel(xlsx)
    elif os.path.exists(csv):
        print('Reading', csv)
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError('Neither merged_genie.xlsx nor fine_gray_ready_from_xlsx_fixed.csv found in output/')
    # Basic harmonization
    df['AGE'] = pd.to_numeric(df.get('AGE'), errors='coerce')
    df['AJCC_STAGE_NUM'] = pd.to_numeric(df.get('AJCC_STAGE_NUM'), errors='coerce')
    df['MANTIS_BIN'] = df.get('MANTIS_BIN').fillna('Unknown')
    # normalize column names (map Subtpe -> SUBTYPE etc.)
    df = normalize_columns(df)
    # If DFS_STATUS not present or all-NA, fall back to PFS_EVENT (common in file)
    if 'DFS_STATUS' not in df.columns or df['DFS_STATUS'].notna().sum() == 0:
        if 'PFS_EVENT' in df.columns and df['PFS_EVENT'].notna().sum() > 0:
            print('Warning: DFS_STATUS empty — using PFS_EVENT as DFS_STATUS proxy')
            df['DFS_STATUS'] = pd.to_numeric(df.get('PFS_EVENT'), errors='coerce')
            # also set DFS_MONTHS if available
            if 'DFS_MONTHS' not in df.columns and 'PFS_TIME_MONTHS' in df.columns:
                df['DFS_MONTHS'] = df['PFS_TIME_MONTHS']
        else:
            # ensure DFS_STATUS column exists (all NaN)
            df['DFS_STATUS'] = pd.NA
    # ensure TBL flags exist
    if 'TBL_SCORE' in df.columns and 'TBL_HIGH' not in df.columns:
        try:
            q = pd.qcut(df['TBL_SCORE'].rank(method='first'), 4, labels=False, duplicates='drop')
            df['TBL_HIGH'] = (q==3).astype(int)
            df['TBL_LOW'] = (q==0).astype(int)
        except Exception:
            df['TBL_HIGH'] = 0
            df['TBL_LOW'] = 0

    os.makedirs(os.path.join('output','descriptive'), exist_ok=True)
    char_df = summarize_by_group(df, 'DFS_STATUS')
    char_df.to_csv(os.path.join('output','descriptive','patient_characteristics.csv'), index=False)
    top5 = top_genes_by_group(df,'DFS_STATUS',topk=5)
    top5.to_csv(os.path.join('output','descriptive','top5_genes_by_group.csv'), index=False)
    # pick top-3 genes overall for panel C
    gene_cols = [c for c in df.columns if c.startswith('G__')]
    if gene_cols:
        top_overall = df[gene_cols].sum().sort_values(ascending=False).head(3).index.tolist()
        top_genes = [g.replace('G__','') for g in top_overall]
    else:
        top_genes = []
    out_png = plot_panels(df, top_genes)
    print('Wrote patient_characteristics.csv, top5_genes_by_group.csv, and', out_png)

    # --- clustered scatter plot: receptor subtype (x) vs mutation count (y), color by DFS_STATUS ---
    # receptor subtype column candidate names
    subtype_col = None
    for cand in ['RECEPTOR_SUBTYPE','SUBTYPE','RECEPTOR_TYPE','RECEPTOR_SUBTYPE_DETAIL']:
        if cand in df.columns:
            subtype_col = cand
            break
    if subtype_col is None:
        # fallback to SUBTYPE
        subtype_col = 'SUBTYPE'

    # compute mutation count from G__ columns if not present
    gene_cols = [c for c in df.columns if c.startswith('G__')]
    if 'MUT_COUNT' not in df.columns:
            # prefer explicit variant count if present in the merged Excel
            if 'variant_allele_count' in df.columns:
                df['MUT_COUNT'] = pd.to_numeric(df['variant_allele_count'], errors='coerce').fillna(0)
            else:
                if gene_cols:
                    df['MUT_COUNT'] = df[gene_cols].sum(axis=1)
                mc = pd.to_numeric(df.get('MUT_COUNT'), errors='coerce')
                df['MUT_COUNT'] = mc.fillna(0)

    scatter_out = os.path.join('output','figures','dfs_clustered_scatter.png')
    # prepare plotting DataFrame
    plot_df = df[[subtype_col,'MUT_COUNT','DFS_STATUS']].copy()
    plot_df = plot_df.dropna(subset=[subtype_col,'MUT_COUNT','DFS_STATUS'])
    # make subtype categorical and order by frequency
    plot_df[subtype_col] = plot_df[subtype_col].astype(str)
    order = plot_df[subtype_col].value_counts().index.tolist()
    import seaborn as sns
    sns.set(style='whitegrid')
    plt.figure(figsize=(10,6))
    # map subtype to numeric positions for jitter
    pos = {cat:i for i,cat in enumerate(order)}
    x = plot_df[subtype_col].map(pos)
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(x)) - 0.5) * 0.6
    xj = x + jitter
    cmap = {0: 'tab:blue', 1: 'tab:orange'}
    colors = [cmap.get(int(v), 'gray') for v in plot_df['DFS_STATUS']]
    plt.scatter(xj, plot_df['MUT_COUNT'], c=colors, alpha=0.6, s=20)
    # x ticks
    plt.xticks(range(len(order)), order, rotation=45, ha='right')
    plt.xlabel('Receptor subtype')
    plt.ylabel('Mutation count')
    plt.title('Mutation count by receptor subtype (colored by DFS_STATUS)')
    # legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color='tab:blue', label='DFS=0'), mpatches.Patch(color='tab:orange', label='DFS=1')]
    plt.legend(handles=handles)
    plt.tight_layout()
    plt.savefig(scatter_out)
    print('Wrote', scatter_out)


if __name__ == '__main__':
    main()

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from col_normalize import normalize_columns


def find_col(df, candidates):
    for c in candidates:
        for col in df.columns:
            if isinstance(col, str) and col.strip().upper() == c.strip().upper():
                return col
    for c in candidates:
        for col in df.columns:
            if isinstance(col, str) and c.strip().upper() in col.strip().upper():
                return col
    return None


def main():
    path = os.path.join('datasets_analysis_dictionary','merged_genie.xlsx')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_excel(path, engine='openpyxl')
    df = normalize_columns(df)

    sample_col = find_col(df, ['SAMPLE_ID','Tumor_Sample_Barcode','SAMPLE_BARCODE','PATIENT_ID','PATIENT'])
    gene_col = find_col(df, ['HUGO_SYMBOL','Hugo_Symbol','Hugo_Symbol_list','HUGO_SYMBOL_LIST'])
    vartype_col = find_col(df, ['Variant_Type','VARIANT_TYPE'])
    consequence_col = find_col(df, ['Consequence','CONSEQUENCE'])

    if sample_col is None:
        raise KeyError('sample id column not found')

    # Build per-patient Variant_Type and Consequence sets
    # If mutation-level data has multiple rows per sample, we'll aggregate unique types per patient and then create patient-level proportions

    # patient -> set of variant types
    vt_map = {}
    c_map = {}
    # iterate mutation records
    for _,row in df.iterrows():
        sid = row.get(sample_col)
        if pd.isna(sid):
            continue
        if vartype_col:
            vt = row.get(vartype_col)
            if pd.notna(vt):
                vt_map.setdefault(sid, set()).add(str(vt))
        if consequence_col:
            cs = row.get(consequence_col)
            if pd.notna(cs):
                c_map.setdefault(sid, set()).add(str(cs))

    # patient-level data frame
    patient_grp = df[[sample_col,'DFS_STATUS']].drop_duplicates(subset=[sample_col]).set_index(sample_col)['DFS_STATUS']
    # variant type counts per DFS group
    rows_vt = []
    rows_c = []
    groups = sorted(df['DFS_STATUS'].dropna().unique())
    total_per_group = {g: int((patient_grp==g).sum()) for g in groups}

    # collect all variant types and consequences seen
    all_vt = set()
    all_c = set()
    for sid, s in vt_map.items():
        all_vt.update(s)
    for sid, s in c_map.items():
        all_c.update(s)

    for vt in sorted(all_vt):
        row = {'variant_type': vt}
        for g in groups:
            cnt = 0
            for sid, s in vt_map.items():
                if g == patient_grp.get(sid, None) and vt in s:
                    cnt += 1
            total = total_per_group.get(g, 1)
            row[f'prop_group_{g}'] = cnt/total*100
        rows_vt.append(row)

    for c in sorted(all_c):
        row = {'consequence': c}
        for g in groups:
            cnt = 0
            for sid, s in c_map.items():
                if g == patient_grp.get(sid, None) and c in s:
                    cnt += 1
            total = total_per_group.get(g, 1)
            row[f'prop_group_{g}'] = cnt/total*100
        rows_c.append(row)

    vt_df = pd.DataFrame(rows_vt)
    c_df = pd.DataFrame(rows_c)
    os.makedirs(os.path.join('output','descriptive'), exist_ok=True)
    vt_df.to_csv(os.path.join('output','descriptive','variant_type_prop_by_dfs_patient_level.csv'), index=False)
    c_df.to_csv(os.path.join('output','descriptive','consequence_prop_by_dfs_patient_level.csv'), index=False)

    # simple barplots
    os.makedirs(os.path.join('output','figures'), exist_ok=True)
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    if not vt_df.empty:
        ax = axes[0]
        cols = [c for c in vt_df.columns if c.startswith('prop_group_')]
        x = np.arange(len(vt_df))
        width = 0.35
        for i,g in enumerate(groups):
            ax.bar(x + (i-0.5)*width, vt_df[f'prop_group_{g}'], width, label=f'DFS={g}')
        ax.set_xticks(x)
        ax.set_xticklabels(vt_df['variant_type'], rotation=45, ha='right')
        ax.set_title('Patient-level Variant_Type by DFS')
    else:
        axes[0].text(0.5,0.5,'No data', ha='center')

    if not c_df.empty:
        ax = axes[1]
        cols = [c for c in c_df.columns if c.startswith('prop_group_')]
        x = np.arange(len(c_df))
        width = 0.35
        for i,g in enumerate(groups):
            ax.bar(x + (i-0.5)*width, c_df[f'prop_group_{g}'], width, label=f'DFS={g}')
        ax.set_xticks(x)
        ax.set_xticklabels(c_df['consequence'], rotation=45, ha='right')
        ax.set_title('Patient-level Consequence by DFS')
    else:
        axes[1].text(0.5,0.5,'No data', ha='center')

    plt.tight_layout()
    out = os.path.join('output','figures','variant_panel_patient_level.png')
    plt.savefig(out, dpi=300)
    print('Wrote', out)


if __name__ == '__main__':
    main()
