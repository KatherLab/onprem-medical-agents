# Paired-design statistics: runs_N = 3, 5, 10
# Objectives:
# 1) Does the consistency score vary with runs_N (Friedman + Wilcoxon)?
# 2) Does coverage (whether the threshold is met) vary with runs_N (Cochran's Q + McNemar)?
# 3) Does accuracy (correctness within the covered subset) vary with runs_N (Cochran's Q + McNemar)?

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests


def _holm_pairwise(pairs):
    """
    pairs: list of dict with key 'p_raw'
    """
    pvals = [x["p_raw"] for x in pairs]
    reject, p_adj, _, _ = multipletests(pvals, method="holm")
    for i, row in enumerate(pairs):
        row["p_holm"] = p_adj[i]
        row["significant"] = bool(reject[i])
    return pairs


def paired_analysis_runsN(
    df: pd.DataFrame,
    sample_col: str = "sample_id",      # Replace with your unique sample ID column name
    run_col: str = "runs_N",
    score_col: str = "Consistency_Dx",
    correct_col: str = "final_decision", # 0/1, 1=correct
    run_levels=(3, 5, 10),
    threshold=0.85,
):
    """
    Retain only samples in run_levels that are present in all three runs (complete pairing).
    """
    d = df.copy()
    d = d[d[run_col].isin(run_levels)]

    # Ensure every sample has records for all three runs (complete pairing).
    cnt = d.groupby(sample_col)[run_col].nunique()
    keep_ids = cnt[cnt == len(run_levels)].index
    d = d[d[sample_col].isin(keep_ids)].copy()

    # Wide-format tables (one row per sample).
    wide_score = d.pivot(index=sample_col, columns=run_col, values=score_col)
    wide_correct = d.pivot(index=sample_col, columns=run_col, values=correct_col)

    # Fix the column order to run_levels.
    wide_score = wide_score.loc[:, run_levels]
    wide_correct = wide_correct.loc[:, run_levels]

    # -------- 1) consistency score: Friedman + Wilcoxon --------
    fr_stat, fr_p = friedmanchisquare(
        wide_score[run_levels[0]].values,
        wide_score[run_levels[1]].values,
        wide_score[run_levels[2]].values,
    )

    score_pairs = []
    for a, b in combinations(run_levels, 2):
        # Paired Wilcoxon; guard against errors from all-zero differences.
        diff = wide_score[a] - wide_score[b]
        if np.allclose(diff.values, 0):
            p = 1.0
            stat = 0.0
        else:
            stat, p = wilcoxon(wide_score[a], wide_score[b], zero_method="wilcox", alternative="two-sided")
        score_pairs.append({"run_i": a, "run_j": b, "stat": stat, "p_raw": p})
    score_pairs = _holm_pairwise(score_pairs)
    score_pairwise_df = pd.DataFrame(score_pairs)

    # -------- 2) coverage: Cochran's Q + McNemar --------
    # Coverage definition: score >= threshold.
    wide_cov = (wide_score >= threshold).astype(int)

    cq_cov = cochrans_q(wide_cov.values)  # returns object with statistic, pvalue
    cov_pairs = []
    for a, b in combinations(run_levels, 2):
        tab = pd.crosstab(wide_cov[a], wide_cov[b]).reindex(index=[0,1], columns=[0,1], fill_value=0)
        # McNemar for paired binary
        m = mcnemar(tab.values, exact=False, correction=True)
        cov_pairs.append({"run_i": a, "run_j": b, "stat": float(m.statistic), "p_raw": float(m.pvalue)})
    cov_pairs = _holm_pairwise(cov_pairs)
    cov_pairwise_df = pd.DataFrame(cov_pairs)

    # -------- 3) accuracy (within the covered subset): Cochran's Q + McNemar --------
    # Definition: evaluate correctness only for samples with coverage = 1.
    # Set uncovered samples to NaN, then retain only samples covered in all three groups for paired comparison.
    wide_acc = wide_correct.where(wide_cov == 1, np.nan)

    acc_complete = wide_acc.dropna(axis=0, how="any")
    # Return an empty result when there are no comparable samples.
    if len(acc_complete) == 0:
        acc_global = {"statistic": np.nan, "pvalue": np.nan, "n_complete": 0}
        acc_pairwise_df = pd.DataFrame(columns=["run_i", "run_j", "stat", "p_raw", "p_holm", "significant"])
    else:
        acc_bin = acc_complete.astype(int)
        cq_acc = cochrans_q(acc_bin.values)

        acc_pairs = []
        for a, b in combinations(run_levels, 2):
            tab = pd.crosstab(acc_bin[a], acc_bin[b]).reindex(index=[0,1], columns=[0,1], fill_value=0)
            m = mcnemar(tab.values, exact=False, correction=True)
            acc_pairs.append({"run_i": a, "run_j": b, "stat": float(m.statistic), "p_raw": float(m.pvalue)})
        acc_pairs = _holm_pairwise(acc_pairs)
        acc_pairwise_df = pd.DataFrame(acc_pairs)
        acc_global = {"statistic": float(cq_acc.statistic), "pvalue": float(cq_acc.pvalue), "n_complete": len(acc_complete)}

    # Summary table (descriptive statistics).
    summary = []
    n_total = len(wide_score)
    for r in run_levels:
        cov_rate = wide_cov[r].mean()
        # Standard accuracy definition: the correctness rate within the covered subset for this run.
        covered = wide_cov[r] == 1
        acc_rate = wide_correct.loc[covered, r].mean() if covered.sum() > 0 else np.nan
        summary.append({
            "runs_N": r,
            "n_paired_total": n_total,
            "coverage_rate": cov_rate,
            "accuracy_on_covered": acc_rate,
            "n_covered": int(covered.sum()),
        })
    summary_df = pd.DataFrame(summary)

    out = {
        "summary": summary_df,
        "consistency_global_friedman": {"statistic": float(fr_stat), "pvalue": float(fr_p), "n": n_total},
        "consistency_pairwise_wilcoxon": score_pairwise_df,
        "coverage_global_cochranQ": {"statistic": float(cq_cov.statistic), "pvalue": float(cq_cov.pvalue), "n": n_total},
        "coverage_pairwise_mcnemar": cov_pairwise_df,
        "accuracy_global_cochranQ_on_covered_intersection": acc_global,
        "accuracy_pairwise_mcnemar_on_covered_intersection": acc_pairwise_df,
        "wide_score": wide_score,
        "wide_coverage": wide_cov,
        "wide_correct": wide_correct,
    }
    return out


# =========================
# Example usage
# =========================
import pandas as pd

df3 = pd.read_csv("/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/evaluation_logs/63/3runs/63.10/master_feature_dataframe.csv")
df5 = pd.read_csv("/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/evaluation_logs/63/5runs/63.10/master_feature_dataframe.csv")
df10 = pd.read_csv("/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/evaluation_logs/63/10runs_63_104/63.10/master_feature_dataframe.csv")

# If the table has no runs_N column and the 3/5/10 runs come from separate files, concatenate them and add runs_N first.
# This is an example; skip it if runs_N already exists in one table.
df3["runs_N"]=3; df5["runs_N"]=5; df10["runs_N"]=10
df_master = pd.concat([df3, df5, df10], ignore_index=True)

# Important: convert values to 0/1 for statistical tests.
df_master["final_decision"] = df_master["final_decision"].astype(str).str.lower().map({"true": 1, "false": 0})

# Optional: check for values that could not be mapped.
if df_master["final_decision"].isna().any():
    print("Warning: final_decision 有未识别值：")
    print(df_master.loc[df_master["final_decision"].isna(), "final_decision"].head())

dup = (
    df_master.groupby(["hadm_id", "runs_N"])
    .size()
    .reset_index(name="n")
    .query("n > 1")
)
print("重复 (hadm_id, runs_N) 数量:", len(dup))

# res = paired_analysis_runsN(
#     df_master,
#     sample_col="hadm_id",
#     run_col="runs_N",
#     score_col="Consistency_Dx",
#     correct_col="final_decision",
#     run_levels=(3,5,10),
#     threshold=0.85
# )

# print(res["summary"])
# print(res["consistency_global_friedman"])
# print(res["coverage_global_cochranQ"])
# print(res["accuracy_global_cochranQ_on_covered_intersection"])

thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
rows = []
for th in thresholds:
    r = paired_analysis_runsN(
        df_master,
        sample_col="hadm_id",
        run_col="runs_N",
        score_col="Consistency_Dx",
        correct_col="final_decision",
        run_levels=(3,5,10),
        threshold=th
    )

    s = r["summary"].set_index("runs_N")

    rows.append({
        "threshold": th,
        "p_consistency_friedman": r["consistency_global_friedman"]["pvalue"],
        "coverage_N3": s.loc[3, "coverage_rate"],
        "coverage_N5": s.loc[5, "coverage_rate"],
        "coverage_N10": s.loc[10, "coverage_rate"],
        "p_coverage_cochranQ": r["coverage_global_cochranQ"]["pvalue"],
        "accuracy_N3": s.loc[3, "accuracy_on_covered"],
        "accuracy_N5": s.loc[5, "accuracy_on_covered"],
        "accuracy_N10": s.loc[10, "accuracy_on_covered"],
        "p_accuracy_cochranQ_on_covered_intersection": r["accuracy_global_cochranQ_on_covered_intersection"]["pvalue"],
        "n_paired": r["consistency_global_friedman"]["n"],
        "n_acc_complete": r["accuracy_global_cochranQ_on_covered_intersection"]["n_complete"],
    })

out_df = pd.DataFrame(rows).sort_values("threshold")

cols_round3 = [
    "coverage_N3",
    "coverage_N5",
    "coverage_N10",
    "accuracy_N3",
    "accuracy_N5",
    "accuracy_N10"
]
for c in cols_round3:
    out_df[c] = pd.to_numeric(out_df[c], errors="coerce").round(3)

cols_round4 = [
    "p_consistency_friedman",
    "p_coverage_cochranQ",
    "p_accuracy_cochranQ_on_covered_intersection",
]

def fmt_p(x):
    if pd.isna(x):
        return ""
    x = float(x)
    if x < 1e-3:
        return f"{x:.3e}"   # e.g., 1.598e-08
    return f"{x:.3f}"       # Standard decimal notation

for c in cols_round4:
    out_df[c] = pd.to_numeric(out_df[c], errors="coerce").map(fmt_p)


out_path = "/mnt/bulk-sirius/lizhang/LiWS/Medical_Llama_Agents/data/plots/63/63.10/63.10_N/threshold_stats_summary.csv"
out_df.to_csv(out_path, index=False)
print(f"saved: {out_path}")
