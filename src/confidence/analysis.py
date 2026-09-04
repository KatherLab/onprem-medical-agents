import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import mannwhitneyu, norm
import plotly.express as px

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None

DATASET_NAME_MAP = {
    'appendicitis': 'Appendicitis',
    'cholecystitis': 'Cholecystitis',
    'diverticulitis': 'Diverticulitis',
    'pancreatitis': 'Pancreatitis',
    'pneumonia': 'Pneumonia',
    'pulmonary_embolism': 'Pulm. embolism',
    'uti': 'UTI',
    'Overall': 'Overall',
    'vivabench_pubmed': 'VivaBench-PubMed',
}


def _mm_to_inches(mm: float) -> float:
    """Convert millimetres to inches for journal-sized figure presets."""
    return mm / 25.4


# --- Global style setup ---
def setup_style():
    """Set global plotting parameters."""
    try:
        arial_path = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf" # verify path
        fm.fontManager.addfont(arial_path)
        plt.rcParams['font.family'] = 'Arial'
    except FileNotFoundError:
        print("WARNING: Arial font not found; using default sans-serif.")
        plt.rcParams['font.family'] = 'sans-serif'

    # 5-7 pt sans-serif text.
    plt.rcParams['font.size'] = 6
    plt.rcParams['axes.labelsize'] = 6
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['xtick.labelsize'] = 5
    plt.rcParams['ytick.labelsize'] = 5
    plt.rcParams['legend.fontsize'] = 5
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['lines.linewidth'] = 0.8 #1.0
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

# Apply the style settings.
setup_style()

class Analyzer:
    """
    Runs the final statistical analysis, modeling, and visualization tasks.
    """
    def __init__(self, config: dict):
        self.config = config
        self.plots_dir = config['results']['plots_base_dir']
        statistics_config = config.get('statistics', {})
        self.bootstrap_iterations = int(statistics_config.get('bootstrap_iterations', 2000))
        self.bootstrap_confidence_level = float(statistics_config.get('bootstrap_confidence_level', 0.95))
        self.bootstrap_random_seed = int(statistics_config.get('bootstrap_random_seed', 42))
        self.delong_pairs = statistics_config.get(
            'delong_pairs',
            [['Consistency_Dx', 'ProbScore_Dx']],
        )
        # sns.set_style("whitegrid")
        # plt.rcParams['font.size'] = 12
        self.palette = {
            'decision_true': '#2c77a0',
            'decision_false': '#F3857D',

            # --- Diagnosis metrics (suffix _Dx) ---
            'Consistency_Dx': '#5D9BB5',      # deep blue
            'ProbScore_Dx': '#9C89B8',        # deep purple
            'Bottom3Prob_Dx': '#C17C74',      # muted red-brown
            'TraceConsistency_Dx': '#4C956C', # green
            'SemanticCertainty_Dx': '#7B6FD0',  # indigo
            'ConceptDensity_Dx': '#3F9E44',         # deep orange
            'LingCert_Dx': '#D4A373',   # deep brown
            
            # --- Reasoning metrics (suffix _R) ---
            'Consistency_R': '#81BED8',     # blue
            'ProbScore_R': '#B6B3D6',       # purple
            'ConceptDensity_R':  '#B5D386',        # orange
            'LingCert_R': '#E3CDB2',  # light brown
        }
        self._auc_rows = []
        self._delong_rows = []
        self._nonredundant_rows = []
        self.export_dpi = 600
        self.figure_width_single = _mm_to_inches(88)
        self.figure_width_double = _mm_to_inches(180)
        print("Analyzer initialized.")

    def _get_figure_size(self, kind: str) -> tuple[float, float]:
        """Return journal-style figure sizes in inches."""
        presets = {
            'heatmap': (self.figure_width_single*0.7, 2.8*0.7),
            'vif': (self.figure_width_single, 2.4),
            'roc_double': (self.figure_width_double/2, 1.6),
            # 'roc_double': (self.figure_width_double*0.65, 2.4),
            'scatter_single': (self.figure_width_single*0.7, 3.0*0.7),
            'heatmap_single': (self.figure_width_single*0.6, 2.8*0.6),
            'violin_single': (self.figure_width_single, 2.8),
            'box_single': (self.figure_width_single * 0.35, 1.2),
            'kde_double': (self.figure_width_single, 2.9),
        }
        return presets.get(kind, (self.figure_width_single, 2.8))

    def _save_figure(self, fig, output_dir: str, stem: str, bbox_inches: str = 'tight'):
        """
        Save every figure in both SVG (for current workflow) and PDF (for journal-ready vector export).
        """
        svg_path = os.path.join(output_dir, f'{stem}.svg')
        pdf_path = os.path.join(output_dir, f'{stem}.pdf')
        fig.savefig(svg_path, format='svg', bbox_inches=bbox_inches)
        fig.savefig(pdf_path, format='pdf', bbox_inches=bbox_inches)
        print(f"    - Saved: {os.path.basename(svg_path)}")
        print(f"    - Saved: {os.path.basename(pdf_path)}")

    def _display_dataset_name(self, dataset_name: str) -> str:
        """Return a publication-friendly dataset label for figure text."""
        return DATASET_NAME_MAP.get(dataset_name, dataset_name)

    def run(self, df: pd.DataFrame):
        """
        Run the full analysis and visualization pipeline.
        """
        print("\n--- Running Analysis & Visualization Step ---")

        # =========================================================
        # 0) Determine which features to analyze.
        # =========================================================

        # 1) Read feature flags.
        feature_config = self.config.get('feature_engineering', {})
        enable_reasoning = feature_config.get('enable_reasoning_analysis', False)
        print(f"  - Configuration: Reasoning Analysis is {'ENABLED' if enable_reasoning else 'DISABLED'}")

        # 2) Define feature groups.
        # A) Diagnosis features (always included)
        target_features = [
            'ProbScore_Dx', 'Consistency_Dx', 'LingCert_Dx', 'ConceptDensity_Dx',
            # 'Bottom3Prob_Dx',
            # 'TraceConsistency_Dx',
            # 'SemanticCertainty_Dx',
        ]

        # B) Reasoning features (optional)
        if enable_reasoning:
            reasoning_features = [
                'ProbScore_R', 'Consistency_R', 'LingCert_R', 'ConceptDensity_R'
            ]
            target_features.extend(reasoning_features)

        # 3) Safe filtering: keep only columns that exist in the DataFrame.
        # This matches config intent and avoids crashes if a feature failed to compute.
        self.features_to_analyze = [f for f in target_features if f in df.columns]

        # Print final feature list.
        print(f"  - Final feature list for analysis ({len(self.features_to_analyze)}):")
        for f in self.features_to_analyze:
            print(f"    * {f}")

        if not self.features_to_analyze:
            print("  - WARNING: No valid features found in DataFrame to analyze!")
            return


        # =========================================================
        # 1) Global analysis (run once on the full dataset).
        # =========================================================
        overall_output_dir = os.path.join(self.plots_dir, "Overall")
        os.makedirs(overall_output_dir, exist_ok=True)
        
        report_path = os.path.join(overall_output_dir, 'A1_statistical_analysis_summary.txt')

        print(f"\n--- Performing Global Analysis (on Overall dataset) ---")
        print(f"    - Statistical summary will be saved to: {report_path}")
        
        # Use write mode to clear any old content.
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Statistical Analysis Summary (Overall Dataset)\n")
            f.write("="*50 + "\n")

        # Run analysis steps and append to the report.
        self._analyze_predictor_independence(df, overall_output_dir, report_path)
        self._analyze_statistical_significance(df, report_path)
        self._analyze_nonredundant_discrimination(df, overall_output_dir, report_path)


        # =========================================================
        # 2) Loop over datasets and generate descriptive plots.
        # =========================================================
        print("\n--- Generating Descriptive Plots for Each Dataset ---")
        datasets_cfg = self.config['datasets']
        benchmark = self.config.get('benchmark')

        # Backward-compatible:
        # - old config: datasets is a flat list
        # - new config: datasets is a dict keyed by benchmark
        if isinstance(datasets_cfg, dict):
            datasets_to_analyze = datasets_cfg.get(benchmark, [])
        else:
            datasets_to_analyze = datasets_cfg

        datasets_to_analyze = datasets_to_analyze + ["Overall"]

        
        for dataset_name in datasets_to_analyze:
            if dataset_name == "Overall":
                df_subset = df.copy()
            else:
                df_subset = df[df['dataset'] == dataset_name].copy()

            if df_subset.empty:
                print(f"  - No data for '{dataset_name}', skipping.")
                continue
            
            subset_output_dir = os.path.join(self.plots_dir, dataset_name)
            os.makedirs(subset_output_dir, exist_ok=True)
            
            print(f"  - Analyzing subset: {dataset_name} ({len(df_subset)} records)")
            self._generate_descriptive_plots(df_subset, dataset_name, subset_output_dir)

            # Run paired ROC comparisons on the same subset and save rows for final export.
            delong_df = self._run_delong_comparisons(df_subset, dataset_name)
            if delong_df is not None and not delong_df.empty:
                self._delong_rows.extend(delong_df.to_dict(orient='records'))

        self._write_auc_summary(overall_output_dir)
        self._write_delong_outputs(overall_output_dir, report_path)

        print("\nAnalysis and Visualization Complete!")

    def _compute_auc_with_bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        n_boot: int | None = None,
        confidence_level: float | None = None,
        seed: int | None = None,
    ) -> dict:
        """
        Compute ROC AUC and a bootstrap percentile confidence interval.

        The resampling is stratified by class so the bootstrap samples preserve
        positive/negative balance more reliably on smaller datasets.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        n_boot = n_boot or self.bootstrap_iterations
        confidence_level = confidence_level or self.bootstrap_confidence_level
        seed = self.bootstrap_random_seed if seed is None else seed

        if len(np.unique(y_true)) < 2:
            return {
                'auc': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n': int(len(y_true)),
                'n_pos': int(np.sum(y_true == 1)),
                'n_neg': int(np.sum(y_true == 0)),
                'valid_bootstrap_iterations': 0,
            }

        base_auc = roc_auc_score(y_true, y_score)
        rng = np.random.default_rng(seed)

        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        auc_samples = []

        for _ in range(n_boot):
            boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            boot_idx = np.concatenate([boot_pos, boot_neg])

            y_true_boot = y_true[boot_idx]
            y_score_boot = y_score[boot_idx]

            if len(np.unique(y_true_boot)) < 2:
                continue

            auc_samples.append(roc_auc_score(y_true_boot, y_score_boot))

        if auc_samples:
            alpha = 1.0 - confidence_level
            ci_lower = float(np.percentile(auc_samples, 100 * (alpha / 2)))
            ci_upper = float(np.percentile(auc_samples, 100 * (1 - alpha / 2)))
        else:
            ci_lower = np.nan
            ci_upper = np.nan

        return {
            'auc': float(base_auc),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': int(len(y_true)),
            'n_pos': int(len(pos_idx)),
            'n_neg': int(len(neg_idx)),
            'valid_bootstrap_iterations': int(len(auc_samples)),
        }

    def _compute_midrank(self, x: np.ndarray) -> np.ndarray:
        """
        Compute midranks for DeLong's U-statistic formulation.
        """
        order = np.argsort(x)
        sorted_x = x[order]
        midranks = np.zeros(len(x), dtype=float)

        i = 0
        while i < len(sorted_x):
            j = i
            while j < len(sorted_x) and sorted_x[j] == sorted_x[i]:
                j += 1
            midranks[i:j] = 0.5 * (i + j - 1) + 1
            i = j

        result = np.empty(len(x), dtype=float)
        result[order] = midranks
        return result

    def _fast_delong(self, predictions_sorted_transposed: np.ndarray, label_1_count: int):
        """
        Fast DeLong implementation for correlated ROC AUC estimates.

        Input shape: [n_classifiers, n_examples], with positive samples first.
        """
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m

        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]

        k = predictions_sorted_transposed.shape[0]
        tx = np.empty((k, m), dtype=float)
        ty = np.empty((k, n), dtype=float)
        tz = np.empty((k, m + n), dtype=float)

        for r in range(k):
            tx[r, :] = self._compute_midrank(positive_examples[r, :])
            ty[r, :] = self._compute_midrank(negative_examples[r, :])
            tz[r, :] = self._compute_midrank(predictions_sorted_transposed[r, :])

        aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

        sx = np.cov(v01)
        sy = np.cov(v10)
        delong_cov = sx / m + sy / n
        return aucs, delong_cov

    def _delong_auc_test(self, y_true: np.ndarray, y_score_1: np.ndarray, y_score_2: np.ndarray) -> dict:
        """
        Compare two correlated ROC AUC values using DeLong's test.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_score_1 = np.asarray(y_score_1, dtype=float)
        y_score_2 = np.asarray(y_score_2, dtype=float)

        if len(np.unique(y_true)) < 2:
            return {
                'auc_1': np.nan,
                'auc_2': np.nan,
                'auc_diff': np.nan,
                'z_stat': np.nan,
                'p_value': np.nan,
            }

        order = np.argsort(-y_true)
        label_1_count = int(np.sum(y_true == 1))
        predictions_sorted = np.vstack([y_score_1, y_score_2])[:, order]
        aucs, covariance = self._fast_delong(predictions_sorted, label_1_count)

        if np.ndim(covariance) == 0:
            variance = float(covariance)
        else:
            contrast = np.array([1.0, -1.0])
            variance = float(contrast @ covariance @ contrast.T)

        variance = max(variance, 0.0)
        auc_diff = float(aucs[0] - aucs[1])

        if variance <= 0:
            z_stat = np.nan
            p_value = np.nan
        else:
            z_stat = auc_diff / np.sqrt(variance)
            p_value = float(2 * norm.sf(abs(z_stat)))

        return {
            'auc_1': float(aucs[0]),
            'auc_2': float(aucs[1]),
            'auc_diff': auc_diff,
            'z_stat': float(z_stat) if not np.isnan(z_stat) else np.nan,
            'p_value': p_value,
        }

    def _run_delong_comparisons(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame | None:
        """
        Run configured DeLong comparisons on one dataset subset.
        """
        rows = []

        for pair in self.delong_pairs:
            if len(pair) != 2:
                continue

            feature_1, feature_2 = pair
            if feature_1 not in df.columns or feature_2 not in df.columns:
                continue

            # Use the shared non-null sample set so the ROC comparison is paired.
            valid_df = df[[feature_1, feature_2, 'final_decision']].dropna().copy()
            if valid_df.empty or valid_df['final_decision'].nunique() < 2:
                continue

            y_true = valid_df['final_decision'].astype(int).to_numpy()
            y_score_1 = valid_df[feature_1].astype(float).to_numpy()
            y_score_2 = valid_df[feature_2].astype(float).to_numpy()

            result = self._delong_auc_test(y_true, y_score_1, y_score_2)
            result.update({
                'dataset': dataset_name,
                'feature_1': feature_1,
                'feature_2': feature_2,
                'n': int(len(valid_df)),
                'n_pos': int(np.sum(y_true == 1)),
                'n_neg': int(np.sum(y_true == 0)),
            })
            rows.append(result)

        if not rows:
            return None

        results_df = pd.DataFrame(rows)
        if len(results_df) > 1 and multipletests is not None:
            _, p_adj, _, _ = multipletests(results_df['p_value'].values, alpha=0.05, method='fdr_bh')
            results_df['p_adj_fdr'] = p_adj
        else:
            results_df['p_adj_fdr'] = results_df['p_value']

        return results_df

    def _write_auc_summary(self, output_dir: str):
        """
        Persist the per-dataset ROC/AUC summary table used by the figures.
        """
        if not self._auc_rows:
            return

        auc_df = pd.DataFrame(self._auc_rows)
        auc_df = auc_df.sort_values(by=['dataset', 'stream', 'feature']).reset_index(drop=True)
        auc_path = os.path.join(output_dir, 'auc_summary.csv')
        auc_df.to_csv(auc_path, index=False)
        print(f"  - Saved: {auc_path}")

    def _write_delong_outputs(self, output_dir: str, report_path: str):
        """
        Save DeLong results both as a CSV and as a text appendix in the report.
        """
        if not self._delong_rows:
            return

        delong_df = pd.DataFrame(self._delong_rows)
        delong_df = delong_df.sort_values(by=['dataset', 'feature_1', 'feature_2']).reset_index(drop=True)
        delong_path = os.path.join(output_dir, 'delong_results.csv')
        delong_df.to_csv(delong_path, index=False)
        print(f"  - Saved: {delong_path}")

        with open(report_path, 'a', encoding='utf-8') as f:
            f.write("\n--- Paired ROC Comparison (DeLong Test) ---\n")
            for _, row in delong_df.iterrows():
                line = (
                    f"  - {row['dataset']}: {row['feature_1']} vs {row['feature_2']} | "
                    f"AUC1={row['auc_1']:.3f}, AUC2={row['auc_2']:.3f}, "
                    f"diff={row['auc_diff']:.3f}, z={row['z_stat']:.3f}, "
                    f"p={row['p_value']:.4g}, p_fdr={row['p_adj_fdr']:.4g}"
                )
                f.write(line + "\n")

    def _fit_descriptive_logistic_scores(
        self,
        df: pd.DataFrame,
        predictors: list[str],
    ) -> tuple[np.ndarray, np.ndarray, int] | tuple[None, None, int]:
        """
        Fit a descriptive logistic model on the full cohort and return in-sample scores.

        This is intentionally a full-cohort descriptive analysis rather than an
        out-of-sample predictive validation procedure.
        """
        valid_df = df[predictors + ['final_decision']].dropna().copy()
        if valid_df.empty or valid_df['final_decision'].nunique() < 2:
            return None, None, 0

        X = valid_df[predictors].astype(float).to_numpy()
        y = valid_df['final_decision'].astype(int).to_numpy()

        model = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=self.bootstrap_random_seed,
        )
        model.fit(X, y)
        y_score = model.predict_proba(X)[:, 1]
        return y, y_score, int(len(valid_df))

    def _analyze_nonredundant_discrimination(self, df: pd.DataFrame, output_dir: str, report_path: str):
        """
        Assess whether ProbScore_Dx and Consistency_Dx retain non-redundant
        discriminative information when modeled jointly on the full cohort.
        """
        required_cols = ['ProbScore_Dx', 'Consistency_Dx', 'final_decision']
        if not all(col in df.columns for col in required_cols):
            return

        model_specs = [
            ('probscore_only', ['ProbScore_Dx']),
            ('consistency_only', ['Consistency_Dx']),
            ('joint_probscore_consistency', ['ProbScore_Dx', 'Consistency_Dx']),
        ]

        scores_by_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for model_name, predictors in model_specs:
            y_true, y_score, n = self._fit_descriptive_logistic_scores(df, predictors)
            if y_true is None or y_score is None:
                continue

            auc_stats = self._compute_auc_with_bootstrap_ci(
                y_true=y_true,
                y_score=y_score,
                seed=self.bootstrap_random_seed,
            )
            row = {
                'analysis_scope': 'overall_full_cohort',
                'outcome': 'final_decision',
                'model_name': model_name,
                'predictors': ' + '.join(predictors),
                'n': n,
                'n_pos': auc_stats['n_pos'],
                'n_neg': auc_stats['n_neg'],
                'auc': auc_stats['auc'],
                'ci_lower': auc_stats['ci_lower'],
                'ci_upper': auc_stats['ci_upper'],
                'bootstrap_iterations': self.bootstrap_iterations,
                'valid_bootstrap_iterations': auc_stats['valid_bootstrap_iterations'],
            }
            self._nonredundant_rows.append(row)
            scores_by_model[model_name] = (y_true, y_score)

        if not self._nonredundant_rows:
            return

        summary_df = pd.DataFrame(self._nonredundant_rows)
        summary_path = os.path.join(output_dir, 'nonredundant_info_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"  - Saved: {summary_path}")

        comparisons = [
            ('joint_probscore_consistency', 'probscore_only'),
            ('joint_probscore_consistency', 'consistency_only'),
        ]

        delong_rows = []
        for better_model, baseline_model in comparisons:
            if better_model not in scores_by_model or baseline_model not in scores_by_model:
                continue
            y_true_joint, y_score_joint = scores_by_model[better_model]
            y_true_base, y_score_base = scores_by_model[baseline_model]
            if len(y_true_joint) != len(y_true_base) or not np.array_equal(y_true_joint, y_true_base):
                continue

            result = self._delong_auc_test(y_true_joint, y_score_joint, y_score_base)
            result.update({
                'dataset': 'Overall',
                'feature_1': better_model,
                'feature_2': baseline_model,
                'n': int(len(y_true_joint)),
                'n_pos': int(np.sum(y_true_joint == 1)),
                'n_neg': int(np.sum(y_true_joint == 0)),
            })
            delong_rows.append(result)

        with open(report_path, 'a', encoding='utf-8') as f:
            f.write("\n--- Non-Redundant Discriminative Information Analysis ---\n")
            for _, row in summary_df.iterrows():
                f.write(
                    f"  - {row['model_name']} ({row['predictors']}): "
                    f"AUC={row['auc']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
                    f"n={int(row['n'])}, n_pos={int(row['n_pos'])}, n_neg={int(row['n_neg'])}\n"
                )

            if delong_rows:
                f.write("  - Joint vs single-metric paired DeLong comparisons:\n")
                for row in delong_rows:
                    f.write(
                        f"    * {row['feature_1']} vs {row['feature_2']}: "
                        f"diff={row['auc_diff']:.3f}, z={row['z_stat']:.3f}, p={row['p_value']:.4g}\n"
                    )

    def _analyze_predictor_independence(self, df: pd.DataFrame, output_dir: str, report_path: str):
        """Analyze predictor independence and save results."""
        print("  - Analyzing predictor independence...")

        # --- a) Correlation analysis (Pearson & Spearman) ---
        print("  - Calculating correlation matrices...")
        # Only analyze features that exist.
        corr_features = [
            feature
            for feature in [
                'Consistency_Dx',
                'ProbScore_Dx',
                # 'Bottom3Prob_Dx',
                # 'TraceConsistency_Dx',
                # 'SemanticCertainty_Dx',
                'Consistency_R',
                'ProbScore_R',
                'LingCert_R',
            ]
            if feature in df.columns
        ]
        if not corr_features:
            return
        # 1) Pearson correlation (linear relationship)
        pearson_corr_matrix = df[corr_features].corr(method='pearson')
        # 2) Spearman correlation (monotonic relationship)
        spearman_corr_matrix = df[corr_features].corr(method='spearman')
        
        # b) VIF
        X_vif = add_constant(df[corr_features].dropna()) # ensure no NaNs
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        vif_results = vif_data[vif_data["feature"] != "const"].copy()

        # Append results to the report.
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write("\n--- Predictor Independence Analysis ---\n\n")
            f.write("1. Pearson Correlation Matrix (Linear Relationship):\n")
            f.write(pearson_corr_matrix.to_string())
            f.write("\n\n2. Spearman Rank Correlation Matrix (Monotonic Relationship):\n")
            f.write(spearman_corr_matrix.to_string())
            f.write("\n\n3. Variance Inflation Factor (VIF):\n") 
            f.write(vif_results.to_string(index=False))
            f.write("\n" + "="*50 + "\n")
            f.write("\n" + "="*50 + "\n")

        # --- e) Visualization ---
        # 1) Pearson correlation heatmap (main)
        fig, ax = plt.subplots(figsize=self._get_figure_size('heatmap'), dpi=self.export_dpi)
        sns.heatmap(
            pearson_corr_matrix,
            annot=True,
            fmt=".2f", # two decimals
            annot_kws={"fontsize": 5},
            cmap='RdBu', #'Blues',
            vmin=-1, vmax=1,
            linewidths=1, # thicker grid lines
            linecolor='white',
            ax=ax
        )
        
        # Remove spines but keep ticks.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # ax.set_title('Pearson Correlation of Confidence Predictors', loc='left', weight='bold', pad=20)
        plt.xticks(rotation=30, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        self._save_figure(fig, output_dir, 'P2_pearson_correlation_matrix')
        plt.close()

        # 2) Spearman correlation heatmap (optional)
        fig, ax = plt.subplots(figsize=self._get_figure_size('heatmap'), dpi=self.export_dpi)
        sns.heatmap(
            spearman_corr_matrix, annot=True, fmt=".2f", cmap='Blues', annot_kws={"fontsize": 5},
            vmin=-1, vmax=1, linewidths=.5, linecolor='white', ax=ax
        )
        # ax.set_title('Spearman Rank Correlation of Confidence Predictors', loc='left', weight='bold')
        plt.xticks(rotation=30, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        self._save_figure(fig, output_dir, 'P2_spearman_correlation_matrix', bbox_inches='tight')
        plt.close()
        

        # --- 3) VIF horizontal bar chart ---
        # Rename features for prettier labels.
        vif_results['Feature Label'] = vif_results['feature'].str.replace('_Score', '').str.capitalize()
        vif_results.sort_values(by='VIF', ascending=True, inplace=True)

        # a) Prepare a color list aligned with vif_results rows.
        bar_colors = vif_results['feature'].map(self.palette).tolist()
        
        # b) Create figure.
        fig_vif, ax_vif = plt.subplots(figsize=self._get_figure_size('vif'), dpi=self.export_dpi)

        # c) Draw horizontal bar chart with per-feature colors.
        ax_vif.barh(
            vif_results['Feature Label'], 
            vif_results['VIF'], 
            color=bar_colors # use prepared colors
        )

        # Remove top/right spines.
        ax_vif.spines['top'].set_visible(False)
        ax_vif.spines['right'].set_visible(False)

        # Add VIF=5 threshold line.
        threshold = 5
        # ax_vif.axvline(x=threshold, color='red', linestyle='--', linewidth=1, label=f'High Collinearity Threshold (VIF = {threshold})')
        ax_vif.axvline(x=threshold, color='red', linestyle='--', linewidth=0.6)

        # Styling.
        # ax_vif.set_title('Variance Inflation Factor (VIF)', loc='left', weight='bold')
        ax_vif.set_xlabel('VIF Value')
        ax_vif.set_xlim(0, max(6, vif_results['VIF'].max() * 1.1))

        # Add value labels at the end of each bar.
        for index, row in vif_results.iterrows():
            ax_vif.text(row['VIF'] + 0.1, row['Feature Label'], f'{row["VIF"]:.2f}', va='center')

        ax_vif.legend(frameon=False, loc='lower right')
        ax_vif.tick_params(axis='y', length=0)

        plt.tight_layout()
        
        # Save chart.
        self._save_figure(fig_vif, output_dir, 'P2_vif_analysis_barchart')
        plt.close()


    def _generate_descriptive_plots(self, df: pd.DataFrame, dataset_name: str, output_dir: str):
        """Generate all descriptive plots for a given dataset subset."""
        display_dataset_name = self._display_dataset_name(dataset_name)
        df['final_decision_str'] = df['final_decision'].astype(str)
        df['final_decision_int'] = df['final_decision'].astype(int) # for accuracy calculations
        y_true = df['final_decision']
        
        # --- a) Dual-stream ROC curves (Diagnosis vs. Reasoning) ---
        # Create a 1x2 canvas.
        
        current_figsize = self._get_figure_size('roc_double')
        title_fontsize = 7 #if dataset_name == "Overall" else 6
        fig, axes = plt.subplots(1, 2, figsize=current_figsize, dpi=self.export_dpi, sharey=True)
        
        # Two subplot configurations.
        plot_configs = [
            {'ax': axes[0], 'suffix': '_Dx', 'panel': 'Diagnosis'},
            {'ax': axes[1], 'suffix': '_R',  'panel': 'Reasoning'}
        ]

        for config in plot_configs:
            ax = config['ax']
            suffix = config['suffix']
            stream_name = 'Diagnosis' if suffix == '_Dx' else 'Reasoning'
            
            # 1) Select features for this subplot.
            current_features = [f for f in self.features_to_analyze if f.endswith(suffix)]
            
            # 2) Basic styling.
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Draw diagonal.
            ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
            
            if not current_features:
                # If no features (e.g., reasoning disabled), skip.
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                continue

            # 3) Plot curves.
            for feature in current_features:
                # Drop NaNs to avoid errors.
                valid_data = df[[feature, 'final_decision']].dropna()
                if valid_data.empty: continue

                y_true = valid_data['final_decision'].astype(int).to_numpy()
                y_score = valid_data[feature].astype(float).to_numpy()
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_stats = self._compute_auc_with_bootstrap_ci(
                    y_true=y_true,
                    y_score=y_score,
                    seed=self.bootstrap_random_seed,
                )
                
                # Simplify legend labels by removing suffix for side-by-side comparison.
                # e.g., "ProbScore_Dx" -> "ProbScore"
                clean_label = feature.replace(suffix, '')
                ci_label = (
                    f"{clean_label} ({auc_stats['auc']:.3f})"
                    # f"[{auc_stats['ci_lower']:.3f}, {auc_stats['ci_upper']:.3f}])"
                )
                
                ax.plot(fpr, tpr, lw=0.8, #1.0, 
                        label=ci_label,
                        color=self.palette.get(feature, 'black') # use configured colors
                        )

                self._auc_rows.append({
                    'dataset': dataset_name,
                    'stream': stream_name,
                    'feature': feature,
                    'auc': auc_stats['auc'],
                    'ci_lower': auc_stats['ci_lower'],
                    'ci_upper': auc_stats['ci_upper'],
                    'n': auc_stats['n'],
                    'n_pos': auc_stats['n_pos'],
                    'n_neg': auc_stats['n_neg'],
                    'bootstrap_iterations': self.bootstrap_iterations,
                    'valid_bootstrap_iterations': auc_stats['valid_bootstrap_iterations'],
                })
            
            # 4) Axes settings.
            # ax.set_xlim([-0.02, 1.02])
            ax.set_xlim([0.0, 1.02])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            # ax.set_title(config['title'], loc='left', fontsize=title_fontsize, fontweight='bold')
            ax.text(0.0, 1.03, config['panel'], transform=ax.transAxes, fontsize=6, fontweight='bold')
            ax.legend(loc="lower right", frameon=False, fontsize=5)

        # Only show Y-axis label on the left subplot.
        axes[0].set_ylabel('True Positive Rate')

        fig.text(
            0.52, 1.00, display_dataset_name,
            ha='center',
            va='bottom',
            fontsize=7,
            fontweight='bold'
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        plt.tight_layout()
        # Keep filename consistent.
        self._save_figure(fig, output_dir, 'P1_roc_curve_comparison')
        plt.close()

        # --- b) Paired scatter plots for core predictors ---
        # Pick two pairs with the clearest story.
        mechanism_pairs = []

        # Pair A: Internal vs behavioral (Diagnosis) -> "Discordance Risk"
        if 'ProbScore_Dx' in df.columns and 'Consistency_Dx' in df.columns:
            mechanism_pairs.append({
                'x': 'ProbScore_Dx', 
                'y': 'Consistency_Dx', 
                'title': 'Discordance Zone: High Confidence, Low Stability',
                'filename': 'P2_Scatter_Internal_vs_Behavioral_Dx'
            })
        # if 'Bottom3Prob_Dx' in df.columns and 'Consistency_Dx' in df.columns:
        #     mechanism_pairs.append({
        #         'x': 'Bottom3Prob_Dx', 
        #         'y': 'Consistency_Dx', 
        #         'title': 'Discordance Zone: High Confidence, Low Stability',
        #         'filename': 'P2_Scatter_Bottom3Prob_Dx_vs_Behavioral_Dx'
        #     })

        # Pair B: Internal vs concept density (Reasoning) -> "Jargon Trap"
        if 'LingCert_R' in df.columns and 'Consistency_R' in df.columns:
            mechanism_pairs.append({
                'x': 'LingCert_R', 
                'y': 'Consistency_R', 
                'title': 'LingCert_R vs. Consistency_R',
                'filename': 'P2_Scatter_LingCert_R_vs_Consistency_R'
            })
        # Split data by decision.
        # df['final_decision_str'] = df['final_decision'].astype(str)
        df_true = df[df['final_decision_str'] == 'True']
        df_false = df[df['final_decision_str'] == 'False']

        # for x_var, y_var in core_pairs:
        for pair in mechanism_pairs:
            x_var = pair['x']
            y_var = pair['y']

            fig, ax = plt.subplots(figsize=self._get_figure_size('scatter_single'), dpi=self.export_dpi)

            # 1) Plot majority points (True) with higher transparency.
            ax.scatter(
                df_true[x_var],
                df_true[y_var],
                color='#CFCFD0', #self.palette['decision_true'],
                alpha=0.7,# 0.4,  # higher transparency
                s=25,       # moderate size
                # edgecolor='none', # remove outlines for softer background
                edgecolor='white', # add white outline for contrast
                linewidth=0.2,
                label='Correct'
            )

            # 2) Plot minority points (False) with lower transparency and outlines.
            ax.scatter(
                df_false[x_var],
                df_false[y_var],
                color='#CFCFD0', #self.palette['decision_false'],
                marker='X',
                alpha=1.0,  # more visible
                s=35,       # slightly larger
                edgecolor='white', # add white outline
                linewidth=0.2,
                label='Incorrect'
            )
            # ax.set_title('Concordance of Internal vs. Expressed Confidence', loc='left', weight='bold')
            # Axis labels: replace underscores for paper-style labels.
            # ax.set_xlabel(x_var.replace("_", " "))
            # ax.set_ylabel(y_var.replace("_", " "))
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)

            legend = ax.legend(title='Decision', frameon=False)
            plt.tight_layout()
            stem = f'P2_{x_var}_vs_{y_var}'
            self._save_figure(fig, output_dir, stem, bbox_inches='tight')
            plt.close()

            # --- Interactive version of the same plot ---
            fig_interactive = px.scatter(
                df, 
                x=x_var,
                y=y_var,
                color='final_decision_str',
                color_discrete_map={'True': self.palette['decision_true'], 'False': self.palette['decision_false']},
                hover_data=df.columns, # show all info on hover
                title='Interactive: Concordance of Internal vs. Expressed Confidence'
            )
            fig_interactive.update_traces(marker=dict(size=5, opacity=0.7))
            html_name = f'P2_{x_var}_vs_{y_var}_interactive.html'
            fig_interactive.write_html(os.path.join(output_dir, html_name))
            print(f"    - Saved: {html_name}")

        # ==============================================================================
        # 3. Grid-Based Accuracy Heatmap (Grid Analysis)
        # ==============================================================================
        # Only for Diagnosis internal vs behavioral (basis for Selective Autonomy).
        if len(df) > 30 and 'ProbScore_Dx' in df.columns and 'Consistency_Dx' in df.columns:
            try:
                # 1) Dynamic binning.
                # Use qcut for roughly equal counts per bin; fallback to cut if skewed.
                try:
                    df['Prob_Bin'] = pd.qcut(df['ProbScore_Dx'], q=3, labels=['Low', 'Med', 'High'], duplicates='drop')
                    df['Cons_Bin'] = pd.qcut(df['Consistency_Dx'], q=3, labels=['Low', 'Med', 'High'], duplicates='drop')
                except ValueError:
                    # Fallback: if data is too concentrated, use equal-width bins.
                    df['Prob_Bin'] = pd.cut(df['ProbScore_Dx'], bins=3, labels=['Low', 'Med', 'High'])
                    df['Cons_Bin'] = pd.cut(df['Consistency_Dx'], bins=3, labels=['Low', 'Med', 'High'])

                # 2) Mean accuracy per grid cell.
                pivot_table = df.pivot_table(
                    values='final_decision_int', 
                    index='Cons_Bin', 
                    columns='Prob_Bin', 
                    aggfunc='mean',
                    observed=False # avoid categorical FutureWarning
                )
                
                # Ensure consistent ordering (High in top-right).
                pivot_table = pivot_table.reindex(index=['High', 'Med', 'Low'], columns=['Low', 'Med', 'High'])

                # 3) Plot.
                fig, ax = plt.subplots(figsize=self._get_figure_size('heatmap_single'), dpi=self.export_dpi)
                
                sns.heatmap(
                    pivot_table, 
                    annot=True,      # show values
                    fmt=".1%",       # percentage format (e.g., 95.2%)
                    annot_kws={"fontsize": 5},
                    cmap="RdBu",   # red (low) -> yellow -> blue (high)
                    vmin=0.0, vmax=1.0,
                    linewidths=1, 
                    linecolor='white',
                    cbar_kws={'label': 'Diagnostic Accuracy'},
                    ax=ax
                )

                # ax.set_title('Interaction: Accuracy by Confidence Bins', loc='left', fontweight='bold')
                ax.set_xlabel('Internal Probability (Bin)')
                ax.set_ylabel('Behavioral Consistency (Bin)')
                
                plt.tight_layout()
                self._save_figure(fig, output_dir, 'P3_Interaction_Accuracy_Heatmap', bbox_inches='tight')
                plt.close()

            except Exception as e:
                print(f"  - WARNING: Could not generate Heatmap. Reason: {e}")

        # Only for LingCert_R vs behavioral (basis for Selective Autonomy).
        if len(df) > 30 and 'LingCert_R' in df.columns and 'Consistency_Dx' in df.columns:
            try:
                # 1) Dynamic binning.
                # Use qcut for roughly equal counts per bin; fallback to cut if skewed.
                try:
                    df['Ling_Bin'] = pd.qcut(df['LingCert_R'], q=3, labels=['Low', 'Med', 'High'], duplicates='drop')
                    df['Cons_Bin'] = pd.qcut(df['Consistency_Dx'], q=3, labels=['Low', 'Med', 'High'], duplicates='drop')
                except ValueError:
                    # Fallback: if data is too concentrated, use equal-width bins.
                    df['Ling_Bin'] = pd.cut(df['LingCert_R'], bins=3, labels=['Low', 'Med', 'High'])
                    df['Cons_Bin'] = pd.cut(df['Consistency_Dx'], bins=3, labels=['Low', 'Med', 'High'])

                # 2) Mean accuracy per grid cell.
                pivot_table = df.pivot_table(
                    values='final_decision_int', 
                    index='Cons_Bin', 
                    columns='Ling_Bin', 
                    aggfunc='mean',
                    observed=False # avoid categorical FutureWarning
                )
                
                # Ensure consistent ordering (High in top-right).
                pivot_table = pivot_table.reindex(index=['High', 'Med', 'Low'], columns=['Low', 'Med', 'High'])

                # 3) Plot.
                fig, ax = plt.subplots(figsize=self._get_figure_size('heatmap_single'), dpi=self.export_dpi)
                
                sns.heatmap(
                    pivot_table, 
                    annot=True,      # show values
                    fmt=".1%",       # percentage format (e.g., 95.2%)
                    annot_kws={"fontsize": 5},
                    cmap="RdBu",   # red (low) -> yellow -> blue (high)
                    vmin=0.0, vmax=1.0,
                    linewidths=1, 
                    linecolor='white',
                    cbar_kws={'label': 'Diagnostic Accuracy'},
                    ax=ax
                )

                # ax.set_title('Interaction: Accuracy by Confidence Bins', loc='left', fontweight='bold')
                ax.set_xlabel('Reason Linguistic Certainty (Bin)')
                ax.set_ylabel('Diagnosis Behavioral Consistency (Bin)')
                
                plt.tight_layout()
                self._save_figure(fig, output_dir, 'P3_LingCert_R_Consistency_Dx_Interaction', bbox_inches='tight')
                plt.close()

            except Exception as e:
                print(f"  - WARNING: Could not generate Heatmap. Reason: {e}")
            
        for feature in self.features_to_analyze:
            # e) Violin plot distribution.
            fig, ax = plt.subplots(figsize=self._get_figure_size('violin_single'), dpi=self.export_dpi)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)                
            
            sns.violinplot(
                data=df, x='final_decision_str', y=feature, order=['True', 'False'],
                palette={'True': self.palette[feature], 'False': self.palette['decision_false']},
                # inner=None, # remove inner boxplot
                linewidth=0.5, # remove outlines
                scale='area',
                ax=ax,
            )
            
            ax.set_title(f'{display_dataset_name}: {feature}', loc='left', weight='bold', fontsize=7)
            ax.set_xlabel('Final Decision')
            ax.set_ylabel(feature.replace("_", " "))
            plt.tight_layout()
            self._save_figure(fig, output_dir, f'P1_{feature}_violin_plot')
            plt.close()

            # f) Box plot (manual positioning for two boxes)
            fig, ax = plt.subplots(figsize=self._get_figure_size('box_single'), dpi=self.export_dpi)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            data_true = df[df['final_decision_str'] == 'True'][feature].dropna()
            data_false = df[df['final_decision_str'] == 'False'][feature].dropna()

            # Desired gap and width.
            width = 0.5 # control box width
            gap = 0.2                     # data-space gap between boxes
            center_dist = width + gap     # 0.7
            positions = [0.5 - center_dist/2, 0.5 + center_dist/2]   # [0.15, 0.85]

            bp = ax.boxplot(
                [data_true, data_false],
                positions=positions,
                widths=width,           # box width
                patch_artist=True,    # allow filled colors
                showcaps=True,
                showfliers=False,
                boxprops=dict(edgecolor='none'),
                whiskerprops=dict(color='black', linewidth=0.5),
                capprops=dict(color='black', linewidth=0.5),
                medianprops=dict(color='black', linewidth=0.6),
            )

            # Apply colors to the two boxes.
            colors = [self.palette[feature], self.palette['decision_false']]
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)

            # Place x ticks at box centers.
            ax.set_xticks(positions)
            ax.set_xticklabels(['True', 'False'])

            # ax.set_xlabel('Final Decision', fontsize=6)
            # ax.set_ylabel(feature, fontsize=6)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='both', labelsize=5)

            plt.tight_layout()
            self._save_figure(fig, output_dir, f'P1_{feature}_box_plot', bbox_inches='tight')
            plt.close()


            # g) Distribution plot (KDE)
            fig, ax = plt.subplots(figsize=self._get_figure_size('kde_double'), dpi=self.export_dpi)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Use kdeplot with hue to split and color groups.
            sns.kdeplot(
                data=df, 
                x=feature, 
                hue='final_decision_str',
                hue_order=['True', 'False'], # ensure order/color match
                
                # --- styling ---
                fill=True,       # fill under curve
                alpha=0.6,       # transparency
                linewidth=1.5,
                
                # --- key params ---
                common_norm=False, # each curve normalized separately
                
                palette={'True': self.palette[feature], 'False': self.palette['decision_false']},
                ax=ax
            )
            # --------------------
            
            ax.set_title(f'{display_dataset_name}: {feature}', loc='left', weight='bold', fontsize=7)
            ax.set_xlabel(feature.replace("_", " "))
            ax.set_ylabel('Density')
            
            # Style the legend.
            legend = ax.get_legend()
            if legend:
                legend.set_title('Decision')
                legend.get_frame().set_linewidth(0.0) # remove legend frame
            
            plt.tight_layout()
            self._save_figure(fig, output_dir, f'P1_{feature}_kde_plot', bbox_inches='tight')
            plt.close()
            
        print("  - Saved: Violin, box plots and KDE distribution plots for all features.")



    def _analyze_statistical_significance(self, df: pd.DataFrame, report_path: str):
        """
        Compute per-feature p-values via Mann-Whitney U test and apply Holm correction.
        """
        print("  - Performing statistical significance tests (Mann-Whitney U + Holm correction)...")

        results = []  # (feature, stat, raw_p, n_true, n_false)

        # 1) Collect raw p-values.
        for feature in self.features_to_analyze:
            group_true = df[df['final_decision'] == True][feature].dropna()
            group_false = df[df['final_decision'] == False][feature].dropna()

            if len(group_true) < 1 or len(group_false) < 1:
                continue

            stat, p_raw = mannwhitneyu(group_true, group_false, alternative='two-sided')
            results.append((feature, float(stat), float(p_raw), int(len(group_true)), int(len(group_false))))

        if not results:
            print("    - WARNING: No valid features for significance testing.")
            return

        features = [r[0] for r in results]
        p_raws = np.array([r[2] for r in results], dtype=float)

        # 2) Holm correction.
        if multipletests is None:
            # Fallback if statsmodels is missing: use raw p-values.
            p_adj = p_raws
            print("    - WARNING: statsmodels not found; Holm correction skipped.")
        else:
            # multipletests returns: reject, pvals_corrected, ...
            _, p_adj, _, _ = multipletests(p_raws, alpha=0.05, method="holm")

        # 3) Write report (raw + Holm).
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write("\n--- Statistical Significance (Mann-Whitney U Test, two-sided) ---\n")
            f.write("P values are reported as raw and Holm-adjusted (family = all metrics in Fig. 3c).\n\n")

            for (feature, stat, p_raw, n_t, n_f), p_holm in zip(results, p_adj):
                line = (
                    f"  - {feature}: U={stat:.2f}, "
                    f"n_true={n_t}, n_false={n_f}, "
                    f"p_raw={p_raw:.4g}, p_holm={float(p_holm):.4g}"
                )
                print(line)
                f.write(line + "\n")

                # Determine significance based on Holm-adjusted p-values.
                if p_holm < 1e-4:
                    f.write("    - Significant after Holm correction (p_holm < 0.0001).\n")
                elif p_holm < 0.05:
                    f.write("    - Significant after Holm correction (p_holm < 0.05).\n")
                else:
                    f.write("    - Not significant after Holm correction.\n")
