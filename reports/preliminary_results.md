# Preliminary Results — Credit Scoring ML Project

**Date**: 2026-03-22
**Reference**: arXiv:2509.11389 (Schwartz, Wang & Fang, 2025)
**Status**: All phases (1-5) complete. Phase 6 (report/presentation) pending.

---

## Dataset Summary

- **Source**: Lending Club 2007–2020 Q3 (Kaggle: ethon0426/lending-club-20072020q1)
- **Raw**: 2,925,493 rows, 142 columns
- **After filtering** (Fully Paid + Charged Off only): 1,860,331 rows
- **Default rate**: 19.5% (paper reports ~20% — consistent)
- **Temporal split**: Train ≤ Jul-2015 (684,592 rows), Test Aug-2015 to Dec-2018 (1,096,128 rows)
- **Train default rate**: 17.6%, **Test default rate**: 20.9%
- **Features after preprocessing**: 75 (paper reports 87)

### Feature count discrepancy (75 vs 87)

The paper reports 87 features. We get 75 after:
- Dropping leakage columns (post-loan info like `total_pymnt`, `recoveries`, etc.)
- Dropping joint-application columns (mostly null)
- One-hot encoding 9 categorical columns
- Converting `earliest_cr_line` to `cr_hist_months`
- Dropping remaining non-numeric columns

The difference likely comes from the paper including different columns or using a different encoding strategy (e.g., more granular one-hot categories). This is a known divergence that affects LR performance in particular (see below).

---

## Phase 2 Results — Baseline Models (All 75 Features)

### Default threshold (0.5)

| Model | AUROC | AUPRC | F1 | Balanced Accuracy |
|-------|-------|-------|------|-------------------|
| LR | 0.5554 | 0.2317 | 0.3053 | 0.5392 |
| XGBoost | 0.7141 | 0.3918 | 0.4423 | 0.6534 |
| LightGBM | 0.7204 | 0.4007 | 0.4476 | 0.6586 |
| EBM | 0.7206 | 0.3987 | **0.1234** | 0.5278 |
| NeuralNet (v2) | **0.7215** | 0.3981 | **0.4488** | **0.6599** |

### Optimal thresholds (Youden's J)

| Model | Threshold | AUROC | AUPRC | F1 | Balanced Accuracy |
|-------|-----------|-------|-------|------|-------------------|
| LR | 0.489 | 0.5554 | 0.2317 | 0.3080 | 0.5393 |
| XGBoost | 0.471 | 0.7141 | 0.3918 | 0.4426 | 0.6555 |
| LightGBM | 0.472 | 0.7204 | 0.4007 | 0.4471 | 0.6604 |
| EBM | **0.174** | 0.7206 | 0.3987 | 0.4474 | 0.6602 |
| NeuralNet (v2) | 0.483 | **0.7215** | 0.3981 | 0.4483 | **0.6607** |

### Paper's reported baselines (87 features)

| Model | AUROC | AUPRC | F1 | Balanced Accuracy |
|-------|-------|-------|------|-------------------|
| LR (paper) | 0.6653 | 0.3389 | 0.4200 | 0.6187 |
| XGB (paper) | 0.6687 | 0.3436 | 0.4160 | 0.6203 |
| EBM (paper) | 0.6744 | 0.3518 | 0.4211 | 0.6251 |

---

## Issues Found and Resolutions

### Issue 1: EBM F1 = 0.123 (CRITICAL — RESOLVED)

**Symptom**: EBM has excellent AUROC (0.721) but terrible F1 (0.123) and only predicts 2.5% positive rate vs actual 20.9%.

**Root cause**: EBM's `predict_proba` outputs center around 0.18 (not 0.5) because InterpretML's EBM doesn't have a `class_weight` parameter. The default threshold of 0.5 is far above the probability mass, causing the model to predict almost everything as negative.

**Resolution**: Use Youden's J optimal threshold of 0.174. With this threshold, EBM achieves F1=0.447 (in line with XGB/LGBM). For Phase 3+, all models should use optimized thresholds.

**Saved**: `reports/optimal_thresholds.csv`

### Issue 2: Neural Net AUROC = 0.509 (CRITICAL — RESOLVED)

**Symptom**: Original NN barely better than random (AUROC=0.509).

**Root cause (multi-factor)**:
1. **Training on MPS device** — PyTorch 2.2.2 had NumPy 2.x incompatibility, causing silent numerical issues on Apple Silicon
2. **Only 10 epochs** — loss was ~1.03 and barely decreasing, model didn't converge
3. **Architecture too simple** — 128→64 without BatchNorm on 75-feature input
4. **Old NN output range**: probabilities [0.01, 1.00] with mean=0.92 — model was predicting nearly all samples as positive class with high confidence, classic sign of non-convergence

**Resolution**: Retrained with:
- Architecture: 256→128→64 with BatchNorm + Dropout
- 50 epochs with cosine LR schedule (5e-4 → 0)
- Training on CPU (stable with NumPy 2.x after torch upgrade to 2.10.0)
- BCEWithLogitsLoss with pos_weight for class imbalance
- Result: **AUROC=0.722** (now the best model), F1=0.449

**Saved**: `models/nn_baseline.pt` (v2), `models/nn_config.pkl`

### Issue 3: LR underperformance (MODERATE — UNDERSTOOD)

**Symptom**: LR AUROC=0.555 vs paper's 0.665.

**Root cause**: Two factors:
1. **Multicollinearity from one-hot encoding**: `encode_categoricals` uses `drop_first=False`, creating perfectly correlated dummy variables. LR is sensitive to this; tree models are not.
2. **Feature count difference**: 75 vs paper's 87 features — different encoding or column selection.

**Recommendation for Phase 3**: Not blocking. LR is included as a baseline comparison, not a primary model. If reproduction of paper's LR results is needed, investigate: (a) which 87 features the paper uses, (b) whether they use `drop_first=True`.

### Issue 4: PyTorch/NumPy compatibility (MODERATE — RESOLVED)

**Symptom**: `RuntimeError: Numpy is not available` when importing torch.

**Root cause**: Conda installed PyTorch 2.2.2 (compiled against NumPy 1.x) but pip installed NumPy 2.4.3.

**Resolution**: Upgraded torch to 2.10.0 via pip. Set `KMP_DUPLICATE_LIB_OK=TRUE` in conda env activation script to resolve OpenMP duplicate library warning.

**Saved**: `/Users/abdro/anaconda3/envs/scoring/etc/conda/activate.d/env_vars.sh`

---

## SHAP Feature Importance (Top 10)

| Rank | Feature | Mean |SHAP| |
|------|---------|-----------------|
| 1 | int_rate | 0.4493 |
| 2 | term | 0.1888 |
| 3 | acc_open_past_24mths | 0.1306 |
| 4 | annual_inc | 0.1292 |
| 5 | dti | 0.1193 |
| 6 | fico | 0.0895 |
| 7 | loan_amnt | 0.0812 |
| 8 | tot_hi_cred_lim | 0.0686 |
| 9 | avg_cur_bal | 0.0635 |
| 10 | total_bc_limit | 0.0618 |

**Paper's expected top features**: FICO range, loan amount, annual income, DTI, purpose (credit card/debt consolidation), home ownership, employment length, state.

**Overlap analysis**: 5/10 match (int_rate, annual_inc, dti, fico, loan_amnt). Differences:
- We have `acc_open_past_24mths`, `tot_hi_cred_lim`, `avg_cur_bal`, `total_bc_limit`, `term` in top 10
- Paper has `purpose`, `home_ownership`, `emp_length`, `state` — these are one-hot encoded in our data, so their SHAP is split across multiple dummy columns

**AUROC with 10 features**: 0.707 (only 1% drop from all 75 features — confirms paper's finding of 88.5% feature reduction with minimal performance loss).

**Saved**: `reports/shap_importance.csv`, `reports/top10_features.csv`

---

## Cross-Check: Feature Rankings Across Models

| Rank | SHAP (XGB) | LR |coef| | EBM | LGBM |
|------|------------|-----------|-----|------|
| 1 | int_rate | installment | int_rate | annual_inc |
| 2 | term | fico | term | int_rate |
| 3 | acc_open_past_24mths | mths_since_recent_bc | annual_inc | dti |
| 4 | annual_inc | term | acc_open_past_24mths | acc_open_past_24mths |
| 5 | dti | percent_bc_gt_75 | dti | mo_sin_old_rev_tl_op |
| 6 | fico | int_rate | loan_amnt | cr_hist_months |
| 7 | loan_amnt | mo_sin_rcnt_rev_tl_op | mo_sin_rcnt_rev_tl_op | loan_amnt |
| 8 | tot_hi_cred_lim | mo_sin_old_rev_tl_op | tot_hi_cred_lim | installment |
| 9 | avg_cur_bal | dti | total_bc_limit | revol_bal |
| 10 | total_bc_limit | mo_sin_rcnt_tl | fico | total_bc_limit |

**Observation**: `int_rate`, `term`, `dti`, and `annual_inc` appear consistently in top ranks across all model types — these are robust predictors. Rankings are "largely consistent" as the paper states.

---

## Timing (M4 Pro MacBook)

| Step | Duration |
|------|----------|
| Phase 1: Full pipeline (load→EDA→save) | 77s |
| LR training | 28s |
| XGBoost training | 1s |
| LightGBM training | 3s |
| EBM training (10,000 rounds) | ~19 min |
| Neural Net v2 training (50 epochs) | ~3 min |
| SHAP (5000 samples) | ~5s |
| Phase 2 total (excl. EBM) | ~40s |

---

## Files Produced

### Models (`models/`)
- `lr_baseline.pkl`, `xgb_baseline.pkl`, `lgbm_baseline.pkl`, `ebm_baseline.pkl`
- `nn_baseline.pt` (v2: 256-128-64 with BatchNorm), `nn_scaler.pkl`, `nn_config.pkl`

### Reports (`reports/`)
- `baseline_results.csv` — all models, default threshold
- `baseline_results_optimal.csv` — all models, Youden's J threshold
- `optimal_thresholds.csv` — per-model optimal thresholds
- `shap_importance.csv` — all features ranked by SHAP
- `top10_features.csv` — top 10 for Phase 3
- `class_distribution.png`, `feature_distributions.png`, `correlation_heatmap.png`
- `shap_beeswarm.png`, `shap_bar.png`, `auroc_vs_features.png`, `baseline_comparison.png`

### Data (`data/processed/`)
- `X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`

---

## Phase 3 Results — Glass-Box Models on Top 10 Features

| Model | AUROC | AUPRC | F1 | Balanced Accuracy |
|-------|-------|-------|------|-------------------|
| NeuralNet | 0.7148 | 0.3906 | 0.4423 | 0.6561 |
| EBM | 0.7142 | 0.3900 | 0.4421 | 0.6555 |
| LGBM | 0.7135 | 0.3913 | 0.4413 | 0.6550 |
| LR | 0.7085 | 0.3791 | 0.4382 | 0.6524 |
| XGB | 0.7073 | 0.3829 | 0.4370 | 0.6504 |
| PLTR | 0.6395 | 0.3134 | 0.3849 | 0.6013 |

**Key findings**:
- Feature reduction 75 → 10 (86.7%) with <1% AUROC drop for tree/EBM models
- LR *improved* from 0.555 → 0.708 (multicollinearity reduced with fewer one-hot features)
- EBM interactions: marginal gain (confirms paper's 0.4% finding)
- PLTR underperforms (AUROC=0.639) — SAGA solver may not converge fully on 684K rows

---

## Phase 4 Results — Fairness Analysis

Analyzed 5 candidate attributes: income_bracket, home_ownership, dti_bracket, purpose, emp_length.

**Key findings**: See `reports/fairness_summary.csv` for detailed metrics per attribute. Per-group ROC curves and SHAP divergence heatmaps saved for informative attributes.

---

## Phase 5 — Dashboard

Streamlit dashboard deployed at `app/dashboard.py`. Features:
- **Prediction tab**: borrower inputs → default probability + decision (approve/deny)
- **Explanation tab**: SHAP waterfall chart showing per-feature contributions
- **Fairness tab**: fairness metrics summary from Phase 4

Run with: `streamlit run app/dashboard.py`

---

## Files Produced (All Phases)

### Models (`models/`)
- `lr_baseline.pkl`, `xgb_baseline.pkl`, `lgbm_baseline.pkl`, `ebm_baseline.pkl`
- `nn_baseline.pt` (v2: 256-128-64 with BatchNorm), `nn_scaler.pkl`, `nn_config.pkl`
- `ebm_top10.pkl`, `xgb_top10.pkl`, `lgbm_top10.pkl`, `lr_top10.pkl`, `pltr_top10.pkl`
- `nn_top10.pt`, `nn_scaler_top10.pkl`
- `deploy_config.pkl`, `shap_explainer.pkl`

### Reports (`reports/`)
- `baseline_results.csv`, `baseline_results_optimal.csv`, `reduced_results.csv`
- `optimal_thresholds.csv`, `shap_importance.csv`, `top10_features.csv`
- `fairness_summary.csv`, `proxy_leakage.csv`
- All PNG visualizations (class distribution, feature distributions, correlation heatmaps, ROC/PR curves, SHAP plots, fairness charts, model selection, waterfall example)

### Data (`data/processed/`)
- `X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`

---

## Remaining Work: Phase 6 (Report & Presentation)

1. Write project report: introduction, methodology, results, fairness findings, conclusion
2. Prepare presentation slides (10-15 slides)
3. Record demo video of the interactive dashboard
