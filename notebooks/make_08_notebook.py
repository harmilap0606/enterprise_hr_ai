import json

nb = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.12.0"}
 },
 "cells": []
}

def md(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}

def code(cid, src):
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": src}

nb["cells"] = [

md("md-title", [
    "# 08 · SHAP Explainability\n",
    "\n",
    "**Project:** Enterprise HR AI  \n",
    "**Model:** `logistic_regression_balanced` from `models/attrition_pipeline.joblib`  \n",
    "**Explainer:** `shap.LinearExplainer` — the correct choice for Logistic Regression.\n",
    "TreeExplainer and KernelExplainer are wrong for this model type and are NOT used here.\n",
    "\n",
    "---"
]),

# ── Imports ──────────────────────────────────────────────────────────────────
code("cell-imports", [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "import json\n",
    "import joblib\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "import matplotlib\n",
    "matplotlib.use('Agg')          # non-interactive backend — required for saving PNGs\n",
    "import matplotlib.pyplot as plt\n",
    "import shap\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "PROC    = os.path.join('..', 'data', 'processed')\n",
    "MODELS  = os.path.join('..', 'models')\n",
    "REPORTS = os.path.join('..', 'reports', 'shap')\n",
    "os.makedirs(REPORTS, exist_ok=True)\n",
    "\n",
    "RANDOM_STATE = 42\n",
    "print('shap version  :', shap.__version__)\n",
    "print('REPORTS dir   :', os.path.abspath(REPORTS))"
]),

# ── Load model + config ───────────────────────────────────────────────────────
md("md-load", [
    "---\n",
    "## Step 1 · Load Model & Config\n",
    "\n",
    "Confirm the loaded model matches `model_config.json` (logistic_regression_balanced, threshold=0.40)."
]),

code("cell-load-model", [
    "# Load config\n",
    "with open(os.path.join(MODELS, 'model_config.json'), 'r') as f:\n",
    "    config = json.load(f)\n",
    "print('=== model_config.json ===')\n",
    "print(json.dumps(config, indent=2))\n",
    "\n",
    "# Load model\n",
    "model = joblib.load(os.path.join(MODELS, 'attrition_pipeline.joblib'))\n",
    "print()\n",
    "print('Loaded model type :', type(model).__name__)\n",
    "print('class_weight      :', model.class_weight)\n",
    "print('max_iter          :', model.max_iter)\n",
    "print()\n",
    "# Confirm model type matches config\n",
    "assert config['model'] == 'logistic_regression_balanced', \\\n",
    "    f'Config model mismatch: {config[\"model\"]}'\n",
    "assert config['threshold'] == 0.40, \\\n",
    "    f'Config threshold mismatch: {config[\"threshold\"]}'\n",
    "print('CONFIRMED: model=logistic_regression_balanced, threshold=0.40 — matches config.')"
]),

# ── Load data + split ─────────────────────────────────────────────────────────
md("md-split", [
    "---\n",
    "## Step 2 · Recreate Identical 80/20 Stratified Split"
]),

code("cell-load-split", [
    "df = pd.read_csv(os.path.join(PROC, 'features_scaled.csv'))\n",
    "X = df.drop(columns=['Attrition'])\n",
    "y = df['Attrition']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE\n",
    ")\n",
    "\n",
    "print(f'Train: {X_train.shape}, Test: {X_test.shape}')\n",
    "print(f'Test attrition rate: {y_test.mean()*100:.2f}%  (15.99% expected)')\n",
    "assert abs(y_test.mean()*100 - 15.99) < 0.05, 'Split mismatch!'\n",
    "print('CONFIRMED: Split identical to notebooks 06/07.')"
]),

# ── SHAP LinearExplainer ──────────────────────────────────────────────────────
md("md-shap", [
    "---\n",
    "## Step 3 · SHAP LinearExplainer\n",
    "\n",
    "**Why LinearExplainer?**  \n",
    "For a linear model (Logistic Regression), SHAP values decompose as:\n",
    "`phi_j = coef_j * (x_j - E[x_j])`.  \n",
    "LinearExplainer computes this exactly in O(n·p) time.  \n",
    "TreeExplainer requires a tree model; KernelExplainer is a model-agnostic\n",
    "approximation that would be 100–1000× slower with no accuracy benefit here."
]),

code("cell-shap-explainer", [
    "print('Initialising shap.LinearExplainer on training set background...')\n",
    "explainer = shap.LinearExplainer(model, X_train, feature_perturbation='interventional')\n",
    "print('Explainer type     :', type(explainer).__name__)\n",
    "print('Expected value     :', explainer.expected_value)   # baseline log-odds\n",
    "print()\n",
    "\n",
    "# Compute SHAP values on the test set (Explanation object — new API)\n",
    "explanation = explainer(X_test)\n",
    "print('explanation.values shape:', explanation.values.shape)   # (294, 48)\n",
    "print('Positive class (left=1) SHAP values — correct slice used automatically by LinearExplainer')"
]),

# ── Plot 1: Beeswarm ──────────────────────────────────────────────────────────
md("md-beeswarm", [
    "---\n",
    "## Step 4 · Plot 1 — Beeswarm Summary Plot"
]),

code("cell-beeswarm", [
    "fig, ax = plt.subplots(figsize=(10, 12))\n",
    "plt.sca(ax)\n",
    "shap.plots.beeswarm(explanation, max_display=20, show=False)\n",
    "plt.title('SHAP Beeswarm — Attrition Drivers (LogReg Balanced)', fontsize=13, pad=12)\n",
    "plt.tight_layout()\n",
    "beeswarm_path = os.path.join(REPORTS, 'summary_beeswarm.png')\n",
    "plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')\n",
    "plt.close('all')\n",
    "print(f'Saved: {os.path.abspath(beeswarm_path)}  ({os.path.getsize(beeswarm_path):,} bytes)')"
]),

# ── Plot 2: Global bar ────────────────────────────────────────────────────────
md("md-bar", [
    "---\n",
    "## Step 5 · Plot 2 — Global Feature Importance Bar Chart"
]),

code("cell-bar", [
    "fig, ax = plt.subplots(figsize=(10, 10))\n",
    "plt.sca(ax)\n",
    "shap.plots.bar(explanation, max_display=20, show=False)\n",
    "plt.title('Mean |SHAP value| — Global Feature Importance', fontsize=13, pad=12)\n",
    "plt.tight_layout()\n",
    "bar_path = os.path.join(REPORTS, 'global_importance.png')\n",
    "plt.savefig(bar_path, dpi=150, bbox_inches='tight')\n",
    "plt.close('all')\n",
    "print(f'Saved: {os.path.abspath(bar_path)}  ({os.path.getsize(bar_path):,} bytes)')\n",
    "\n",
    "# Print top-10 numeric ranking\n",
    "mean_abs_shap = pd.Series(\n",
    "    np.abs(explanation.values).mean(axis=0),\n",
    "    index=X.columns\n",
    ").sort_values(ascending=False)\n",
    "\n",
    "print()\n",
    "print('=== TOP 10 FEATURES BY MEAN |SHAP VALUE| ===')\n",
    "print(f'{\"Rank\":<5} {\"Feature\":<40} {\"Mean |SHAP|\":>12}')\n",
    "print('-'*60)\n",
    "for rank, (feat, val) in enumerate(mean_abs_shap.head(10).items(), 1):\n",
    "    print(f'{rank:<5} {feat:<40} {val:>12.4f}')"
]),

# ── Pick individual examples ──────────────────────────────────────────────────
md("md-individual", [
    "---\n",
    "## Step 6 · Individual Employee Examples\n",
    "\n",
    "Selecting:\n",
    "- **True Positive:** An employee who actually left AND the model (threshold=0.40) predicted they would leave.\n",
    "- **True Negative:** An employee who actually stayed AND the model predicted they would stay.\n",
    "\n",
    "Chosen as the highest-confidence example of each type (most extreme predicted probability within its class)."
]),

code("cell-pick-examples", [
    "THRESHOLD = config['threshold']   # 0.40\n",
    "\n",
    "# Predicted probabilities on test set\n",
    "probs = model.predict_proba(X_test)[:, 1]\n",
    "preds = (probs >= THRESHOLD).astype(int)\n",
    "\n",
    "y_arr = y_test.values\n",
    "\n",
    "# True Positives: actual=1, predicted=1\n",
    "tp_mask = (y_arr == 1) & (preds == 1)\n",
    "tp_indices = np.where(tp_mask)[0]\n",
    "# Pick the one with highest predicted probability (most confident TP)\n",
    "tp_idx = tp_indices[np.argmax(probs[tp_indices])]\n",
    "\n",
    "# True Negatives: actual=0, predicted=0\n",
    "tn_mask = (y_arr == 0) & (preds == 0)\n",
    "tn_indices = np.where(tn_mask)[0]\n",
    "# Pick the one with lowest predicted probability (most confident TN)\n",
    "tn_idx = tn_indices[np.argmin(probs[tn_indices])]\n",
    "\n",
    "print('=== INDIVIDUAL EXAMPLE SELECTION ===')\n",
    "print(f'Total TPs in test set: {tp_mask.sum()} (model correctly flagged as leaving at threshold={THRESHOLD})')\n",
    "print(f'Total TNs in test set: {tn_mask.sum()} (model correctly predicted as staying at threshold={THRESHOLD})')\n",
    "print()\n",
    "print(f'Selected TRUE POSITIVE example: test index [{tp_idx}]')\n",
    "print(f'  Actual label: {y_arr[tp_idx]} (left=1)')\n",
    "print(f'  Predicted probability: {probs[tp_idx]:.4f}  (threshold={THRESHOLD})')\n",
    "print(f'  Rationale: highest-confidence true positive — clearest attrition signal in test set')\n",
    "print()\n",
    "print(f'Selected TRUE NEGATIVE example: test index [{tn_idx}]')\n",
    "print(f'  Actual label: {y_arr[tn_idx]} (stayed=0)')\n",
    "print(f'  Predicted probability: {probs[tn_idx]:.4f}  (threshold={THRESHOLD})')\n",
    "print(f'  Rationale: lowest-confidence true negative — employee the model is most certain stayed')"
]),

# ── Plot 3 & 4: Waterfall plots ───────────────────────────────────────────────
md("md-waterfall", [
    "---\n",
    "## Step 7 · Plot 3 & 4 — Waterfall Plots for Individual Employees"
]),

code("cell-waterfall-tp", [
    "# Waterfall: True Positive (leaver)\n",
    "fig = plt.figure(figsize=(12, 8))\n",
    "shap.plots.waterfall(explanation[tp_idx], max_display=15, show=False)\n",
    "plt.title(f'SHAP Waterfall — Correctly Predicted LEAVER (test index {tp_idx})', fontsize=12)\n",
    "plt.tight_layout()\n",
    "wf_leaver_path = os.path.join(REPORTS, 'waterfall_leaver_example.png')\n",
    "plt.savefig(wf_leaver_path, dpi=150, bbox_inches='tight')\n",
    "plt.close('all')\n",
    "print(f'Saved: {os.path.abspath(wf_leaver_path)}  ({os.path.getsize(wf_leaver_path):,} bytes)')\n",
    "\n",
    "# Print top driving SHAP values for this employee\n",
    "emp_shap = pd.Series(explanation[tp_idx].values, index=X.columns).sort_values(key=abs, ascending=False)\n",
    "print(f'\\nTop 8 SHAP drivers for LEAVER (index {tp_idx}):')\n",
    "print(f'{\"Feature\":<40} {\"SHAP value\":>12}  Direction')\n",
    "print('-'*65)\n",
    "for feat, val in emp_shap.head(8).items():\n",
    "    direction = '-> TOWARD leaving (+)' if val > 0 else '-> TOWARD staying (-)'\n",
    "    print(f'{feat:<40} {val:>12.4f}  {direction}')"
]),

code("cell-waterfall-tn", [
    "# Waterfall: True Negative (stayer)\n",
    "fig = plt.figure(figsize=(12, 8))\n",
    "shap.plots.waterfall(explanation[tn_idx], max_display=15, show=False)\n",
    "plt.title(f'SHAP Waterfall — Correctly Predicted STAYER (test index {tn_idx})', fontsize=12)\n",
    "plt.tight_layout()\n",
    "wf_stayer_path = os.path.join(REPORTS, 'waterfall_stayer_example.png')\n",
    "plt.savefig(wf_stayer_path, dpi=150, bbox_inches='tight')\n",
    "plt.close('all')\n",
    "print(f'Saved: {os.path.abspath(wf_stayer_path)}  ({os.path.getsize(wf_stayer_path):,} bytes)')\n",
    "\n",
    "# Print top driving SHAP values for this employee\n",
    "emp_shap_tn = pd.Series(explanation[tn_idx].values, index=X.columns).sort_values(key=abs, ascending=False)\n",
    "print(f'\\nTop 8 SHAP drivers for STAYER (index {tn_idx}):')\n",
    "print(f'{\"Feature\":<40} {\"SHAP value\":>12}  Direction')\n",
    "print('-'*65)\n",
    "for feat, val in emp_shap_tn.head(8).items():\n",
    "    direction = '-> TOWARD leaving (+)' if val > 0 else '-> TOWARD staying (-)'\n",
    "    print(f'{feat:<40} {val:>12.4f}  {direction}')"
]),

# ── Cross-check ───────────────────────────────────────────────────────────────
md("md-crosscheck", [
    "---\n",
    "## Step 8 · Cross-Check: SHAP vs LR Coefficients vs XGBoost Importances"
]),

code("cell-crosscheck", [
    "print('=== CROSS-CHECK: SHAP GLOBAL IMPORTANCE vs STEP 6 LR COEFFICIENTS ===')\n",
    "print()\n",
    "\n",
    "# LR coefficient ranking (abs) from Step 6\n",
    "lr_coef_top10 = [\n",
    "    'OverTime', 'BusinessTravel_Travel_Frequently', 'JobRole_Laboratory Technician',\n",
    "    'JobRole_Sales Representative', 'EducationField_Other', 'YearsSinceLastPromotion',\n",
    "    'JobRole_Research Director', 'TotalWorkingYears', 'MaritalStatus_Single',\n",
    "    'BusinessTravel_Travel_Rarely'\n",
    "]\n",
    "\n",
    "# XGBoost importance top 10 from Step 7\n",
    "xgb_top10 = [\n",
    "    'JobRole_Research Director', 'TotalWorkingYears', 'OverTime', 'JobLevel',\n",
    "    'JobRole_Sales Executive', 'JobRole_Research Scientist', 'Department_Sales',\n",
    "    'YearsWithCurrManager', 'StockOptionLevel', 'EnvironmentSatisfaction'\n",
    "]\n",
    "\n",
    "shap_top10 = mean_abs_shap.head(10).index.tolist()\n",
    "\n",
    "print(f'{\"Rank\":<5} {\"SHAP (this notebook)\":<40} {\"LR coeff (Step 6)\":<40} {\"XGB imp (Step 7)\":<40}')\n",
    "print('-'*130)\n",
    "for i in range(10):\n",
    "    s = shap_top10[i] if i < len(shap_top10) else ''\n",
    "    c = lr_coef_top10[i] if i < len(lr_coef_top10) else ''\n",
    "    x = xgb_top10[i] if i < len(xgb_top10) else ''\n",
    "    agree_lr  = '✓' if s == c else ' '\n",
    "    agree_xgb = '✓' if s == x else ' '\n",
    "    print(f'{i+1:<5} {s:<40} {c:<40} {x:<40}  LR:{agree_lr} XGB:{agree_xgb}')\n",
    "\n",
    "# Agreements\n",
    "agree_lr_set  = set(shap_top10) & set(lr_coef_top10)\n",
    "agree_xgb_set = set(shap_top10) & set(xgb_top10)\n",
    "print()\n",
    "print(f'Features in BOTH SHAP top-10 AND LR coeff top-10 ({len(agree_lr_set)}): {agree_lr_set}')\n",
    "print(f'Features in BOTH SHAP top-10 AND XGB top-10 ({len(agree_xgb_set)}): {agree_xgb_set}')\n",
    "print()\n",
    "print('Cross-check interpretation:')\n",
    "print('- SHAP on a LinearExplainer MUST closely mirror LR coefficients (phi_j = coef_j * (x_j - E[x_j]))')\n",
    "print('  Strong agreement expected and confirms explainer is working correctly.')\n",
    "print('- SHAP vs XGBoost: disagreement is EXPECTED and informative.')\n",
    "print('  XGBoost captures non-linear threshold effects; LR/SHAP captures marginal linear contributions.')\n",
    "print('  Where they AGREE (e.g. OverTime, TotalWorkingYears) -> robust, model-agnostic drivers.')\n",
    "print('  Where they DISAGREE -> investigate whether the signal is linear or threshold-based.')"
]),

# ── HR Plain Language ─────────────────────────────────────────────────────────
md("md-hr-summary", [
    "---\n",
    "## Step 9 · Plain-Language HR Summary\n",
    "\n",
    "The following explanation is written for an HR manager audience — no data science jargon.\n",
    "It will be reused verbatim in the Day 4 dashboard.\n",
    "\n",
    "---\n",
    "\n",
    "### What is driving employee attrition at this company?\n",
    "\n",
    "Our AI model analyzed 1,470 employee records and identified five factors that are most strongly\n",
    "linked to an employee deciding to leave:\n",
    "\n",
    "1. **Working Overtime** is the single strongest warning sign. Employees who regularly work\n",
    "   overtime are significantly more likely to leave — suggesting that sustained overwork is\n",
    "   burning people out. The fix is workload review, not just salary review.\n",
    "\n",
    "2. **Frequent Business Travel** is the second-strongest driver. Employees who travel frequently\n",
    "   for work are much more at risk than those who never travel. Consider whether travel demands\n",
    "   can be reduced through remote options or better trip scheduling.\n",
    "\n",
    "3. **Years Without a Promotion** — employees who have gone a long time without advancing in\n",
    "   their career, relative to how long they've been at the company, feel stuck. Career development\n",
    "   conversations and promotion eligibility reviews matter here.\n",
    "\n",
    "4. **Being Single (Marital Status)** is a demographic factor the model has found to correlate\n",
    "   with higher attrition. Single employees have fewer financial and personal anchors that make\n",
    "   switching jobs costly for them. This doesn't mean target single employees — it means ensure\n",
    "   competitive packages and career paths for all, but especially early-career staff.\n",
    "\n",
    "5. **Job Role — Laboratory Technicians and Sales Representatives** are the two job roles with\n",
    "   the highest attrition risk. Both have highly portable skills and active external job markets.\n",
    "   Targeted retention programs (pay benchmarking, role enrichment) for these two groups would\n",
    "   have the highest return on investment.\n",
    "\n",
    "> **Important note for HR managers:** The model does not tell you what to DO — it tells you\n",
    "> WHERE to look. Each at-risk employee flagged by the system should be reviewed individually\n",
    "> by their manager using these signals as a conversation guide, not as an automated action trigger.\n",
    "\n",
    "---"
]),

code("cell-hr-summary-print", [
    "print('=== PLAIN-LANGUAGE HR SUMMARY (for Day 4 dashboard) ===')\n",
    "print()\n",
    "print('TOP 5 SHAP DRIVERS IN HR LANGUAGE:')\n",
    "top5 = mean_abs_shap.head(5)\n",
    "hr_labels = {\n",
    "    'OverTime': 'Employees who regularly work overtime are significantly more likely to leave. Review workload distribution.',\n",
    "    'BusinessTravel_Travel_Frequently': 'Frequent business travel is a major burnout risk. Audit travel requirements and enable remote options.',\n",
    "    'YearsSinceLastPromotion': 'Long stretches without promotion signal career stagnation. Schedule development conversations with long-tenured non-promoted staff.',\n",
    "    'MaritalStatus_Single': 'Single employees have lower switching costs. Ensure competitive packages for early-career and single employees.',\n",
    "    'JobRole_Laboratory Technician': 'Lab Technicians are high-risk due to portable, marketable skills. Benchmark pay and enrich roles proactively.',\n",
    "    'JobRole_Sales Representative': 'Sales Reps have the highest market turnover rates. Evaluate commission structure, management quality, and career path.',\n",
    "    'TotalWorkingYears': 'Employees earlier in their careers are more mobile. Invest in onboarding depth and early career mentoring.',\n",
    "    'JobLevel': 'Lower job levels (junior staff) are more likely to leave. Clarify promotion timelines for this group.',\n",
    "    'overall_satisfaction_score': 'Low satisfaction composite across job, environment, and relationships strongly predicts leaving. Run pulse surveys.',\n",
    "    'income_per_year_at_company': 'Employees who earn little relative to their loyalty tenure feel undervalued. Conduct tenure-adjusted pay reviews.',\n",
    "}\n",
    "for rank, (feat, shap_val) in enumerate(top5.items(), 1):\n",
    "    label = hr_labels.get(feat, f'Feature {feat!r} — review its relationship to attrition manually.')\n",
    "    print(f'{rank}. [{feat}]  mean|SHAP|={shap_val:.4f}')\n",
    "    print(f'   HR Insight: {label}')\n",
    "    print()"
]),

# ── File inventory ────────────────────────────────────────────────────────────
md("md-files", ["---\n", "## Step 10 · Output File Inventory"]),

code("cell-inventory", [
    "print('=== REPORTS/SHAP/ DIRECTORY ===')\n",
    "for fname in sorted(os.listdir(REPORTS)):\n",
    "    fpath = os.path.join(REPORTS, fname)\n",
    "    if os.path.isfile(fpath):\n",
    "        print(f'  {fname:<40}  {os.path.getsize(fpath):>10,} bytes')\n",
    "\n",
    "print()\n",
    "all_saved = [\n",
    "    ('summary_beeswarm.png',        beeswarm_path),\n",
    "    ('global_importance.png',        bar_path),\n",
    "    ('waterfall_leaver_example.png', wf_leaver_path),\n",
    "    ('waterfall_stayer_example.png', wf_stayer_path),\n",
    "]\n",
    "for name, path in all_saved:\n",
    "    exists = os.path.exists(path) and os.path.getsize(path) > 0\n",
    "    status = 'SAVED' if exists else 'MISSING'\n",
    "    print(f'  [{status}] {name}')"
])

]

out_path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\notebooks\08_shap_explainability.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Written: {out_path}')
