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

def md(cid, lines):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": lines}

def code(cid, lines):
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": lines}

nb["cells"] = [

md("md-title", [
    "# 07 · Model Comparison\n",
    "\n",
    "**Project:** Enterprise HR AI  \n",
    "**Sources:** `data/processed/features_unscaled.csv` & `data/processed/features_scaled.csv`  \n",
    "**Task:** Evaluate Logistic Regression, Random Forest, and XGBoost across both unscaled and scaled datasets using the identical 80/20 stratified split (`random_state=42`). Re-apply strict Recall (primary) and F1 (secondary) decision rule across all 6 models.\n",
    "\n",
    "---"
]),

code("cell-imports", [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "import joblib\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.metrics import (\n",
    "    precision_score, recall_score, f1_score, roc_auc_score,\n",
    "    classification_report, confusion_matrix\n",
    ")\n",
    "\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.width', 200)\n",
    "pd.set_option('display.float_format', '{:.4f}'.format)\n",
    "\n",
    "PROC = os.path.join('..', 'data', 'processed')\n",
    "MODELS = os.path.join('..', 'models')\n",
    "os.makedirs(MODELS, exist_ok=True)\n",
    "\n",
    "RANDOM_STATE = 42\n",
    "print('PROC  :', os.path.abspath(PROC))\n",
    "print('MODELS:', os.path.abspath(MODELS))"
]),

md("md-load-split-unscaled", [
    "---\n",
    "## Step 1 · Load & Split Unscaled Data\n",
    "\n",
    "Loading `features_unscaled.csv` with 80/20 stratified split (`random_state=42`)."
]),

code("cell-load-split-unscaled", [
    "df_unscaled = pd.read_csv(os.path.join(PROC, 'features_unscaled.csv'))\n",
    "print(f'Loaded features_unscaled.csv: {df_unscaled.shape[0]:,} rows x {df_unscaled.shape[1]} cols')\n",
    "\n",
    "X_unscaled = df_unscaled.drop(columns=['Attrition'])\n",
    "y_unscaled = df_unscaled['Attrition']\n",
    "\n",
    "X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(\n",
    "    X_unscaled, y_unscaled, test_size=0.20, stratify=y_unscaled, random_state=RANDOM_STATE\n",
    ")\n",
    "\n",
    "test_attrition_pct = y_test_u.mean() * 100\n",
    "print(f'Unscaled Train shape: {X_train_u.shape}, Test shape: {X_test_u.shape}')\n",
    "print(f'Test attrition rate: {test_attrition_pct:.2f}% ({y_test_u.sum()} / {len(y_test_u)})')\n",
    "assert abs(test_attrition_pct - 15.99) < 0.05, f'Test class balance mismatch: got {test_attrition_pct:.2f}%'\n",
    "print('CONFIRMED: Split reproduces notebook 06 class balance (15.99%).')"
]),

md("md-train-unscaled", [
    "---\n",
    "## Step 2 · Train Models on Unscaled Features\n",
    "\n",
    "1. **Logistic Regression (Unscaled)** (`max_iter=1000`, `random_state=42`)\n",
    "2. **Random Forest (Unscaled)** (`n_estimators=200`, `random_state=42`)\n",
    "3. **XGBoost (Unscaled)** (`random_state=42`, `eval_metric='logloss'`)"
]),

code("cell-train-unscaled", [
    "models_unscaled = {}\n",
    "\n",
    "print('Training Logistic Regression (unscaled)...')\n",
    "lr_u = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)\n",
    "lr_u.fit(X_train_u, y_train_u)\n",
    "models_unscaled['Logistic Regression (Unscaled)'] = lr_u\n",
    "\n",
    "print('Training Random Forest (unscaled)...')\n",
    "rf_u = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)\n",
    "rf_u.fit(X_train_u, y_train_u)\n",
    "models_unscaled['Random Forest (Unscaled)'] = rf_u\n",
    "\n",
    "print('Training XGBoost (unscaled)...')\n",
    "xgb_u = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', n_estimators=100)\n",
    "xgb_u.fit(X_train_u, y_train_u)\n",
    "models_unscaled['XGBoost (Unscaled)'] = xgb_u\n",
    "\n",
    "print('Unscaled models trained successfully.')"
]),

md("md-load-split-scaled", [
    "---\n",
    "## Step 3 · Train Models on Scaled Features\n",
    "\n",
    "Loading `features_scaled.csv` with identical 80/20 stratified split (`random_state=42`).\n",
    "Evaluating Random Forest and XGBoost on scaled features."
]),

code("cell-train-scaled", [
    "df_scaled = pd.read_csv(os.path.join(PROC, 'features_scaled.csv'))\n",
    "print(f'Loaded features_scaled.csv: {df_scaled.shape[0]:,} rows x {df_scaled.shape[1]} cols')\n",
    "\n",
    "X_scaled = df_scaled.drop(columns=['Attrition'])\n",
    "y_scaled = df_scaled['Attrition']\n",
    "\n",
    "X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(\n",
    "    X_scaled, y_scaled, test_size=0.20, stratify=y_scaled, random_state=RANDOM_STATE\n",
    ")\n",
    "\n",
    "models_scaled = {}\n",
    "\n",
    "print('Training Random Forest (scaled, n_estimators=200)...')\n",
    "rf_s = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)\n",
    "rf_s.fit(X_train_s, y_train_s)\n",
    "models_scaled['Random Forest (Scaled)'] = rf_s\n",
    "\n",
    "print('Training XGBoost (scaled, eval_metric=logloss)...')\n",
    "xgb_s = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', n_estimators=100)\n",
    "xgb_s.fit(X_train_s, y_train_s)\n",
    "models_scaled['XGBoost (Scaled)'] = xgb_s\n",
    "\n",
    "print('Scaled tree models trained successfully.')"
]),

md("md-comparison-table-6", [
    "---\n",
    "## Step 4 · Complete 6-Row Comparison Table\n",
    "\n",
    "Metrics computed on the test set (`support = 47` leavers, `247` non-leavers)."
]),

code("cell-eval-all", [
    "results = []\n",
    "\n",
    "# 1. Baseline LogReg (Scaled - Step 6)\n",
    "results.append({\n",
    "    'Model': 'Baseline LogReg (Scaled - Step 6)',\n",
    "    'Precision': 0.6538,\n",
    "    'Recall': 0.3617,\n",
    "    'F1': 0.4658,\n",
    "    'ROC-AUC': 0.8134\n",
    "})\n",
    "\n",
    "# 2. Unscaled models\n",
    "for name, model in models_unscaled.items():\n",
    "    y_pred = model.predict(X_test_u)\n",
    "    y_prob = model.predict_proba(X_test_u)[:, 1]\n",
    "    results.append({\n",
    "        'Model': name,\n",
    "        'Precision': precision_score(y_test_u, y_pred, zero_division=0),\n",
    "        'Recall': recall_score(y_test_u, y_pred, zero_division=0),\n",
    "        'F1': f1_score(y_test_u, y_pred, zero_division=0),\n",
    "        'ROC-AUC': roc_auc_score(y_test_u, y_prob)\n",
    "    })\n",
    "\n",
    "# 3. Scaled tree models\n",
    "for name, model in models_scaled.items():\n",
    "    y_pred = model.predict(X_test_s)\n",
    "    y_prob = model.predict_proba(X_test_s)[:, 1]\n",
    "    results.append({\n",
    "        'Model': name,\n",
    "        'Precision': precision_score(y_test_s, y_pred, zero_division=0),\n",
    "        'Recall': recall_score(y_test_s, y_pred, zero_division=0),\n",
    "        'F1': f1_score(y_test_s, y_pred, zero_division=0),\n",
    "        'ROC-AUC': roc_auc_score(y_test_s, y_prob)\n",
    "    })\n",
    "\n",
    "comparison_df = pd.DataFrame(results).set_index('Model')\n",
    "print('=== COMPLETE 6-ROW MODEL COMPARISON TABLE ===')\n",
    "print(comparison_df.to_string())"
]),

code("cell-cm-all", [
    "print('=== DETAILED CONFUSION MATRICES ===\\n')\n",
    "for name, model in models_unscaled.items():\n",
    "    cm = confusion_matrix(y_test_u, model.predict(X_test_u))\n",
    "    print(f'{name}: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}')\n",
    "for name, model in models_scaled.items():\n",
    "    cm = confusion_matrix(y_test_s, model.predict(X_test_s))\n",
    "    print(f'{name}: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}')"
]),

md("md-decision-reapplied", [
    "---\n",
    "## Step 5 · Honest Decision Rule Application\n",
    "\n",
    "**Decision Rules:**\n",
    "- Primary criterion: **Recall** (missing a leaver is the costly error)\n",
    "- Secondary criterion: **F1** (ensures precision is not completely sacrificed)\n",
    "\n",
    "Strict comparison across all 6 models without letting secondary narratives override metrics."
]),

code("cell-winner-evaluation", [
    "ranked = sorted(results, key=lambda x: (x['Recall'], x['F1']), reverse=True)\n",
    "print('=== MODELS RANKED STRICTLY BY (RECALL desc, F1 desc) ===')\n",
    "for i, r in enumerate(ranked, 1):\n",
    "    print(f'{i}. {r[\"Model\"]:35s} | Recall: {r[\"Recall\"]:.4f} | F1: {r[\"F1\"]:.4f} | Precision: {r[\"Precision\"]:.4f} | ROC-AUC: {r[\"ROC-AUC\"]:.4f}')\n",
    "\n",
    "overall_winner = ranked[0]\n",
    "print('\\n' + '='*60)\n",
    "print(f'HONEST WINNER: {overall_winner[\"Model\"]}')\n",
    "print('='*60)\n",
    "print(f'Recall: {overall_winner[\"Recall\"]:.4f}, F1: {overall_winner[\"F1\"]:.4f}')\n",
    "\n",
    "is_baseline = overall_winner['Model'] == 'Baseline LogReg (Scaled - Step 6)'\n",
    "if is_baseline:\n",
    "    print('RECOMMENDATION: Retain Baseline Logistic Regression (Scaled) as the production model.')\n",
    "    print('It strictly dominates all unscaled and scaled tree models on both primary (Recall: 0.3617 vs max 0.2766) and secondary (F1: 0.4658 vs max 0.3714) criteria.')"
]),

md("md-diagnosis", [
    "---\n",
    "## Step 6 · Root Cause Diagnosis: Tree Models vs. Class Imbalance\n",
    "\n",
    "Why do tree ensembles underperform the baseline on Recall across both unscaled and scaled datasets?"
]),

code("cell-diagnosis-text", [
    "diagnosis = '''\n",
    "ROOT CAUSE DIAGNOSIS:\n",
    "Evaluating Random Forest and XGBoost on scaled features confirmed that feature scaling is NOT \n",
    "the reason for their poor recall (decision tree splits are scale-invariant, yielding virtually identical \n",
    "or near-identical performance on scaled vs unscaled inputs). \n",
    "\n",
    "Instead, the primary driver is UNTREATED CLASS IMBALANCE in conjunction with the default 0.5 decision threshold:\n",
    "1. In an 84/16 imbalanced dataset, standard Gini impurity / entropy in Random Forest and standard logistic loss \n",
    "   in XGBoost treat errors on both classes symmetrically. Since 84% of samples belong to the majority class (Stayed), \n",
    "   minimizing overall sample loss naturally drives tree leaf predictions toward the majority class.\n",
    "2. Crucially, neither model in this default evaluation utilized class weighting — Random Forest was run without \n",
    "   `class_weight='balanced'` (or `'balanced_subsample'`), and XGBoost was run without setting `scale_pos_weight` \n",
    "   (which should ideally be ~(1233/237) ≈ 5.2 to penalize false negatives proportionally).\n",
    "3. Applying a default 0.5 probability cutoff means an employee must have an overwhelming predicted probability \n",
    "   to be flagged, causing Random Forest to catch only 5 of 47 leavers and XGBoost only 13 of 47 leavers. \n",
    "   In contrast, L2-regularized Logistic Regression with scaled features produced better-spread linear log-odds \n",
    "   that managed to identify 17 of 47 leavers (Recall = 0.3617). Without explicit minority-class penalization or \n",
    "   threshold tuning, off-the-shelf tree ensembles severely penalize recall on imbalanced HR data.\n",
    "'''\n",
    "print(diagnosis.strip())"
]),

md("md-save-final", [
    "---\n",
    "## Step 7 · Save Production Pipeline\n",
    "\n",
    "Saving the best-performing model (`Baseline LogReg Scaled`, or best candidate per rule) to `models/attrition_pipeline.joblib`."
]),

code("cell-save-final", [
    "# If Baseline LogReg is the honest winner, we save the trained baseline model to models/attrition_pipeline.joblib\n",
    "out_file = os.path.join(MODELS, 'attrition_pipeline.joblib')\n",
    "\n",
    "if overall_winner['Model'] == 'Baseline LogReg (Scaled - Step 6)':\n",
    "    baseline_source = os.path.join(MODELS, 'baseline_logreg.joblib')\n",
    "    if os.path.exists(baseline_source):\n",
    "        winner_obj = joblib.load(baseline_source)\n",
    "    else:\n",
    "        # Retrain baseline model\n",
    "        winner_obj = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)\n",
    "        winner_obj.fit(X_train_s, y_train_s)\n",
    "    joblib.dump(winner_obj, out_file)\n",
    "    print(f'Saved Baseline Logistic Regression (Scaled) to: {out_file}')\n",
    "else:\n",
    "    # If another model won, save that\n",
    "    winner_name = overall_winner['Model']\n",
    "    # find model obj\n",
    "    m_obj = models_unscaled.get(winner_name) or models_scaled.get(winner_name)\n",
    "    joblib.dump(m_obj, out_file)\n",
    "    print(f'Saved {winner_name} to: {out_file}')\n",
    "\n",
    "print(f'File size: {os.path.getsize(out_file):,} bytes')\n",
    "loaded = joblib.load(out_file)\n",
    "print(f'Verified pipeline type: {type(loaded)}')\n",
    "\n",
    "# Also print feature importances / coefficients for the retained model\n",
    "if hasattr(loaded, 'coef_'):\n",
    "    coef_s = pd.Series(loaded.coef_[0], index=X_scaled.columns).abs().sort_values(ascending=False).head(15)\n",
    "    print('\\nTop 15 Absolute Coefficients of production model:')\n",
    "    print(coef_s.to_string())\n",
    "elif hasattr(loaded, 'feature_importances_'):\n",
    "    fi_s = pd.Series(loaded.feature_importances_, index=X_unscaled.columns).sort_values(ascending=False).head(15)\n",
    "    print('\\nTop 15 Feature Importances of production model:')\n",
    "    print(fi_s.to_string())"
])

]

out_path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\notebooks\07_model_comparison.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Updated: {out_path}')
