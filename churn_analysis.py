"""
╔══════════════════════════════════════════════════════════════╗
║   ONLINE GAMING CHURN PREDICTION — ADVANCED DATA MINING      ║
║   XGBoost, LightGBM, CatBoost, K-Means Clustering & SHAP     ║
╚══════════════════════════════════════════════════════════════╝
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
import os, warnings, json
import joblib
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score,
                             f1_score, accuracy_score, matthews_corrcoef)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
os.makedirs("plots", exist_ok=True)

# ── THEME ──────────────────────────────────────────────────────
BG = "#f4eefa"
CARD = "#ffffff"
NEON_CYAN = "#00b4d8"   # More vibrant cyan
NEON_PINK = "#f72585"   # Much stronger pink
NEON_PURPLE = "#7209b7" # Deep purple
NEON_GREEN = "#06d6a0"  # Bright emerald green
GOLD = "#ff9f1c"        # Strong gold/orange
GRID_COLOR = "#d5c6e0"  # Slightly darker grid lines
TEXT = "#2b1c3d"        # Very dark purple for high contrast text
PALETTE = [NEON_CYAN, NEON_PINK, NEON_PURPLE, NEON_GREEN, GOLD, "#f4a261"]

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID_COLOR,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
    'axes.grid': True, 'grid.color': GRID_COLOR, 'grid.alpha': 0.3,
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"plots/{name}", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"  ✓ saved plots/{name}")

# ═══════════════════════════════════════════════════════════════
# 1. LOAD & ENGINEER
# ═══════════════════════════════════════════════════════════════
print("\n🎮 Loading dataset...")
df = pd.read_csv("online_gaming_behavior_dataset.csv")

# Target: Churn = Low engagement
df['Churn'] = (df['EngagementLevel'] == 'Low').astype(int)
df.drop(columns=['PlayerID','EngagementLevel'], inplace=True)

# ── Feature Engineering ────────────────────────────────────────
print("🔧 Engineering features...")
df['PlayTimePerSession'] = np.where(df['SessionsPerWeek'] > 0,
    df['PlayTimeHours'] / df['SessionsPerWeek'], 0.0)
df['AchievementRate'] = np.where(df['PlayerLevel'] > 0,
    df['AchievementsUnlocked'] / df['PlayerLevel'], 0.0)
df['TotalWeeklyMinutes'] = df['SessionsPerWeek'] * df['AvgSessionDurationMinutes']
df['IsHighLevel'] = (df['PlayerLevel'] >= df['PlayerLevel'].quantile(0.75)).astype(int)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0,18,25,35,50], labels=['Teen','Young Adult','Adult','Senior'])

# Clean any residual NaN / Inf
num_only = df.select_dtypes(include=[np.number]).columns
df[num_only] = df[num_only].replace([np.inf, -np.inf], 0).fillna(0)

# ═══════════════════════════════════════════════════════════════
# 2. PLAYER PERSONAS (UNSUPERVISED LEARNING)
# ═══════════════════════════════════════════════════════════════
print("\n🧩 Performing K-Means Clustering for Player Personas...")
# Select behavioral features for clustering
cluster_features = ['PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 'AvgSessionDurationMinutes', 'AchievementsUnlocked']
X_cluster = df[cluster_features]
scaler_cluster = MinMaxScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Persona'] = kmeans.fit_predict(X_cluster_scaled)
persona_names = {0: "Casual Players", 1: "Hardcore Grinders", 2: "Whales (High Spenders)"}
df['PersonaName'] = df['Persona'].map(persona_names)

# 01 ── Radar Chart for Personas ────────────────────────────────
categories = cluster_features
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
fig.patch.set_facecolor(BG)

for i, (color, ax) in enumerate(zip([NEON_CYAN, NEON_PINK, GOLD], axes)):
    ax.set_facecolor(BG)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=TEXT, size=10)
    ax.tick_params(axis='x', pad=15)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels([], color=GRID_COLOR, size=8)
    ax.set_ylim(0, 1)

    # Add subtle grid rings
    ax.grid(color=GRID_COLOR, linestyle='--', linewidth=1, alpha=0.7)

    values = kmeans.cluster_centers_[i].tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2.5, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.5)
    ax.set_title(persona_names[i], size=14, color=color, fontweight='bold', y=1.15)

fig.suptitle("Player Personas (Behavioral Clusters)", size=18, color=TEXT, y=1.05, fontweight='bold')
plt.tight_layout()
save(fig, "01_persona_radar.png")

# ═══════════════════════════════════════════════════════════════
# 2b. EDA PLOTS
# ═══════════════════════════════════════════════════════════════
print("\n📊 Generating EDA plots...")

# Removed Churn Distribution

# 02 ── Correlation Heatmap ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
corr_cols = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek',
             'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked',
             'PlayTimePerSession', 'AchievementRate', 'TotalWeeklyMinutes', 'Churn']
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
from matplotlib.colors import LinearSegmentedColormap
neon_cmap = LinearSegmentedColormap.from_list("neon", [NEON_PINK, BG, NEON_CYAN])
sns.heatmap(corr, mask=mask, cmap=neon_cmap, annot=True, fmt='.2f', center=0,
            ax=ax, linewidths=0.5, linecolor=GRID_COLOR, annot_kws={'size': 9},
            cbar_kws={'shrink': 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=17, fontweight='bold')
save(fig, "02_correlation_heatmap.png")

# 03 ── Feature Distributions: Churned vs Retained ─────────────
dist_features = ['PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes',
                 'TotalWeeklyMinutes', 'PlayerLevel', 'AchievementsUnlocked']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, (feat, ax) in enumerate(zip(dist_features, axes.flatten())):
    for label, color, name in [(0, NEON_CYAN, 'Retained'), (1, NEON_PINK, 'Churned')]:
        data = df[df['Churn'] == label][feat]
        ax.hist(data, bins=30, color=color, alpha=0.6, label=name, edgecolor=BG)
    ax.set_title(feat, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID_COLOR)
fig.suptitle("Feature Distributions by Churn Status", fontsize=17, fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, "03_feature_distributions.png")

# Removed Box Plots

# 05 ── Persona Churn Rate ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
persona_churn = df.groupby('PersonaName')['Churn'].agg(['mean', 'count']).reset_index()
persona_churn.columns = ['Persona', 'ChurnRate', 'Count']
bars = ax.bar(persona_churn['Persona'], persona_churn['ChurnRate'],
              color=[NEON_CYAN, NEON_PINK, GOLD], edgecolor=BG, width=0.5)
for bar, rate, count in zip(bars, persona_churn['ChurnRate'], persona_churn['Count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{rate:.1%}\n(n={count:,})', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=TEXT)
ax.set_title("Churn Rate by Player Persona", fontsize=17, fontweight='bold')
ax.set_ylabel("Churn Rate")
ax.set_ylim(0, max(persona_churn['ChurnRate']) * 1.35)
save(fig, "04_persona_churn_rate.png")

# ═══════════════════════════════════════════════════════════════
# 3. PREPROCESSING FOR CLASSIFICATION
# ═══════════════════════════════════════════════════════════════
print("\n⚙️  Preprocessing...")
X = df.drop(columns=['Churn', 'Persona', 'PersonaName'])
y = df['Churn']

cat_features = ['Gender','Location','GameGenre','GameDifficulty','AgeGroup']
num_features = [c for c in X.columns if c not in cat_features]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
])

X_processed = preprocessor.fit_transform(X)
X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=0.0, neginf=0.0)
cat_encoded = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
feature_names = list(num_features) + list(cat_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# 06 ── Class Balance Before/After SMOTE ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, title in zip(axes, [y_train, y_train_sm], ['Before SMOTE', 'After SMOTE']):
    counts = pd.Series(data).value_counts().sort_index()
    bars = ax.bar(['Retained (0)', 'Churned (1)'], counts.values,
                  color=[NEON_CYAN, NEON_PINK], edgecolor=BG, width=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{v:,}', ha='center', va='bottom', fontsize=12, fontweight='bold', color=TEXT)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
fig.suptitle('Class Balance: SMOTE Oversampling Effect', fontsize=17, fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, '05_smote_balance.png')

# ═══════════════════════════════════════════════════════════════
# 4. MULTI-MODEL TRAINING (INCL. LIGHTGBM & CATBOOST)
# ═══════════════════════════════════════════════════════════════
print("\n🤖 Training models...")

scale_pos = (len(y_train)-sum(y_train))/sum(y_train)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', tree_method='hist', random_state=42, scale_pos_weight=scale_pos),
    'LightGBM': lgb.LGBMClassifier(objective='binary', random_state=42, scale_pos_weight=scale_pos, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=300, random_seed=42, scale_pos_weight=scale_pos, verbose=0, thread_count=-1, allow_writing_files=False)
}

results = {}
for name, model in models.items():
    print(f"   Training {name}...")
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    # CatBoost returns boolean arrays sometimes, ensure ints
    if name == 'CatBoost':
        y_pred = y_pred.astype(int)

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_proba': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'mcc': matthews_corrcoef(y_test, y_pred),
    }
    print(f"     AUC={results[name]['auc']:.4f}  F1={results[name]['f1']:.4f}")

# Find best model based on F1
best_model_name = max(results, key=lambda k: results[k]['f1'])
print(f"\n🏆 Best Base Model: {best_model_name}")

# Hyperparameter tuning for the best model (XGBoost is typically chosen, but let's tune XGBoost regardless)
print("\n🔍 Hyperparameter tuning XGBoost...")
param_grid = {
    'max_depth': [5, 7],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [200, 300]
}
grid = GridSearchCV(
    xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', tree_method='hist', random_state=42, scale_pos_weight=scale_pos),
    param_grid, cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring='f1', n_jobs=-1, verbose=0)
grid.fit(X_train_sm, y_train_sm)
best_xgb = grid.best_estimator_

y_pred_best = best_xgb.predict(X_test)
y_proba_best = best_xgb.predict_proba(X_test)[:,1]
results['Tuned XGBoost'] = {
    'model': best_xgb, 'y_pred': y_pred_best, 'y_proba': y_proba_best,
    'accuracy': accuracy_score(y_test, y_pred_best),
    'f1': f1_score(y_test, y_pred_best),
    'auc': roc_auc_score(y_test, y_proba_best),
    'mcc': matthews_corrcoef(y_test, y_pred_best),
}
print(f"   Tuned XGBoost — F1={results['Tuned XGBoost']['f1']:.4f}")

# ═══════════════════════════════════════════════════════════════
# 5. EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════════
print("\n📈 Generating evaluation plots...")

# 5a ── Model comparison radar-style bar chart ──────────────────
fig, ax = plt.subplots(figsize=(14,6))
metrics = ['accuracy','f1','auc','mcc']
metric_labels = ['Accuracy','F1-Score (Churn)','ROC-AUC','MCC']
x = np.arange(len(metrics))
model_names = list(results.keys())
w = 0.8 / len(model_names)

for i, name in enumerate(model_names):
    vals = [results[name][m] for m in metrics]
    color = PALETTE[i % len(PALETTE)]
    bars = ax.bar(x + i*w, vals, w, label=name, color=color, edgecolor=BG, alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)

ax.set_xticks(x + w * (len(model_names)-1) / 2)
ax.set_xticklabels(metric_labels, fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_title("Advanced Model Performance Comparison", fontsize=17, fontweight='bold')
ax.legend(facecolor=CARD, edgecolor=GRID_COLOR, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
save(fig, "06_model_comparison.png")

# 5b ── ROC curves overlay ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,8))
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    color = PALETTE[i % len(PALETTE)]
    ax.plot(fpr, tpr, color=color, lw=2.5 if name == 'Tuned XGBoost' else 1.5, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1], ls='--', color='#555', lw=1.5)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Model Ensemble", fontsize=16, fontweight='bold')
ax.legend(facecolor=CARD, edgecolor=GRID_COLOR, fontsize=10)
save(fig, "07_roc_curves.png")

# 5c ── Precision-Recall curves ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10,8))
for i, (name, res) in enumerate(results.items()):
    prec, rec, _ = precision_recall_curve(y_test, res['y_proba'])
    ap = average_precision_score(y_test, res['y_proba'])
    color = PALETTE[i % len(PALETTE)]
    ax.plot(rec, prec, color=color, lw=2.5 if name == 'Tuned XGBoost' else 1.5, label=f"{name} (AP={ap:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves", fontsize=16, fontweight='bold')
ax.legend(facecolor=CARD, edgecolor=GRID_COLOR, fontsize=10)
save(fig, "08_precision_recall.png")

# 5d ── Confusion Matrices Grid ─────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
for idx, (name, res) in enumerate(results.items()):
    ax = axes.flatten()[idx]
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'],
                linewidths=1, linecolor=GRID_COLOR, cbar=False,
                annot_kws={'size': 14, 'fontweight': 'bold'})
    ax.set_title(name, fontsize=13, fontweight='bold', color=PALETTE[idx % len(PALETTE)])
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
# Hide the extra subplot if any
for j in range(len(results), len(axes.flatten())):
    axes.flatten()[j].set_visible(False)
fig.suptitle('Confusion Matrices — All Models', fontsize=17, fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, '09_confusion_matrices.png')

# 5e ── Feature Importance (Tuned XGBoost) ──────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
importances = best_xgb.feature_importances_
sorted_idx = np.argsort(importances)[-19:]  # top 19
ax.barh(range(len(sorted_idx)), importances[sorted_idx], color=NEON_CYAN, edgecolor=BG, alpha=0.85)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=11)
ax.set_ylim(-0.5, len(sorted_idx) - 0.5)
ax.set_xlabel('Importance Score')
ax.set_title('Top 19 Feature Importances — Tuned XGBoost', fontsize=16, fontweight='bold')
for i, v in enumerate(importances[sorted_idx]):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=10, color=NEON_CYAN)
save(fig, '10_feature_importance.png')

# Removed Learning Curve and Calibration Curves

# ═══════════════════════════════════════════════════════════════
# 6. SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════
print("\n🧠 SHAP analysis on Tuned XGBoost...")
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)

# 6a ── SHAP Summary Plot ───────────────────────────────────────
fig = plt.figure(figsize=(10, 14))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, max_display=19, plot_size=(10, 14))
plt.gcf().patch.set_facecolor(BG); plt.gca().set_facecolor(CARD)
plt.gca().tick_params(colors=TEXT)
save(fig, '11_shap_summary.png')

# 6b ── SHAP Feature Importance (Bar) ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 14))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False, max_display=19, color=NEON_PURPLE, plot_size=(10, 14))
plt.gcf().patch.set_facecolor(BG); plt.gca().set_facecolor(CARD)
plt.gca().tick_params(colors=TEXT)
save(fig, '12_shap_bar.png')

# Removed SHAP Waterfall Plot

print("\n💾 Model training complete (saving disabled).")

print("\n✅ Advanced Analysis Complete!")
