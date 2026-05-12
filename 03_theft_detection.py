# 03_theft_detection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)
import xgboost as xgb
import joblib

df = pd.read_csv('data/processed/smart_grid_features.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

FEATURE_COLS = [
    'consumption_kwh',
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_48h', 'lag_168h',
    'rolling_mean_24h', 'rolling_std_24h', 'rolling_mean_7d',
    'delta_1h',
    'voltage', 'current', 'power_factor', 'weather_encoded',
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour'
]
TARGET = 'theft_flag'

X = df[FEATURE_COLS]
y = df[TARGET]

print(f"Class balance - Normal: {(y==0).sum():,}  |  Theft: {(y==1).sum():,}")
print(f"Theft rate: {y.mean()*100:.1f}%")

# Chronological split - theft detection in production runs forward in time
split_idx = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Validation carve-out from training data for early stopping
val_split = int(len(X_train) * 0.90)
X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

xgb_clf = xgb.XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.80, min_child_weight=3,
    scale_pos_weight=scale_pos_weight,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric='auc',
    random_state=42, tree_method='hist', n_jobs=-1,
    early_stopping_rounds=30
)

xgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
joblib.dump(xgb_clf, 'models/xgb_theft_classifier.pkl')

y_pred = xgb_clf.predict(X_test)
y_prob = xgb_clf.predict_proba(X_test)[:, 1]

print("\nTheft Detection Results")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Theft']))
print(f"AUC-ROC   : {roc_auc_score(y_test, y_prob):.4f}")
print(f"Avg Prec. : {average_precision_score(y_test, y_prob):.4f}")

# CV on training data only
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(
    xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42, tree_method='hist', n_jobs=-1
    ),
    X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1
)
print(f"\n5-Fold CV AUC-ROC: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal', 'Theft'], yticklabels=['Normal', 'Theft'],
            annot_kws={'size': 14})
axes[0].set_title('Confusion Matrix', fontsize=13)
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
axes[1].plot(fpr, tpr, color='steelblue', linewidth=2, label=f'XGBoost (AUC = {auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random baseline')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='steelblue')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontsize=13)
axes[1].legend()

prec, rec, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
axes[2].plot(rec, prec, color='tomato', linewidth=2, label=f'XGBoost (AP = {ap:.3f})')
axes[2].axhline(y_test.mean(), color='k', linestyle='--', linewidth=1, label='No-skill baseline')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].set_title('Precision-Recall Curve', fontsize=13)
axes[2].legend()

plt.tight_layout()
plt.savefig('outputs/figures/04_theft_classification_results.png', dpi=150)
plt.close()

df_test = X_test.copy()
df_test['theft_prob'] = y_prob
df_test['actual'] = y_test.values
hourly_risk = df_test.groupby('hour')['theft_prob'].mean()

fig, ax = plt.subplots(figsize=(12, 4))
hourly_risk.plot(kind='bar', ax=ax, color='tomato', edgecolor='white')
ax.set_title('Average Theft Probability by Hour of Day', fontsize=13)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Mean Theft Probability')
ax.set_xticklabels(range(24), rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figures/05_theft_by_hour.png', dpi=150)
plt.close()

print("\nTheft detection complete. Plots saved.")