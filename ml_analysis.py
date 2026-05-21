

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    StratifiedKFold, 
    cross_val_score, 
    train_test_split,
    cross_validate
)
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    matthews_corrcoef,
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc
)
from sklearn.ensemble import RandomForestClassifier
from tpot import TPOTClassifier
import shap
import pickle
from datetime import datetime

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



################################################################################
# 1. LOAD PREPROCESSED DATA
################################################################################

print("\n[STEP 1] Loading preprocessed data...")

# Load data from R preprocessing script

data_file = "metabolomics_data_processed.csv"

try:
    df = pd.read_csv(data_file)
    print(f"✓ Data loaded: {data_file}")
except FileNotFoundError:
    print(f"ERROR: File not found: {data_file}")
    print("Please run the R preprocessing script first (univariate_analysis_CORRECTED.R)")
    exit(1)

# Separate features and labels
X = df.drop('Group', axis=1)
y = df['Group'].values

feature_names = X.columns.tolist()

print(f"  Samples: {len(X)}")
print(f"  Features: {len(feature_names)}")
print(f"  Controls: {sum(y == 0)}")
print(f"  CRC cases: {sum(y == 1)}")
print(f"  Class balance: {sum(y==1)/len(y):.1%} CRC")

################################################################################
# 2. TRAIN-TEST SPLIT
################################################################################

print("\n[STEP 2] Creating train-test split...")

# Split data: 80% train, 20% test (hold-out)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"  Training set: {len(X_train)} samples")
print(f"  Test set (hold-out): {len(X_test)} samples")
print(f"  Training CRC%: {sum(y_train==1)/len(y_train):.1%}")
print(f"  Test CRC%: {sum(y_test==1)/len(y_test):.1%}")

################################################################################
# 3. NESTED CROSS-VALIDATION WITH TPOT
################################################################################

print("\n[STEP 3] Nested Cross-Validation (5x5) with TPOT...")
print("  This may take 10-30 minutes depending on your CPU...")

# Outer CV for performance estimation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Storage for nested CV results
nested_cv_scores = {
    'accuracy': [],
    'roc_auc': [],
    'mcc': []
}

fold_num = 0
best_models = []

print("\nStarting outer CV folds...")

for train_idx, val_idx in outer_cv.split(X_train, y_train):
    fold_num += 1
    print(f"\n  Outer Fold {fold_num}/5:")
    
    # Split data for this outer fold
    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_train = y_train[train_idx]
    y_fold_val = y_train[val_idx]
    
    # Inner CV for hyperparameter optimization 
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # TPOT AutoML
    tpot = TPOTClassifier(
        generations=5,              # Number of iterations
        population_size=20,         # Population size
        cv=inner_cv,               # Inner CV
        scoring='roc_auc',         # Optimization metric
        random_state=RANDOM_STATE,
        verbosity=0,               # Suppress output
        n_jobs=-1,                 # Use all CPUs
        max_time_mins=5,           # Max 5 minutes per fold
        max_eval_time_mins=0.5,    # Max 30 seconds per pipeline
        early_stop=3               # Stop if no improvement
    )
    
    print(f"    Training TPOT (max 5 min)...", end=" ", flush=True)
    tpot.fit(X_fold_train, y_fold_train)
    print("✓")
    
    # Predict on validation fold
    y_pred = tpot.predict(X_fold_val)
    y_pred_proba = tpot.predict_proba(X_fold_val)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_fold_val, y_pred)
    roc = roc_auc_score(y_fold_val, y_pred_proba)
    mcc = matthews_corrcoef(y_fold_val, y_pred)
    
    nested_cv_scores['accuracy'].append(acc)
    nested_cv_scores['roc_auc'].append(roc)
    nested_cv_scores['mcc'].append(mcc)
    
    print(f"    Accuracy: {acc:.3f}")
    print(f"    ROC-AUC: {roc:.3f}")
    print(f"    MCC: {mcc:.3f}")
    
    # Store best model from this fold
    best_models.append(tpot.fitted_pipeline_)

# Calculate nested CV statistics
print("\n" + "="*80)
print("NESTED CROSS-VALIDATION RESULTS (5x5):")
print("="*80)

for metric_name, scores in nested_cv_scores.items():
    mean = np.mean(scores)
    std = np.std(scores)
    print(f"{metric_name.upper():12s} = {mean:.3f} ± {std:.3f}")

################################################################################
# 4. FINAL MODEL TRAINING (ENTIRE TRAINING SET)
################################################################################

print("\n[STEP 4] Training final model on entire training set...")

# Train TPOT on entire training set
tpot_final = TPOTClassifier(
    generations=10,                 # More generations for final model
    population_size=50,             # Larger population
    cv=5,                           # 5-fold CV for hyperparameter tuning
    scoring='roc_auc',
    random_state=RANDOM_STATE,
    verbosity=2,                    # Show progress
    n_jobs=-1,
    max_time_mins=15,               # More time for final model
    max_eval_time_mins=1
)

print("\nTraining final TPOT model (max 15 min)...")
tpot_final.fit(X_train, y_train)

print("\n✓ Final model trained")
print(f"  Best pipeline: {tpot_final.fitted_pipeline_}")

# Save model
with open('tpot_final_model.pkl', 'wb') as f:
    pickle.dump(tpot_final.fitted_pipeline_, f)
print("  Model saved: tpot_final_model.pkl")

# Export pipeline code
tpot_final.export('tpot_pipeline.py')
print("  Pipeline code saved: tpot_pipeline.py")

################################################################################
# 5. HOLD-OUT VALIDATION
################################################################################

print("\n[STEP 5] Hold-out validation on test set...")

# Predict on hold-out test set
y_test_pred = tpot_final.predict(X_test)
y_test_proba = tpot_final.predict_proba(X_test)[:, 1]

# Calculate metrics
test_acc = accuracy_score(y_test, y_test_pred)
test_roc = roc_auc_score(y_test, y_test_proba)
test_mcc = matthews_corrcoef(y_test, y_test_pred)

print("\n" + "="*80)
print("HOLD-OUT VALIDATION RESULTS:")
print("="*80)
print(f"Accuracy:  {test_acc:.3f}")
print(f"ROC-AUC:   {test_roc:.3f}")
print(f"MCC:       {test_mcc:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print(f"             Predicted")
print(f"               NC   CC")
print(f"  Actual NC   {cm[0,0]:3d}  {cm[0,1]:3d}")
print(f"         CC   {cm[1,0]:3d}  {cm[1,1]:3d}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, 
                          target_names=['Control', 'CRC'],
                          digits=3))

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

# Bootstrap CI for AUC
from sklearn.utils import resample

bootstrap_aucs = []
n_bootstraps = 2000

print(f"\nCalculating bootstrap CI (n={n_bootstraps})...", end=" ", flush=True)
for i in range(n_bootstraps):
    indices = resample(range(len(y_test)), random_state=i)
    if len(np.unique(y_test[indices])) < 2:
        continue
    score = roc_auc_score(y_test[indices], y_test_proba[indices])
    bootstrap_aucs.append(score)

ci_lower = np.percentile(bootstrap_aucs, 2.5)
ci_upper = np.percentile(bootstrap_aucs, 97.5)
ci_mean = np.mean(bootstrap_aucs)
ci_std = np.std(bootstrap_aucs)

print("✓")
print(f"\nHold-out AUC: {test_roc:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")

################################################################################
# 6. SHAP ANALYSIS
################################################################################

print("\n[STEP 6] SHAP analysis for feature importance...")

# Get the best model (usually RandomForest or similar)
final_model = tpot_final.fitted_pipeline_

# Check if model has a classifier step
if hasattr(final_model, 'steps'):
    classifier = final_model.steps[-1][1]
else:
    classifier = final_model

# Create SHAP explainer
print("  Creating SHAP explainer...", end=" ", flush=True)

try:
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test)
    
    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    print("✓")
    
    # SHAP summary plot
    print("  Generating SHAP plots...")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, 
                      feature_names=feature_names,
                      show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ SHAP summary plot saved: shap_summary_plot.png")
    
    # Feature importance
    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    importance_df.to_csv('feature_importance_shap.csv', index=False)
    print("    ✓ Feature importance saved: feature_importance_shap.csv")
    
    print("\nTop 10 Features (SHAP importance):")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']:30s} {row['Importance']:.4f}")

except Exception as e:
    print(f"\n  Warning: SHAP analysis failed: {e}")
    print("  This is normal for some pipeline types (e.g., stacking)")

################################################################################
# 7. SAVE RESULTS
################################################################################

print("\n[STEP 7] Saving results...")

# Save all results to a summary file
results_summary = {
    'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'Preprocessing': 'KNN → Log2 → Pareto (NO PQN)',
    'n_samples': len(X),
    'n_features': len(feature_names),
    'n_train': len(X_train),
    'n_test': len(X_test),
    
    'Nested_CV_Accuracy_mean': np.mean(nested_cv_scores['accuracy']),
    'Nested_CV_Accuracy_std': np.std(nested_cv_scores['accuracy']),
    'Nested_CV_AUC_mean': np.mean(nested_cv_scores['roc_auc']),
    'Nested_CV_AUC_std': np.std(nested_cv_scores['roc_auc']),
    'Nested_CV_MCC_mean': np.mean(nested_cv_scores['mcc']),
    'Nested_CV_MCC_std': np.std(nested_cv_scores['mcc']),
    
    'Holdout_Accuracy': test_acc,
    'Holdout_AUC': test_roc,
    'Holdout_AUC_CI_lower': ci_lower,
    'Holdout_AUC_CI_upper': ci_upper,
    'Holdout_MCC': test_mcc,
    
    'Best_Pipeline': str(tpot_final.fitted_pipeline_)
}

results_df = pd.DataFrame([results_summary])
results_df.to_csv('ml_analysis_results.csv', index=False)
print("  ✓ Results saved: ml_analysis_results.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_test_pred,
    'Predicted_Probability': y_test_proba
})
predictions_df.to_csv('test_predictions.csv', index=False)
print("  ✓ Predictions saved: test_predictions.csv")

################################################################################
# 8. FINAL SUMMARY
################################################################################

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print("\nPREPROCESSING:")
print("  KNN imputation → Log2 transformation → Pareto scaling")
print("  NO PQN normalization (removed after reviewer comments)")

print("\nNESTED CROSS-VALIDATION (5x5):")
print(f"  Accuracy:  {np.mean(nested_cv_scores['accuracy']):.3f} ± {np.std(nested_cv_scores['accuracy']):.3f}")
print(f"  ROC-AUC:   {np.mean(nested_cv_scores['roc_auc']):.3f} ± {np.std(nested_cv_scores['roc_auc']):.3f}")
print(f"  MCC:       {np.mean(nested_cv_scores['mcc']):.3f} ± {np.std(nested_cv_scores['mcc']):.3f}")

print("\nHOLD-OUT VALIDATION:")
print(f"  Accuracy:  {test_acc:.3f}")
print(f"  ROC-AUC:   {test_roc:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
print(f"  MCC:       {test_mcc:.3f}")

print("\nOUTPUT FILES:")
print("  - tpot_final_model.pkl (trained model)")
print("  - tpot_pipeline.py (pipeline code)")
print("  - ml_analysis_results.csv (all metrics)")
print("  - test_predictions.csv (test set predictions)")
print("  - shap_summary_plot.png (feature importance)")
print("  - feature_importance_shap.csv (SHAP values)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nFor manuscript reporting, use:")
print(f"  - Nested CV AUC: {np.mean(nested_cv_scores['roc_auc']):.3f} ± {np.std(nested_cv_scores['roc_auc']):.3f}")
print(f"  - Hold-out AUC: {test_roc:.3f} ± {ci_std:.3f}")
print("\nIMPORTANT: Emphasize nested CV as primary result (more robust)")
print("           Hold-out is corroboration only")
print("="*80 + "\n")

