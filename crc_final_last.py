
DATA_PATH       = 'df.xlsx'
OUTPUT_DIR      = 'crc_results,_son'
RANDOM_SEED     = 42
N_REPEATS       = 100     # Tablo 2 ve Tablo 3 için repeat sayısı
N_FOLDS         = 5       # Nested CV fold sayısı
TEST_SIZE       = 0.20    # Hold-out test oranı
TPOT_MAX_MINS   = 5       # TPOT her repeat için maksimum süre (dakika)
OPTUNA_N_TRIALS = 50      # LightGBM ve XGBoost için Optuna trial sayısı
# ════════════════════════════════════════════════════════════

import os, sys, json, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.makedirs(OUTPUT_DIR, exist_ok=True)


try:
    import pkg_resources
except ImportError:
    try:
        import setuptools
        import pkg_resources
    except Exception:
        pass

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from scipy import stats
from scipy.stats import mannwhitneyu

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                              recall_score, f1_score, brier_score_loss,
                              matthews_corrcoef, roc_curve)
from sklearn.impute import KNNImputer

from statsmodels.stats.multitest import multipletests
import lightgbm as lgb
import xgboost as xgb

HOLDOUT_CKPT = os.path.join(OUTPUT_DIR, 'holdout_checkpoint.json')
NESTED_CKPT  = os.path.join(OUTPUT_DIR, 'nested_checkpoint.json')


# ─────────────────────────────────────────────────────────────
# 1. PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess_fold(X_train, X_test, seed=42):
    
    imp  = KNNImputer(n_neighbors=5)
    X_tr = imp.fit_transform(X_train)
    X_te = imp.transform(X_test)

    X_tr = np.log2(X_tr + 1e-9)
    X_te = np.log2(X_te + 1e-9)

    mean  = X_tr.mean(axis=0)
    std   = X_tr.std(axis=0)
    std[std < 1e-10] = 1e-10
    scale = np.sqrt(std)   # Pareto scaling: σ^(1/2)

    X_tr_scaled = (X_tr - mean) / scale
    X_te_scaled = (X_te - mean) / scale

    return X_tr_scaled, X_te_scaled


def preprocess_full(X):
    
    imp  = KNNImputer(n_neighbors=5)
    X_i  = imp.fit_transform(X)
    X_l  = np.log2(X_i + 1e-9)
    mean = X_l.mean(axis=0)
    std  = X_l.std(axis=0)
    std[std < 1e-10] = 1e-10
    return (X_l - mean) / np.sqrt(std)


# ─────────────────────────────────────────────────────────────
# 2. METRİKLER
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    return dict(
        AUC         = float(roc_auc_score(y_true, y_prob)),
        Accuracy    = float(accuracy_score(y_true, y_pred)),
        Precision   = float(precision_score(y_true, y_pred, zero_division=0)),
        Sensitivity = float(recall_score(y_true, y_pred, zero_division=0)),
        F1          = float(f1_score(y_true, y_pred, zero_division=0)),
        Brier       = float(brier_score_loss(y_true, y_prob)),
        MCC         = float(matthews_corrcoef(y_true, y_pred)),
    )


def bootstrap_auc_ci(y_true, y_prob, n_bootstraps=2000, alpha=0.05, seed=42):
    """Bootstrap 95% CI for ROC AUC."""
    rng = np.random.RandomState(seed)
    boot_aucs = []
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    try:
        base_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        return float('nan'), float('nan'), float('nan')
    for _ in range(n_bootstraps):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except ValueError:
            pass
    if len(boot_aucs) > 10:
        ci_lower = float(np.percentile(boot_aucs, 100 * (alpha / 2)))
        ci_upper = float(np.percentile(boot_aucs, 100 * (1 - alpha / 2)))
    else:
        ci_lower = ci_upper = float('nan')
    return base_auc, ci_lower, ci_upper


def youden_sens_spec(y_true, y_prob):
    """Youden J indeksi ile optimal eşik → sensitivity, specificity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = int(np.argmax(tpr - fpr))
    y_o = (y_prob >= thresholds[idx]).astype(int)
    return (float(recall_score(y_true, y_o, zero_division=0)),
            float(1.0 - fpr[idx]))


# ─────────────────────────────────────────────────────────────
# 3. TABLO 1 — tek değişkenli analiz
# ─────────────────────────────────────────────────────────────

def run_table1(X_raw, y, feature_names):
    print("\n[TABLO 1] Tek değişkenli analiz hesaplanıyor...")

    imp = KNNImputer(n_neighbors=5)
    X_imputed = imp.fit_transform(X_raw)
    X_scaled  = preprocess_full(X_raw)

    rows = []
    for i, feat in enumerate(feature_names):
        a_s, b_s = X_scaled[y == 1, i], X_scaled[y == 0, i]
        _, p_mw  = mannwhitneyu(a_s, b_s, alternative='two-sided')

        crc_raw  = X_imputed[y == 1, i]
        ctrl_raw = X_imputed[y == 0, i]
        med_crc  = float(np.median(crc_raw))
        med_ctrl = float(np.median(ctrl_raw))

        if med_ctrl == 0:
            fc     = float('inf') if med_crc > 0 else 1.0
            log2fc = float('inf') if med_crc > 0 else 0.0
        else:
            fc     = med_crc / med_ctrl
            log2fc = float(np.log2(fc + 1e-12))

        reg    = 'Up' if fc > 1 else 'Down'
        log10p = float(-np.log10(p_mw + 1e-300))

        prob = X_imputed[:, i]
        try:
            auc_v, ci_lo, ci_hi = bootstrap_auc_ci(y, prob)
            sens, spec = youden_sens_spec(y, prob)
        except Exception:
            auc_v = ci_lo = ci_hi = sens = spec = float('nan')

        rows.append({
            'Metabolite': feat,
            'FC'        : round(fc, 3),
            'Log2FC'    : round(log2fc, 3),
            '-log10(p)' : round(log10p, 3),
            'p_raw'     : float(p_mw),
            'Regulation': reg,
            'AUC'       : round(auc_v, 3),
            'CI_lo'     : round(ci_lo, 3),
            'CI_hi'     : round(ci_hi, 3),
            'Sensitivity': round(sens, 3),
            'Specificity': round(spec, 3),
        })

    df = pd.DataFrame(rows)
    _, p_adj, _, _ = multipletests(df['p_raw'].values, method='fdr_bh')
    df['Adj_p']     = p_adj
    df['Adj_p_fmt'] = df['Adj_p'].apply(
        lambda x: '<0.001' if x < 0.001 else f'{x:.3f}')
    df = df.sort_values('-log10(p)', ascending=False).reset_index(drop=True)
    df['CI_95'] = df.apply(
        lambda r: f"{r['CI_lo']:.3f}–{r['CI_hi']:.3f}" if not np.isnan(r['CI_lo']) else 'nan', axis=1)

    out = df[['Metabolite','FC','Log2FC','-log10(p)','Adj_p_fmt',
              'Regulation','AUC','CI_95','Sensitivity','Specificity']].copy()
    out.columns = ['Metabolite','FC','Log2FC','-log10(p)','Adj_p',
                   'Regulation','AUC','CI_95','Sensitivity','Specificity']

    path = os.path.join(OUTPUT_DIR, 'Table1_Univariate.csv')
    out.to_csv(path, index=False)
    print(f"  → {path}")
    print(out[['Metabolite','FC','Log2FC','AUC','CI_95','Adj_p']].head(6).to_string(index=False))
    return out


# ─────────────────────────────────────────────────────────────
# 4. TPOT 
# ─────────────────────────────────────────────────────────────

def run_tpot(X_tr, y_tr, X_te, seed):
    
    try:
        try:
            import pkg_resources
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, '-m', 'pip', 'install',
                            'setuptools', '--upgrade', '-q'], check=False)
            import importlib
            import pkg_resources

        from tpot import TPOTClassifier
        tpot = TPOTClassifier(
            scorers         = ['roc_auc'],
            scorers_weights = [1.0],
            max_time_mins   = TPOT_MAX_MINS,
            random_state    = seed,
            verbose         = 0,
            n_jobs          = 1,
        )
        tpot.fit(X_tr, y_tr)
        y_prob = tpot.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        try:
            pipe_name = type(tpot.fitted_pipeline_.steps[-1][1]).__name__
        except Exception:
            pipe_name = 'TPOT'
        try:
            pipe_path = os.path.join(OUTPUT_DIR, f'tpot_pipeline_seed{seed}.py')
            tpot.export(pipe_path)
        except Exception:
            pass  

        return y_pred, y_prob, pipe_name, tpot.fitted_pipeline_

    except Exception as e:
        print(f"    [TPOT HATA] {e}")
        raise


# ─────────────────────────────────────────────────────────────
# 5. OPTUNA 
# ─────────────────────────────────────────────────────────────

def optuna_lgbm(X_tr, y_tr, seed):
    
    def objective(trial):
        p = dict(
            num_leaves        = trial.suggest_categorical('nl', [7, 15, 31]),
            learning_rate     = trial.suggest_categorical('lr', [0.01, 0.05, 0.1]),
            n_estimators      = trial.suggest_int('ne', 100, 500, step=100),
            max_depth         = trial.suggest_int('md', 2, 5),
            min_child_samples = trial.suggest_int('mc', 20, 60, step=10),
            subsample         = trial.suggest_float('ss', 0.4, 0.7),
            colsample_bytree  = trial.suggest_float('cs', 0.3, 0.6),
            reg_alpha         = trial.suggest_float('ra', 0.1, 20.0, log=True),
            reg_lambda        = trial.suggest_float('rl', 0.1, 20.0, log=True),
            min_split_gain    = trial.suggest_float('msg', 0.0, 1.0),
            random_state=seed, verbose=-1, n_jobs=1, subsample_freq=1,
        )
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        sc  = []
        for ti, vi in skf.split(X_tr, y_tr):
            m = lgb.LGBMClassifier(**p)
            m.fit(X_tr[ti], y_tr[ti])
            sc.append(roc_auc_score(y_tr[vi], m.predict_proba(X_tr[vi])[:, 1]))
        return float(np.mean(sc))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)
    b = study.best_params
    return lgb.LGBMClassifier(
        num_leaves=b['nl'], learning_rate=b['lr'], n_estimators=b['ne'],
        max_depth=b['md'], min_child_samples=b['mc'], subsample=b['ss'],
        colsample_bytree=b['cs'], reg_alpha=b['ra'], reg_lambda=b['rl'],
        min_split_gain=b['msg'],
        random_state=seed, verbose=-1, n_jobs=-1, subsample_freq=1,
    )


# ─────────────────────────────────────────────────────────────
# 6. OPTUNA 
# ─────────────────────────────────────────────────────────────

def optuna_xgb(X_tr, y_tr, seed):
    
    def objective(trial):
        p = dict(
            max_depth        = trial.suggest_int('md', 2, 8),
            learning_rate    = trial.suggest_float('lr', 0.01, 0.3, log=True),
            n_estimators     = trial.suggest_int('ne', 100, 600, step=100),
            subsample        = trial.suggest_float('ss', 0.5, 1.0),
            colsample_bytree = trial.suggest_float('cs', 0.5, 1.0),
            reg_alpha        = trial.suggest_float('ra', 1e-3, 2.0, log=True),
            reg_lambda       = trial.suggest_float('rl', 1e-3, 2.0, log=True),
            min_child_weight = trial.suggest_int('mcw', 1, 5),
            gamma            = trial.suggest_float('g', 0.0, 1.0),
            random_state=seed, verbosity=0, n_jobs=1,
        )
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        sc  = []
        for ti, vi in skf.split(X_tr, y_tr):
            m = xgb.XGBClassifier(**p)
            m.fit(X_tr[ti], y_tr[ti])
            sc.append(roc_auc_score(y_tr[vi], m.predict_proba(X_tr[vi])[:, 1]))
        return float(np.mean(sc))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)
    b = study.best_params
    return xgb.XGBClassifier(
        max_depth=b['md'], learning_rate=b['lr'], n_estimators=b['ne'],
        subsample=b['ss'], colsample_bytree=b['cs'],
        reg_alpha=b['ra'], reg_lambda=b['rl'],
        min_child_weight=b['mcw'], gamma=b['g'],
        random_state=seed, verbosity=0, n_jobs=-1,
    )


# ─────────────────────────────────────────────────────────────
# 7. TABLO 2 — 100-repeat hold-out
# ─────────────────────────────────────────────────────────────

def run_table2_holdout(X_raw, y):
    print(f"\n[TABLO 2] {N_REPEATS}-repeat hold-out validation başlıyor...")

    if os.path.exists(HOLDOUT_CKPT):
        with open(HOLDOUT_CKPT) as f:
            ckpt = json.load(f)
        done    = ckpt['done']
        results = ckpt['results']
        pipes   = ckpt['pipes']
        print(f"  Checkpoint bulundu: {done}/{N_REPEATS} repeat tamamlanmış, devam ediliyor...")
    else:
        done    = 0
        results = {'TPOT': [], 'LightGBM': [], 'XGBoost': []}
        pipes   = []

    sss        = StratifiedShuffleSplit(n_splits=N_REPEATS, test_size=TEST_SIZE,
                                        random_state=RANDOM_SEED)
    all_splits = list(sss.split(X_raw, y))
    last_pipe  = None

    for rep in range(done, N_REPEATS):
        t0   = time.time()
        seed = RANDOM_SEED + rep
        tr_idx, te_idx = all_splits[rep]
        X_tr, X_te = preprocess_fold(X_raw[tr_idx], X_raw[te_idx], seed=seed)
        y_tr, y_te = y[tr_idx], y[te_idx]

        # TPOT
        y_pred_t, y_prob_t, pipe_name, fitted = run_tpot(X_tr, y_tr, X_te, seed)
        results['TPOT'].append(compute_metrics(y_te, y_pred_t, y_prob_t))
        pipes.append(pipe_name)
        last_pipe = fitted

        # LightGBM
        lgbm = optuna_lgbm(X_tr, y_tr, seed)
        lgbm.fit(X_tr, y_tr)
        y_prob_l = lgbm.predict_proba(X_te)[:, 1]
        results['LightGBM'].append(
            compute_metrics(y_te, (y_prob_l >= 0.5).astype(int), y_prob_l))

        # XGBoost
        xgb_m = optuna_xgb(X_tr, y_tr, seed)
        xgb_m.fit(X_tr, y_tr)
        y_prob_x = xgb_m.predict_proba(X_te)[:, 1]
        results['XGBoost'].append(
            compute_metrics(y_te, (y_prob_x >= 0.5).astype(int), y_prob_x))

        done = rep + 1
        with open(HOLDOUT_CKPT, 'w') as f:
            json.dump({'done': done, 'results': results, 'pipes': pipes}, f)

        elapsed = time.time() - t0
        t_auc = results['TPOT'][-1]['AUC']
        l_auc = results['LightGBM'][-1]['AUC']
        x_auc = results['XGBoost'][-1]['AUC']
        print(f"  Rep {done:3d}/{N_REPEATS} | TPOT({pipe_name})={t_auc:.3f} "
              f"| LGB={l_auc:.3f} | XGB={x_auc:.3f} | {elapsed/60:.1f}dk")

    metric_keys = ['AUC','Accuracy','Precision','Sensitivity','F1','Brier','MCC']
    rows = []
    for model in ['TPOT', 'LightGBM', 'XGBoost']:
        df_r = pd.DataFrame(results[model])
        row  = {'Model': model}
        for k in metric_keys:
            m = df_r[k].mean()
            s = df_r[k].std()
            row[k] = f'{m:.3f} ± {s:.3f}'
        rows.append(row)

    df_out = pd.DataFrame(rows).set_index('Model')
    df_out.to_csv(os.path.join(OUTPUT_DIR, 'Table2_Holdout.csv'))

    from collections import Counter
    pipe_freq = Counter(pipes)
    pd.Series(pipe_freq, name='count').sort_values(ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, 'TPOT_pipeline_frequencies.csv'))

    print(f"  → Table2_Holdout.csv")
    return df_out, results, last_pipe


# ─────────────────────────────────────────────────────────────
# 8. TABLO 3 — 100×5 nested CV
# ─────────────────────────────────────────────────────────────

def run_table3_nested_cv(X_raw, y):
    print(f"\n[TABLO 3] {N_REPEATS}×{N_FOLDS} nested CV başlıyor...")

    if os.path.exists(NESTED_CKPT):
        with open(NESTED_CKPT) as f:
            ckpt = json.load(f)
        done   = ckpt['done']
        nested = ckpt['nested']
        print(f"  Checkpoint: {done}/{N_REPEATS} repeat tamamlanmış, devam ediliyor...")
    else:
        done   = 0
        nested = {'TPOT': [], 'LightGBM': [], 'XGBoost': []}

    for rep in range(done, N_REPEATS):
        seed     = RANDOM_SEED + rep
        skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        fold_res = {'TPOT': [], 'LightGBM': [], 'XGBoost': []}

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X_raw, y)):
            fseed = seed * 100 + fold
            X_tr, X_te = preprocess_fold(X_raw[tr_idx], X_raw[te_idx], seed=fseed)
            y_tr, y_te = y[tr_idx], y[te_idx]

            y_pred_t, y_prob_t, _, _ = run_tpot(X_tr, y_tr, X_te, fseed)
            fold_res['TPOT'].append(compute_metrics(y_te, y_pred_t, y_prob_t))

            lgbm = optuna_lgbm(X_tr, y_tr, fseed)
            lgbm.fit(X_tr, y_tr)
            y_prob_l = lgbm.predict_proba(X_te)[:, 1]
            fold_res['LightGBM'].append(
                compute_metrics(y_te, (y_prob_l >= 0.5).astype(int), y_prob_l))

            xgb_m = optuna_xgb(X_tr, y_tr, fseed)
            xgb_m.fit(X_tr, y_tr)
            y_prob_x = xgb_m.predict_proba(X_te)[:, 1]
            fold_res['XGBoost'].append(
                compute_metrics(y_te, (y_prob_x >= 0.5).astype(int), y_prob_x))

        for model in ['TPOT', 'LightGBM', 'XGBoost']:
            nested[model].append(
                pd.DataFrame(fold_res[model]).mean().to_dict())

        done = rep + 1
        with open(NESTED_CKPT, 'w') as f:
            json.dump({'done': done, 'nested': nested}, f)

        if done % 10 == 0 or done <= 3:
            t_auc = nested['TPOT'][-1]['AUC']
            l_auc = nested['LightGBM'][-1]['AUC']
            x_auc = nested['XGBoost'][-1]['AUC']
            print(f"  Rep {done:3d}/{N_REPEATS} | TPOT={t_auc:.3f} | LGB={l_auc:.3f} | XGB={x_auc:.3f}")

    metric_keys = ['AUC','Accuracy','Sensitivity','F1','MCC']
    rows = []
    for model in ['TPOT', 'LightGBM', 'XGBoost']:
        df_r = pd.DataFrame(nested[model])
        row  = {'Model': model}
        for k in metric_keys:
            m = df_r[k].mean()
            s = df_r[k].std()
            row[k] = f'{m:.3f} ± {s:.3f}'
        rows.append(row)

    df_out = pd.DataFrame(rows).set_index('Model')
    df_out.to_csv(os.path.join(OUTPUT_DIR, 'Table3_NestedCV.csv'))
    print(f"  → Table3_NestedCV.csv")
    return df_out


# ─────────────────────────────────────────────────────────────
# 9. SHAP 
# ─────────────────────────────────────────────────────────────

def run_shap(X_raw, y, feature_names, last_pipe):
    print("\n[SHAP] Analiz başlıyor (ölçeklendirilmiş veri kullanılıyor)...")
    try:
        import shap

        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                     random_state=RANDOM_SEED)
        tr_idx, te_idx = next(sss.split(X_raw, y))
        X_tr, X_te = preprocess_fold(X_raw[tr_idx], X_raw[te_idx])
        y_tr = y[tr_idx]
        y_te = y[te_idx]
        print(f"  Train: {len(y_tr)} örnek | Test: {len(y_te)} örnek")

        if last_pipe is None:
            _, _, _, last_pipe = run_tpot(X_tr, y_tr, X_te, RANDOM_SEED)

        # Model SADECE train seti üzerinde fit
        last_pipe.fit(X_tr, y_tr)

        try:
            model = last_pipe.steps[-1][1]
        except Exception:
            model = last_pipe

        # SHAP değerleri TEST seti üzerinde hesapla
        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_te)
        except Exception:
            bg = shap.kmeans(X_tr, min(10, len(X_tr)))
            def pred_fn(x): return last_pipe.predict_proba(x)[:, 1]
            explainer   = shap.KernelExplainer(pred_fn, bg)
            shap_values = explainer.shap_values(X_te, nsamples=300)

        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        
        mean_abs = np.abs(sv).mean(axis=0)
        n_top    = min(15, len(feature_names))
        top_idx  = np.argsort(mean_abs)[::-1][:n_top]

        sv_top    = sv[:, top_idx]
        X_top     = X_te[:, top_idx]
        names_top = [feature_names[i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(11, 7))
        plt.rcParams.update({'font.family': 'DejaVu Sans'})

        # Custom colormap: blue → red (low → high feature value)
        cmap = LinearSegmentedColormap.from_list(
            'shap_cmap', ['#3182bd', '#9ecae1', '#fee0d2', '#de2d26'])

        # Normalize feature values per feature (0–1)
        X_norm = np.zeros_like(X_top)
        for j in range(X_top.shape[1]):
            col = X_top[:, j]
            mn, mx = col.min(), col.max()
            X_norm[:, j] = (col - mn) / (mx - mn + 1e-12)

        y_positions = np.arange(n_top)[::-1]

        for j, (yp, sv_col, xn_col) in enumerate(
                zip(y_positions, sv_top.T, X_norm.T)):
            # Jitter for beeswarm
            rng   = np.random.RandomState(j)
            jitter = rng.uniform(-0.2, 0.2, len(sv_col))
            scatter = ax.scatter(
                sv_col, yp + jitter,
                c=xn_col, cmap=cmap,
                vmin=0, vmax=1,
                s=20, alpha=0.75, linewidths=0,
                zorder=3
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(names_top, fontsize=11)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--', zorder=1)
        ax.set_xlabel('SHAP value (impact on model output)', fontsize=12)


        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
        cbar.set_label('Feature value\n(low → high)', fontsize=9)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Low', 'Mid', 'High'], fontsize=8)

        # Grid & style
        ax.grid(axis='x', linestyle=':', linewidth=0.5, alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(-0.6, n_top - 0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'SHAP_beeswarm.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  → SHAP_beeswarm.png")



    except Exception as e:
        print(f"  [UYARI] SHAP atlandı: {e}")
        import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# 10. LIME — 3 Gerçek Pozitif Hasta Örneği
# ─────────────────────────────────────────────────────────────

def run_lime(X_raw, y, feature_names, last_pipe):
    print("\n[LIME] 3 gerçek-pozitif hasta için analiz başlıyor...")
    try:
        import lime.lime_tabular

        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                     random_state=RANDOM_SEED)
        tr_idx, te_idx = next(sss.split(X_raw, y))
        X_tr, X_te = preprocess_fold(X_raw[tr_idx], X_raw[te_idx])
        y_tr       = y[tr_idx]
        y_te       = y[te_idx]

        if last_pipe is None:
            _, _, _, last_pipe = run_tpot(X_tr, y_tr, X_te, RANDOM_SEED)

        def predict_fn(x):
            return last_pipe.predict_proba(x)

        # discretize_continuous=True → kural tabanlı açıklamalar
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data         = X_tr,
            feature_names         = feature_names,
            class_names           = ['neg', 'pos'],
            mode                  = 'classification',
            random_state          = RANDOM_SEED,
            discretize_continuous = True,
        )

        probs       = last_pipe.predict_proba(X_te)[:, 1]
        preds       = (probs >= 0.5).astype(int)
        crc_correct = np.where((preds == 1) & (y_te == 1))[0]
        n_show      = min(3, len(crc_correct))

        if n_show == 0:
            print("  [UYARI] Yeterli doğru CRC tahmini bulunamadı.")
            return

        plt.rcParams.update({'font.family': 'DejaVu Sans'})
        fig = plt.figure(figsize=(22, 6 * n_show), facecolor='white')
        outer_gs = gridspec.GridSpec(n_show, 1, hspace=0.6)

        colors_pos = '#FF8C00'   # turuncu — CRC (pos)
        colors_neg = '#4169E1'   # mavi — Control (neg)

        for plot_idx, sample_idx in enumerate(crc_correct[:n_show]):
            exp = explainer.explain_instance(
                data_row   = X_te[sample_idx],
                predict_fn = predict_fn,
                num_features = len(feature_names),
                num_samples  = 5000,
                labels       = (1,),
            )
            lime_vals = exp.as_list(label=1)
            p_pos = float(probs[sample_idx])
            p_neg = 1.0 - p_pos
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                1, 3, subplot_spec=outer_gs[plot_idx],
                width_ratios=[1.0, 5.5, 2.0], wspace=0.4
            )

            # ── Panel A: Prediction probabilities ──────────────
            ax_prob = fig.add_subplot(inner_gs[0])
            ax_prob.barh([0, 1], [p_neg, p_pos],
                         color=[colors_neg, colors_pos],
                         height=0.45, edgecolor='white', linewidth=1.5)
            ax_prob.set_yticks([0, 1])
            ax_prob.set_yticklabels(['neg', 'pos'], fontsize=11, fontweight='bold')
            ax_prob.set_xlim(0, 1.25)
            ax_prob.set_title('Prediction\nprobabilities',
                              fontsize=10, fontweight='bold', pad=6)
            for bar_obj, val, clr in zip(ax_prob.patches,
                                         [p_neg, p_pos],
                                         [colors_neg, colors_pos]):
                ax_prob.text(val + 0.03,
                             bar_obj.get_y() + bar_obj.get_height() / 2,
                             f'{val:.2f}', va='center', ha='left',
                             fontsize=11, fontweight='bold', color=clr)
            ax_prob.axvline(0, color='gray', linewidth=0.5)
            ax_prob.spines['top'].set_visible(False)
            ax_prob.spines['right'].set_visible(False)
            ax_prob.spines['left'].set_visible(False)
            ax_prob.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax_prob.tick_params(axis='x', labelsize=8)

            # ── Panel B: Feature contributions ─────────────────
            ax_bar = fig.add_subplot(inner_gs[1])
            rules   = [v[0] for v in lime_vals]
            weights = [v[1] for v in lime_vals]
            nonzero = [(r, w) for r, w in zip(rules, weights) if abs(w) > 1e-6]
            if not nonzero:
                nonzero = list(zip(rules[:10], weights[:10]))
            r_sorted = [v[0] for v in sorted(nonzero, key=lambda x: x[1])]
            w_sorted = [v[1] for v in sorted(nonzero, key=lambda x: x[1])]
            bar_colors = [colors_pos if w > 0 else colors_neg for w in w_sorted]
            y_idx = np.arange(len(r_sorted))
            ax_bar.barh(y_idx, w_sorted, color=bar_colors,
                        height=0.55, edgecolor='white', linewidth=1)
            ax_bar.set_yticks(y_idx)
            ax_bar.set_yticklabels([r[:38] for r in r_sorted], fontsize=8)
            ax_bar.axvline(0, color='black', linewidth=0.8)
            ax_bar.set_title('Feature contributions', fontsize=10, fontweight='bold', pad=8)
            ax_bar.set_xlabel('LIME weight', fontsize=9)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.5)
            ax_bar.set_facecolor('#f8f8f8')
            # ── Panel C: Feature values table ───────────────────
            ax_table = fig.add_subplot(inner_gs[2])
            ax_table.axis('off')
            ax_table.set_title('Feature Values\n(Pareto-scaled)',
                               fontsize=10, fontweight='bold', pad=8)

            used_feats = []
            for rule in rules:
                for f in feature_names:
                    if f in rule and f not in used_feats:
                        used_feats.append(f)
                        break
            for f in feature_names:
                if f not in used_feats:
                    used_feats.append(f)
            used_feats = used_feats[:10]

            table_data  = []
            cell_colors = []
            for f in used_feats:
                f_idx   = feature_names.index(f)
                val_str = f"{X_te[sample_idx, f_idx]:.3f}"
                table_data.append([f, val_str])
                matched_color = '#f0f0f0'
                for rule_str, w in lime_vals:
                    if f in rule_str:
                        matched_color = '#FFD59A' if w > 0 else '#A8C4E0'
                        break
                cell_colors.append([matched_color, matched_color])

            tbl = ax_table.table(
                cellText  = table_data,
                colLabels = ['Feature', 'Value'],
                loc       = 'center',
                cellLoc   = 'left',
                colWidths = [0.62, 0.38]
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.5)
            for (ri, ci), cell in tbl.get_celld().items():
                cell.set_height(0.085)
                cell.set_linewidth(0.3)
                if ri == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#2c3e50')
                else:
                    cell.set_facecolor(cell_colors[ri - 1][ci])
                    

        path_lime = os.path.join(OUTPUT_DIR, 'LIME_explanations.png')
        fig.savefig(path_lime, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  → LIME_explanations.png  ({n_show} hasta)")

    except Exception as e:
        print(f"  [UYARI] LIME atlandı: {e}")
        import traceback; traceback.print_exc()








# ─────────────────────────────────────────────────────────────
# 12. EXCEL ÇIKTI — 3 Tablo
# ─────────────────────────────────────────────────────────────

def export_excel(t1, t2, t3):
    path = os.path.join(OUTPUT_DIR, 'All_Tables.xlsx')
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        t1.to_excel(w, sheet_name='Table1_Univariate', index=False)
        t2.to_excel(w, sheet_name='Table2_Holdout')
        t3.to_excel(w, sheet_name='Table3_NestedCV')
    print(f"\n[EXCEL] → All_Tables.xlsx (3 tablo)")


# ─────────────────────────────────────────────────────────────
# 13. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 65)
    print("CRC Fecal EV Metabolomics ")
    print(f"Repeats     : {N_REPEATS}")
    print(f"Folds       : {N_FOLDS}")
    print(f"Test size   : {TEST_SIZE}")
    print(f"TPOT        : max {TPOT_MAX_MINS} dk/repeat")
    print(f"Optuna      : {OPTUNA_N_TRIALS} trials/model/repeat")
    print(f"Output      : {OUTPUT_DIR}")
    
    print(f"\nVeri yükleniyor: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    df['Group'] = df['Factors'].map({'Group:NC': 0, 'Group:CC': 1})
    df          = df.drop(columns=['Factors'])
    feature_names = [c for c in df.columns if c != 'Group']
    X_raw = df[feature_names].values.astype(float)
    y     = df['Group'].values.astype(int)

    print(f"  NC (Control) : {(y == 0).sum()}")
    print(f"  CC (CRC)     : {(y == 1).sum()}")
    print(f"  Metabolitler : {len(feature_names)}")
    print(f"  Eksik değer  : {int(np.isnan(X_raw).sum())}")

    # ── Tablo 1
    t1 = run_table1(X_raw, y, feature_names)

    # ── Tablo 2
    t2, holdout_raw, last_pipe = run_table2_holdout(X_raw, y)

    # ── Tablo 3
    t3 = run_table3_nested_cv(X_raw, y)

    # ── SHAP
    run_shap(X_raw, y, feature_names, last_pipe)

    # ── LIME
    run_lime(X_raw, y, feature_names, last_pipe)

    

    # ── Excel
    export_excel(t1, t2, t3)

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'=' * 65}")
    print(f"TAMAMLANDI — Toplam süre: {elapsed:.1f} dakika")
    print(f"Çıktılar   : {OUTPUT_DIR}")
    print(f"{'=' * 65}")
    print("\n── TABLO 2 (Hold-out) ──")
    print(t2.to_string())
    print("\n── TABLO 3 (Nested CV) ──")
    print(t3.to_string())


if __name__ == '__main__':
    main()
    
    
    
