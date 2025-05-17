import pandas as pd
import numpy as np
try:
    import lightgbm as lgb
except OSError as e:
    raise RuntimeError(
        "LightGBM library failed to load. On macOS this usually means the libomp\n"
        "dependency is missing. Install it via 'brew install libomp' or reinstall\n"
        "LightGBM with OpenMP support."
    ) from e
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from utils import weighted_roc_auc
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_OHE_CATEGORIES = 20


def load_feature_lists(path='feature_description.xlsx'):
    desc = pd.read_excel(path)
    num_cols = desc[desc['Тип'] == 'number']['Признак'].tolist()
    cat_cols = desc[desc['Тип'] == 'category']['Признак'].tolist()
    return num_cols, cat_cols


def cap_rare_categories(df, cat_cols):
    for col in cat_cols:
        vc = df[col].value_counts()
        rare = vc[vc < MAX_OHE_CATEGORIES].index
        df[col] = df[col].where(~df[col].isin(rare), '__OTHER__')
    return df


def impute_start_cluster(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fill missing start_cluster for month 6 in test using previous months."""
    for df in (train_df, test_df):
        if 'month_num' not in df.columns:
            df['month_num'] = df['date'].str.replace('month_', '').astype(int)

    all_cats = pd.concat([
        train_df['start_cluster'],
        test_df['start_cluster']
    ]).dropna().astype(str).unique()
    dtype = pd.CategoricalDtype(categories=all_cats, ordered=False)
    train_df['start_cluster'] = train_df['start_cluster'].astype(str).astype(dtype)
    test_df['start_cluster'] = test_df['start_cluster'].astype(str).astype(dtype)

    mode_val = train_df['start_cluster'].mode()
    mode_val = mode_val.iloc[0] if not mode_val.empty else (
        all_cats[0] if len(all_cats) else 'unknown'
    )

    pivot = test_df.pivot_table(
        index='id', columns='month_num', values='start_cluster', aggfunc='first'
    )
    idx_m6 = test_df[test_df['month_num'] == 6].index
    if len(idx_m6) > 0:
        ids = test_df.loc[idx_m6, 'id']
        val = pivot.reindex(ids).get(5)
        fallback = pivot.reindex(ids).get(4)
        val = val.fillna(fallback).fillna(mode_val)
        test_df.loc[idx_m6, 'start_cluster'] = val.values.astype(str)

    return train_df, test_df, dtype
  
def _add_numeric_features(full: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """Add lag, diff and rolling statistics for numeric columns."""
    # sort to ensure proper temporal ordering within each id
    full = full.sort_values(['id', 'date'])
    for col in num_cols:
        if col not in full.columns:
            continue
        full[col] = pd.to_numeric(full[col], errors='coerce')
        full[f"{col}_lag1"] = full.groupby('id')[col].shift(1)
        full[f"{col}_diff1"] = full[col].fillna(0) - full[f"{col}_lag1"].fillna(0)
        gb = full.groupby('id')[col]
        full[f"{col}_roll_mean_2"] = gb.transform(lambda x: x.rolling(window=2, min_periods=1).mean())
        full[f"{col}_roll_std_2"] = gb.transform(lambda x: x.rolling(window=2, min_periods=1).std())
        for n in [f"{col}_lag1", f"{col}_diff1", f"{col}_roll_mean_2", f"{col}_roll_std_2"]:
            full[n] = full[n].fillna(0)
    return full


def preprocess(train_df, test_df, num_cols, cat_cols):
    full = pd.concat([train_df, test_df], keys=['train', 'test'])
    full = _add_numeric_features(full, num_cols)
    full = cap_rare_categories(full, cat_cols)
    full = pd.get_dummies(full, columns=cat_cols, dummy_na=False)
    train_proc = full.xs('train').drop(['id', 'date'], axis=1)
    test_proc = full.xs('test').drop(['id', 'date'], axis=1)
    return train_proc, test_proc


def main():
    logger.info('Loading data...')
    print('Loading data...')
    train_df = pd.read_parquet('train_data.pqt')
    test_df = pd.read_parquet('test_data.pqt')
    cluster_weights = pd.read_excel('cluster_weights.xlsx').set_index('cluster')['unnorm_weight'].to_dict()
    submission_template = pd.read_csv('sample_submission.csv')

    num_cols, cat_cols = load_feature_lists()
    logger.info(f'Numeric features: {len(num_cols)}, Categorical features: {len(cat_cols)}')
    print(f'Parsed feature lists. Numeric: {len(num_cols)}; Categorical: {len(cat_cols)}')

    # Ensure start_cluster is filled for month 6 rows in test
    train_df, test_df, _ = impute_start_cluster(train_df, test_df)

    le = LabelEncoder()
    le.fit(train_df['end_cluster'].astype(str))
    train_df['target'] = le.transform(train_df['end_cluster'].astype(str))

    train_proc, test_proc = preprocess(
        train_df[['id', 'date'] + num_cols + cat_cols],
        test_df[['id', 'date'] + num_cols + cat_cols],
        num_cols,
        cat_cols,
    )
    print('Preprocessing finished.')

    X = train_proc.values
    y = train_df['target'].values

    # Use shuffled KFold to reduce variance between folds
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(train_df), len(le.classes_)))
    fold_scores = []
    best_iterations = []

    params = dict(
        objective='multiclass',
        boosting_type='gbdt',
        num_class=len(le.classes_),
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_estimators=1000,
    )

    # Precompute sample weights to better match the evaluation metric
    train_weights = train_df['end_cluster'].astype(str).map(cluster_weights).fillna(1.0).values

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f'Starting fold {fold + 1}')
        print(f'Fold {fold + 1} training...')
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X[tr_idx],
            y[tr_idx],
            sample_weight=train_weights[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            eval_sample_weight=[train_weights[val_idx]],
            eval_metric='multi_logloss',
            early_stopping_rounds=50,
            verbose=False,
        )
        best_iter = model.best_iteration_ or params['n_estimators']
        best_iterations.append(best_iter)
        preds = model.predict_proba(X[val_idx], num_iteration=best_iter)
        score = weighted_roc_auc(y[val_idx], preds, cluster_weights, le)
        logger.info(f'Fold {fold + 1} weighted AUC: {score:.4f} at iter {best_iter}')
        print(f'Fold {fold + 1} AUC: {score:.4f}')
        oof_preds[val_idx] = preds
        fold_scores.append(score)

    logger.info(f'Mean CV weighted AUC: {np.mean(fold_scores):.4f}')
    print(f'CV done. Mean AUC: {np.mean(fold_scores):.4f}')

    logger.info('Training final model...')
    print('Training final model...')
    avg_best_iter = int(np.mean(best_iterations)) if best_iterations else params['n_estimators']
    final_model = lgb.LGBMClassifier(**params, n_estimators=avg_best_iter)
    final_model.fit(X, y, sample_weight=train_weights)

    test_features = test_proc[test_df['date'] == 'month_6'].values
    ids = test_df.loc[test_df['date'] == 'month_6', 'id']
    logger.info(f'Predicting on {len(ids)} test rows...')
    print(f'Predicting on {len(ids)} rows...')
    test_pred = final_model.predict_proba(test_features, num_iteration=avg_best_iter)

    submission = pd.DataFrame(test_pred, columns=le.classes_)
    submission.insert(0, 'id', ids.values)
    submission = submission[submission_template.columns]
    submission.to_csv('submission_lightgbm.csv', index=False)
    logger.info('Submission saved to submission_lightgbm.csv')



if __name__ == '__main__':
    main()
