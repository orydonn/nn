import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

MAX_OHE_CATEGORIES = 30
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_feature_lists(path: str):
    df = pd.read_excel(path)
    num_cols = df[df['Тип'] == 'number']['Признак'].tolist()
    cat_cols = df[df['Тип'] == 'category']['Признак'].tolist()
    return num_cols, cat_cols

def load_cluster_weights(path: str):
    df = pd.read_excel(path)
    return dict(zip(df['cluster'].astype(str), df['unnorm_weight']))

def load_data():
    train = pd.read_parquet('train_data.pqt')
    test = pd.read_parquet('test_data.pqt')
    return train, test

def parse_months(df: pd.DataFrame) -> pd.DataFrame:
    df['month_num'] = df['date'].str.replace('month_', '').astype(int)
    return df

def impute_start_cluster(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mode = train['start_cluster'].mode()
    fill_val = mode.iloc[0] if not mode.empty else 'unknown'
    pivot = test.pivot_table(index='id', columns='month_num', values='start_cluster', aggfunc='first')
    idx = test['month_num'] == 6
    if idx.any():
        ids = test.loc[idx, 'id']
        val = pivot.reindex(ids).get(5)
        fallback = pivot.reindex(ids).get(4)
        val = val.fillna(fallback).fillna(fill_val)
        test.loc[idx, 'start_cluster'] = val.values
    return train, test

def cap_categories(train: pd.DataFrame, test: pd.DataFrame, cat_cols: list[str]):
    for col in cat_cols:
        counts = train[col].astype(str).value_counts()
        rare = counts[counts < MAX_OHE_CATEGORIES].index
        train[col] = train[col].astype(str).apply(lambda x: x if x not in rare else '__OTHER__')
        test[col] = test[col].astype(str).apply(lambda x: x if x not in rare else '__OTHER__')
    return train, test

def prepare_full_dataframe(num_cols, cat_cols):
    train, test = load_data()
    train = parse_months(train)
    test = parse_months(test)
    train, test = impute_start_cluster(train, test)
    cat_cols_full = list(cat_cols) + ['start_cluster']
    train, test = cap_categories(train, test, cat_cols_full)
    train['is_train'] = 1
    test['is_train'] = 0
    full = pd.concat([train, test], ignore_index=True)
    # ensure types
    full[num_cols] = full[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    for col in cat_cols_full:
        full[col] = full[col].astype(str)
    full = pd.get_dummies(full, columns=cat_cols_full, dummy_na=False)
    return full

def weighted_roc_auc(y_true: np.ndarray, y_pred: np.ndarray, weights: dict, le: LabelEncoder) -> float:
    labels = np.unique(y_true)
    if labels.size == 0:
        return 0.0
    aucs = roc_auc_score(y_true, y_pred, multi_class='ovr', average=None, labels=labels)
    labels_str = le.inverse_transform(labels)
    w = np.array([weights.get(str(l), 0) for l in labels_str], dtype=float)
    if w.sum() == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    m = min(len(aucs), len(w))
    return float(np.sum(aucs[:m] * w[:m]))

def train_lightgbm_cv(full_df: pd.DataFrame, weights: dict, n_estimators: int = 200):
    train_df = full_df[full_df['is_train'] == 1].copy()
    test_df = full_df[(full_df['is_train'] == 0) & (full_df['month_num'] == 6)].copy()
    le = LabelEncoder()
    y = le.fit_transform(train_df['end_cluster'].astype(str))
    X = train_df.drop(columns=['id', 'date', 'end_cluster', 'is_train'])
    X_test = test_df.drop(columns=['id', 'date', 'end_cluster', 'is_train'])
    ids_test = test_df['id'].values
    months = sorted(train_df['month_num'].unique())
    num_classes = len(le.classes_)
    preds_test = np.zeros((len(ids_test), num_classes))
    scores = []
    for val_month in months:
        train_idx = train_df['month_num'] != val_month
        val_idx = train_df['month_num'] == val_month
        gbm = lgb.LGBMClassifier(objective='multiclass', num_class=num_classes,
                                 n_estimators=n_estimators, random_state=42, n_jobs=-1)
        gbm.fit(X.iloc[train_idx], y[train_idx], eval_set=[(X.iloc[val_idx], y[val_idx])], verbose=False)
        val_pred = gbm.predict_proba(X.iloc[val_idx])
        score = weighted_roc_auc(y[val_idx], val_pred, weights, le)
        scores.append(score)
        preds_test += gbm.predict_proba(X_test) / len(months)
    return preds_test, ids_test, float(np.mean(scores)), le

def main():
    parser = argparse.ArgumentParser(description='Train LightGBM model with cross-validation')
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--output', type=str, default='submission.csv')
    args, _ = parser.parse_known_args()

    num_cols, cat_cols = load_feature_lists('feature_description.xlsx')
    weights = load_cluster_weights('cluster_weights.xlsx')
    full_df = prepare_full_dataframe(num_cols, cat_cols)
    preds, ids, score, le = train_lightgbm_cv(full_df, weights, n_estimators=args.n_estimators)
    logger.info('Mean CV weighted ROC-AUC: %s', score)
    submission = pd.DataFrame(preds, columns=le.classes_)
    submission.insert(0, 'id', ids)
    submission.to_csv(args.output, index=False)
    logger.info('Saved predictions to %s', args.output)

if __name__ == '__main__':
    main()
