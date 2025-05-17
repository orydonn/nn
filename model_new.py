import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_OHE_CATEGORIES = 20


def weighted_roc_auc(y_true, y_pred_proba, weights_dict, le):
    """Compute weighted one-vs-all ROC AUC"""
    if len(y_true) == 0:
        return 0.0
    labels = np.unique(y_true)
    try:
        aucs = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=None, labels=labels)
    except ValueError:
        return 0.0
    labels_str = le.inverse_transform(labels)
    weights = np.array([weights_dict.get(str(lbl), 0) for lbl in labels_str], dtype=float)
    if weights.sum() == 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= weights.sum()
    return float(np.sum(aucs * weights))


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


def preprocess(train_df, test_df, num_cols, cat_cols):
    full = pd.concat([train_df, test_df], keys=['train', 'test'])
    full = cap_rare_categories(full, cat_cols)
    full = pd.get_dummies(full, columns=cat_cols, dummy_na=False)
    train_proc = full.xs('train')
    test_proc = full.xs('test')
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

    le = LabelEncoder()
    le.fit(train_df['end_cluster'].astype(str))
    train_df['target'] = le.transform(train_df['end_cluster'].astype(str))

    train_proc, test_proc = preprocess(train_df[num_cols + cat_cols], test_df[num_cols + cat_cols], num_cols, cat_cols)
    print('Preprocessing finished.')

    X = train_proc.values
    y = train_df['target'].values

    kf = KFold(n_splits=3, shuffle=False)
    oof_preds = np.zeros((len(train_df), len(le.classes_)))
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f'Starting fold {fold + 1}')
        print(f'Fold {fold + 1} training...')
        model = lgb.LGBMClassifier(n_estimators=300, random_state=42)
        model.fit(X[tr_idx], y[tr_idx])
        preds = model.predict_proba(X[val_idx])
        score = weighted_roc_auc(y[val_idx], preds, cluster_weights, le)
        logger.info(f'Fold {fold + 1} weighted AUC: {score:.4f}')
        print(f'Fold {fold + 1} AUC: {score:.4f}')
        oof_preds[val_idx] = preds
        fold_scores.append(score)

    logger.info(f'Mean CV weighted AUC: {np.mean(fold_scores):.4f}')
    print(f'CV done. Mean AUC: {np.mean(fold_scores):.4f}')

    logger.info('Training final model...')
    print('Training final model...')
    final_model = lgb.LGBMClassifier(n_estimators=300, random_state=42)
    final_model.fit(X, y)

    test_features = test_proc[test_df['date'] == 'month_6'].values
    ids = test_df.loc[test_df['date'] == 'month_6', 'id']
    logger.info(f'Predicting on {len(ids)} test rows...')
    print(f'Predicting on {len(ids)} rows...')
    test_pred = final_model.predict_proba(test_features)

    submission = pd.DataFrame(test_pred, columns=le.classes_)
    submission.insert(0, 'id', ids.values)
    submission = submission[submission_template.columns]
    submission.to_csv('submission_lightgbm.csv', index=False)
    logger.info('Submission saved to submission_lightgbm.csv')


if __name__ == '__main__':
    main()
