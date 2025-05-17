import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import weighted_roc_auc

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# --- Hyperparameters & Configuration ---
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train neural network model")
    parser.add_argument("--l2", type=float, default=L2_REG_LAMBDA, help="L2 regularization")
    parser.add_argument("--lr", type=float, default=INITIAL_LR, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Number of epochs")
    return parser.parse_args()
L2_REG_LAMBDA = 0.0001  # L2 Regularization strength (tune this: e.g., 0.00005, 0.0005, 0.001)
INITIAL_LR = 0.001    # Initial Learning Rate (tune this: e.g., 0.0005, 0.002)
BATCH_SIZE = 256      # Batch size (tune this: e.g., 128, 512)
MAX_EPOCHS = 150      # Increased max epochs, early stopping will handle it
MAX_OHE_CATEGORIES = 30 # For capping cardinality of categorical features

# --- 1. Load Data ---
def load_data():
    """Load training, test and auxiliary data files."""
    logger.info("--- 1. Loading Data ---")
    try:
        train_df_orig_loaded = pd.read_parquet("train_data.pqt")
        test_df_orig_loaded = pd.read_parquet("test_data.pqt")
        cluster_weights_df = pd.read_excel("cluster_weights.xlsx")
        sample_submission_df = pd.read_csv("sample_submission.csv")
    except FileNotFoundError as e:
        logger.error(f"Error loading data files: {e}")
        raise RuntimeError("Data loading failed") from e
    return train_df_orig_loaded, test_df_orig_loaded, cluster_weights_df, sample_submission_df


# --- Utility: Weighted ROC AUC ---
le_target = LabelEncoder()

class WeightedRocAucEarlyStopping(tf.keras.callbacks.Callback):
    """Early stopping based on weighted ROC-AUC metric."""

    def __init__(self, validation_data, weights_dict, label_encoder, patience=20, min_delta=1e-4, verbose=1):
        super().__init__()
        self.validation_data = validation_data
        self.weights_dict = weights_dict
        self.label_encoder = label_encoder
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best = -np.inf
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val_ohe, y_val_int = self.validation_data
        y_pred = self.model.predict(X_val, verbose=0)
        score = weighted_roc_auc(y_val_int, y_pred, self.weights_dict, self.label_encoder)
        if logs is not None:
            logs['val_weighted_auc'] = score
        if score > self.best + self.min_delta:
            self.best = score
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose:
                    print(f"WeightedRocAucEarlyStopping: stopping at epoch {epoch + 1} with best={self.best}")
                self.model.stop_training = True
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)


# --- Helper preprocessing functions ---

def parse_month_numbers(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Convert textual month labels to integers."""
    for df in (train_df, test_df):
        df['month_num'] = df['date'].str.replace('month_', '').astype(int)
    return train_df, test_df


def impute_start_cluster(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Simple month_6 start_cluster imputation using previous months."""
    all_cats = pd.concat([train_df['start_cluster'], test_df['start_cluster']]).dropna().astype(str).unique()
    dtype = pd.CategoricalDtype(categories=all_cats, ordered=False)
    train_df['start_cluster'] = train_df['start_cluster'].astype(str).astype(dtype)
    test_df['start_cluster'] = test_df['start_cluster'].astype(str).astype(dtype)
    mode_val = train_df['start_cluster'].mode()
    mode_val = mode_val.iloc[0] if not mode_val.empty else (all_cats[0] if len(all_cats) else 'unknown')
    pivot = test_df.pivot_table(index='id', columns='month_num', values='start_cluster', aggfunc='first')
    idx_m6 = test_df[test_df['month_num'] == 6].index
    if len(idx_m6) > 0:
        ids = test_df.loc[idx_m6, 'id']
        val = pivot.reindex(ids).get(5)
        fallback = pivot.reindex(ids).get(4)
        val = val.fillna(fallback).fillna(mode_val)
        test_df.loc[idx_m6, 'start_cluster'] = val.values.astype(str)
    return train_df, test_df, dtype


def generate_features(full_df: pd.DataFrame, sc_dtype: pd.CategoricalDtype):
    """Create basic lag and rolling features."""
    cat_cols = ["channel_code", "city", "city_type", "okved", "segment", "start_cluster", "ogrn_month", "ogrn_year"]
    num_cols = ["balance_amt_avg", "balance_amt_max", "balance_amt_min", "balance_amt_day_avg", "sum_cred_h_oper_3m", "sum_deb_h_oper_3m", "cnt_cred_h_oper_3m", "cnt_deb_h_oper_3m"]
    final_cat = []
    for col in cat_cols:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype(str)
            if col == "start_cluster":
                full_df[col] = full_df[col].astype(sc_dtype)
            final_cat.append(col)
    for col in num_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
            full_df[f"{col}_lag1"] = full_df.groupby('id')[col].shift(1)
            full_df[f"{col}_diff1"] = full_df[col].fillna(0) - full_df[f"{col}_lag1"].fillna(0)
            gb = full_df.groupby('id')[col]
            full_df[f"{col}_roll_mean_2"] = gb.transform(lambda x: x.rolling(window=2, min_periods=1).mean())
            full_df[f"{col}_roll_std_2"] = gb.transform(lambda x: x.rolling(window=2, min_periods=1).std())
            for n in [f"{col}_lag1", f"{col}_diff1", f"{col}_roll_mean_2", f"{col}_roll_std_2"]:
                full_df[n] = full_df[n].fillna(0)
    full_df['prev_month_start_cluster'] = full_df.groupby('id')['start_cluster'].shift(1).astype(str).fillna('MISSING')
    final_cat.append('prev_month_start_cluster')
    return full_df, num_cols, final_cat


def prepare_train_valid_test(full_df: pd.DataFrame, sc_dtype, train_df_orig: pd.DataFrame, sample_submission_df: pd.DataFrame):
    """Prepare matrices for training/validation/test."""
    train_df = full_df[full_df['is_train'] == 1].copy()
    test_df = full_df[full_df['is_train'] == 0].copy()
    le_target.fit(sorted(set(train_df_orig['end_cluster'].astype(str)) | set(sample_submission_df.columns[1:])))
    num_classes = len(le_target.classes_)
    train_df = train_df.merge(train_df_orig[['id', 'date', 'end_cluster']], on=['id', 'date'], how='left')
    train_df.dropna(subset=['end_cluster'], inplace=True)
    train_df['target'] = le_target.transform(train_df['end_cluster'].astype(str))
    feature_cols = [c for c in full_df.columns if c not in ['id', 'date', 'month_num', 'is_train', 'end_cluster']]
    X_train = train_df[train_df['month_num'].isin([1, 2])][feature_cols]
    y_train = train_df[train_df['month_num'].isin([1, 2])]['target']
    X_valid = train_df[train_df['month_num'] == 3][feature_cols]
    y_valid = train_df[train_df['month_num'] == 3]['target']
    X_test = test_df[test_df['month_num'] == 6][feature_cols]
    ids_test = test_df[test_df['month_num'] == 6]['id']
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train.select_dtypes(include=[np.number]).fillna(0))
    X_valid_num = scaler.transform(X_valid.select_dtypes(include=[np.number]).fillna(0))
    X_test_num = scaler.transform(X_test.select_dtypes(include=[np.number]).fillna(0)) if not X_test.empty else np.empty((0, X_train_num.shape[1]))
    cat_cols = [c for c in feature_cols if c in full_df.columns and (full_df[c].dtype == object or isinstance(full_df[c].dtype, pd.CategoricalDtype))]
    X_train_cat = pd.get_dummies(X_train[cat_cols].astype(str), dummy_na=False)
    X_valid_cat = pd.get_dummies(X_valid[cat_cols].astype(str), dummy_na=False)
    X_test_cat = pd.get_dummies(X_test[cat_cols].astype(str), dummy_na=False)
    X_valid_cat = X_valid_cat.reindex(columns=X_train_cat.columns, fill_value=0)
    X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)
    X_train_final = np.hstack([X_train_num, X_train_cat.values])
    X_valid_final = np.hstack([X_valid_num, X_valid_cat.values])
    X_test_final = np.hstack([X_test_num, X_test_cat.values])
    y_train_ohe = to_categorical(y_train, num_classes=num_classes)
    y_valid_ohe = to_categorical(y_valid, num_classes=num_classes)
    return (X_train_final, y_train, y_train_ohe, X_valid_final, y_valid, y_valid_ohe, X_test_final, ids_test, num_classes)


def preprocess_data(train_df_orig_loaded, test_df_orig_loaded, cluster_weights_df, sample_submission_df):
    logger.info("\n--- 2. Starting preprocessing and feature engineering... ---")
    train_df, test_df = parse_month_numbers(train_df_orig_loaded.copy(), test_df_orig_loaded.copy())
    train_df, test_df, sc_dtype = impute_start_cluster(train_df, test_df)
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    full_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(['id', 'month_num']).reset_index(drop=True)
    full_df, _, _ = generate_features(full_df, sc_dtype)
    return prepare_train_valid_test(full_df, sc_dtype, train_df_orig_loaded, sample_submission_df)

def train_and_predict(X_train_val_final, y_train_val_ohe, X_valid_final, y_valid_ohe, y_valid_int, num_classes, cluster_weights_df, X_test_m6_final, ids_test_month_6, sample_submission_df):
    # --- 6. Build and Train Neural Network ---
    logger.info("\n--- 6. Building and Training Neural Network (Wider First Layer, Tunable Params) ---")
    
    def create_nn_model(input_shape: int, num_classes_out: int, l2_lambda_val: float) -> tf.keras.Model:
        """Builds an improved feed-forward network for tabular data."""
        inputs = Input(shape=(input_shape,))
    
        x = Dense(1024, kernel_initializer="he_normal",
                  kernel_regularizer=regularizers.l2(l2_lambda_val))(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.5)(x)
    
        x = Dense(512, kernel_initializer="he_normal",
                  kernel_regularizer=regularizers.l2(l2_lambda_val))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.4)(x)
    
        x = Dense(256, kernel_initializer="he_normal",
                  kernel_regularizer=regularizers.l2(l2_lambda_val))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)
    
        x = Dense(128, kernel_initializer="he_normal",
                  kernel_regularizer=regularizers.l2(l2_lambda_val))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
    
        outputs = Dense(num_classes_out, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=INITIAL_LR, weight_decay=l2_lambda_val
            ),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(
                    name="auc_keras", multi_label=True, num_labels=num_classes_out
                ),
            ],
        )
        return model
    
    if X_train_val_final.shape[1] > 0: 
        nn_model = create_nn_model(input_shape=X_train_val_final.shape[1],
                                   num_classes_out=num_classes,
                                   l2_lambda_val=L2_REG_LAMBDA)
        nn_model.summary()

        weights_for_metric = cluster_weights_df.set_index("cluster")["unnorm_weight"].to_dict()

        train_sample_weights = np.array([
            weights_for_metric.get(str(label), 1.0)
            for label in le_target.inverse_transform(y_train_val_ohe.argmax(axis=1))
        ])

        w_auc_stop = WeightedRocAucEarlyStopping(
            validation_data=(X_valid_final, y_valid_ohe, y_valid_int.values),
            weights_dict=weights_for_metric,
            label_encoder=le_target,
            patience=20,
            verbose=1,
        )
        reduce_lr = ReduceLROnPlateau(monitor='val_weighted_auc', mode='max', factor=0.2, patience=7, min_lr=0.000001, verbose=1)

        logger.info("\n--- Fitting the NN model (Wider First Layer, Tunable Params) ---")
        history = nn_model.fit(
            X_train_val_final, y_train_val_ohe,
            validation_data=(X_valid_final, y_valid_ohe),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[w_auc_stop, reduce_lr],
            sample_weight=train_sample_weights,
            verbose=1
        )
        logger.info("\n--- NN Model training finished ---")
    
        # --- 7. Evaluate and Predict (Same as before) ---
        # ... (This section is kept identical to the previous working version) ...
        logger.info("\n--- 7. Evaluating and Predicting ---")
        y_valid_proba_nn = nn_model.predict(X_valid_final)
        valid_roc_auc_nn = weighted_roc_auc(y_valid_int.values, y_valid_proba_nn, weights_for_metric, le_target)
        logger.info(f"NN Validation Weighted ROC AUC: {valid_roc_auc_nn}")
    
        logger.info("\nPredicting for submission using NN...")
        if not X_test_m6_final.empty:
            test_pred_proba_nn = nn_model.predict(X_test_m6_final)
            submission_df_nn = pd.DataFrame(test_pred_proba_nn, columns=le_target.classes_)
            submission_df_nn['id'] = ids_test_month_6.values
        else: 
            logger.info("Test set for month 6 is empty. Creating an empty submission structure or default predictions.")
            submission_df_nn = pd.DataFrame(columns=['id'] + list(le_target.classes_))
            submission_df_nn['id'] = ids_test_month_6 
            for cls_col in le_target.classes_: 
                submission_df_nn[cls_col] = 1 / num_classes if num_classes > 0 else 0
    else: 
        logger.info("Error: No features available for training the model. Skipping model training and prediction.")
        submission_df_nn = pd.DataFrame(columns=['id'] + list(le_target.classes_))
        if not ids_test_month_6.empty:
            submission_df_nn['id'] = ids_test_month_6
            for cls_col in le_target.classes_:
                submission_df_nn[cls_col] = 1 / num_classes if num_classes > 0 else 0
        else: 
            submission_df_nn['id'] = pd.Series(dtype='object')
    
    submission_cols_order = ['id'] + list(sample_submission_df.columns[1:])
    for col in submission_cols_order:
        if col not in submission_df_nn.columns and col != 'id':
            logger.info(f"Adding missing column {col} to submission_df_nn")
            submission_df_nn[col] = 1 / num_classes if num_classes > 0 else 0 
    
    submission_df_nn = submission_df_nn[submission_cols_order]
    
    submission_filename = f"tf_nn_l2_{L2_REG_LAMBDA}_lr_{INITIAL_LR}_b_{BATCH_SIZE}_rollfeat.csv" 
    submission_df_nn.to_csv(submission_filename, index=False)
    logger.info(f"\nSubmission file created: {submission_filename}")
    logger.info(submission_df_nn.head())
    logger.info(f"Submission shape: {submission_df_nn.shape}, Sample submission shape: {sample_submission_df.shape}")
    
    if not ids_test_month_6.empty and submission_df_nn.shape[0] != len(ids_test_month_6): 
        logger.warning(f"Warning: Submission row count ({submission_df_nn.shape[0]}) mismatch with expected test IDs for month 6 ({len(ids_test_month_6)}).")
    elif ids_test_month_6.empty and not submission_df_nn.empty :
         logger.warning(f"Warning: Test IDs for month 6 were empty, but submission is not. Submission rows: {submission_df_nn.shape[0]}")
    
    if not all(c in submission_df_nn.columns for c in sample_submission_df.columns) or \
       not all(c in sample_submission_df.columns for c in submission_df_nn.columns):
        logger.warning(f"Warning: Submission columns mismatch sample.")
    
    logger.info("\nScript finished.")
    return submission_filename, submission_df_nn


def main():
    args = parse_args()
    global L2_REG_LAMBDA, INITIAL_LR, BATCH_SIZE, MAX_EPOCHS
    L2_REG_LAMBDA = args.l2
    INITIAL_LR = args.lr
    BATCH_SIZE = args.batch_size
    MAX_EPOCHS = args.epochs

    data = load_data()
    (
        X_train_val_final,
        y_train_val_int,
        y_train_val_ohe,
        X_valid_final,
        y_valid_int,
        y_valid_ohe,
        X_test_m6_final,
        ids_test_month_6,
        num_classes
    ) = preprocess_data(*data)
    train_and_predict(
        X_train_val_final,
        y_train_val_ohe,
        X_valid_final,
        y_valid_ohe,
        y_valid_int,
        num_classes,
        data[2],
        X_test_m6_final,
        ids_test_month_6,
        data[3]
    )


if __name__ == "__main__":
    main()

