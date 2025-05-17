import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# --- Hyperparameters & Configuration ---
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

def weighted_roc_auc(y_true_int: np.ndarray, y_pred_proba: np.ndarray, weights_dict: dict, label_encoder_for_target: LabelEncoder) -> float:
    """Compute weighted ROC-AUC using class weights."""
    actual_present_labels_int = np.unique(y_true_int)
    try:
        known_labels_mask = np.isin(actual_present_labels_int, label_encoder_for_target.transform(label_encoder_for_target.classes_))
        actual_present_labels_int_known = actual_present_labels_int[known_labels_mask]
        
        if len(actual_present_labels_int_known) == 0:
            return 0.0
            
        original_string_labels_present = label_encoder_for_target.inverse_transform(actual_present_labels_int_known)
    except ValueError as e:
        return 0.0

    unnorm_weights = np.array([weights_dict.get(str(label_str), 0) for label_str in original_string_labels_present])

    if unnorm_weights.sum() == 0:
        weights = np.ones_like(unnorm_weights) / len(unnorm_weights) if len(unnorm_weights) > 0 else []
    else:
        weights = unnorm_weights / unnorm_weights.sum()

    if not len(weights) or len(actual_present_labels_int_known) == 0:
        return 0.0
    
    try:
        indices_for_present_labels = label_encoder_for_target.transform(original_string_labels_present)
        if np.any(indices_for_present_labels >= y_pred_proba.shape[1]):
            return 0.0
        classes_roc_auc = roc_auc_score(y_true_int, y_pred_proba, multi_class='ovr', average=None, labels=actual_present_labels_int_known)
    except ValueError as e:
        return 0.0

    if len(classes_roc_auc) != len(weights):
        min_len = min(len(classes_roc_auc), len(weights))
        return (classes_roc_auc[:min_len] * weights[:min_len]).sum() if min_len > 0 else 0.0

    return (classes_roc_auc * weights).sum()


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

# --- 2. Preprocessing and Feature Engineering ---
def preprocess_data(train_df_orig_loaded, test_df_orig_loaded, cluster_weights_df, sample_submission_df):
    logger.info("\n--- 2. Starting preprocessing and feature engineering... ---")
    
    def date_to_numeric(df_to_process):
        df_copy = df_to_process.copy()
        df_copy['month_num'] = df_copy['date'].str.replace('month_', '').astype(int)
        return df_copy
    
    train_df = date_to_numeric(train_df_orig_loaded)
    test_df = date_to_numeric(test_df_orig_loaded)
    
    # --- 2.1. Advanced start_cluster imputation for month_6 (Same as before) ---
    # ... (This section is kept identical to the previous working version) ...
    logger.info("Calculating start_cluster transition matrix for month_6 imputation...")
    all_original_start_clusters = pd.concat([train_df['start_cluster'], test_df['start_cluster']]).dropna().unique()
    all_original_start_clusters_str = [str(cat) for cat in all_original_start_clusters]
    start_cluster_dtype = pd.CategoricalDtype(categories=all_original_start_clusters_str, ordered=False)
    
    train_df['start_cluster'] = train_df['start_cluster'].astype(str).astype(start_cluster_dtype)
    test_df['start_cluster'] = test_df['start_cluster'].astype(str).astype(start_cluster_dtype)
    
    transition_data_frames = []
    train_df_sorted = train_df.sort_values(['id', 'month_num'])
    train_df_sorted['next_month_start_cluster'] = train_df_sorted.groupby('id')['start_cluster'].shift(-1)
    train_transitions = train_df_sorted.dropna(subset=['start_cluster', 'next_month_start_cluster'])
    if not train_transitions.empty:
        transition_data_frames.append(train_transitions[['start_cluster', 'next_month_start_cluster']])
    
    test_df_m4_m5 = test_df[test_df['month_num'].isin([4, 5])].copy()
    test_df_m4_m5_sorted = test_df_m4_m5.sort_values(['id', 'month_num'])
    test_df_m4_m5_sorted['start_cluster'] = test_df_m4_m5_sorted['start_cluster'].astype(start_cluster_dtype)
    test_df_m4_m5_sorted['next_month_start_cluster'] = test_df_m4_m5_sorted.groupby('id')['start_cluster'].shift(-1)
    test_transitions_m4_m5 = test_df_m4_m5_sorted[test_df_m4_m5_sorted['month_num'] == 4]
    test_transitions_m4_m5 = test_transitions_m4_m5.dropna(subset=['start_cluster', 'next_month_start_cluster'])
    if not test_transitions_m4_m5.empty:
        transition_data_frames.append(test_transitions_m4_m5[['start_cluster', 'next_month_start_cluster']])
    
    train_start_cluster_mode = train_df['start_cluster'].mode()
    train_start_cluster_mode = train_start_cluster_mode[0] if not train_start_cluster_mode.empty else None
    if train_start_cluster_mode is None and len(all_original_start_clusters_str) > 0:
        train_start_cluster_mode = all_original_start_clusters_str[0]
    elif train_start_cluster_mode is None:
        train_start_cluster_mode = "other_fallback_mode" 
        logger.warning(f"Warning: train_start_cluster_mode is None. Using '{train_start_cluster_mode}'.")
        if train_start_cluster_mode not in start_cluster_dtype.categories:
            new_categories = list(start_cluster_dtype.categories) + [train_start_cluster_mode]
            start_cluster_dtype = pd.CategoricalDtype(categories=new_categories, ordered=False)
            train_df['start_cluster'] = train_df['start_cluster'].astype(str).astype(start_cluster_dtype)
            test_df['start_cluster'] = test_df['start_cluster'].astype(str).astype(start_cluster_dtype)
    
    if not transition_data_frames:
        logger.warning("Warning: No data available to calculate transition matrix. Using simpler imputation for month_6.")
        month_6_idx_simple = test_df[test_df['month_num'] == 6].index
        if not month_6_idx_simple.empty:
            ids_for_month_6_simple = test_df.loc[month_6_idx_simple, 'id']
            imputed_m6_simple = pd.Series(index=ids_for_month_6_simple.index, dtype=start_cluster_dtype) 
            test_pivot_simple = test_df.pivot_table(index='id', columns='month_num', values='start_cluster', aggfunc='first')
            if 5 in test_pivot_simple.columns:
                map_val_5 = ids_for_month_6_simple.map(test_pivot_simple[5])
                imputed_m6_simple = imputed_m6_simple.fillna(map_val_5.astype(start_cluster_dtype))
            if 4 in test_pivot_simple.columns:
                map_val_4 = ids_for_month_6_simple.map(test_pivot_simple[4])
                imputed_m6_simple = imputed_m6_simple.fillna(map_val_4.astype(start_cluster_dtype))
            if train_start_cluster_mode is not None:
                imputed_m6_simple = imputed_m6_simple.fillna(train_start_cluster_mode)
            test_df.loc[month_6_idx_simple, 'start_cluster'] = imputed_m6_simple.values
    else:
        all_transitions_df = pd.concat(transition_data_frames)
        transition_counts = pd.crosstab(all_transitions_df['start_cluster'], all_transitions_df['next_month_start_cluster'])
        transition_probabilities = transition_counts.apply(lambda x: x / x.sum() if x.sum() > 0 else 0, axis=1).fillna(0)
        month_6_indices_to_fill = test_df[test_df['month_num'] == 6].index
        for idx in month_6_indices_to_fill:
            client_id = test_df.loc[idx, 'id']
            imputed_value_for_client = np.nan 
            sc_m5_series = test_df[(test_df['id'] == client_id) & (test_df['month_num'] == 5)]['start_cluster']
            if not sc_m5_series.empty and pd.notna(sc_m5_series.iloc[0]):
                sc_m5_actual = sc_m5_series.iloc[0]
                if sc_m5_actual in transition_probabilities.index and transition_probabilities.loc[sc_m5_actual].sum() > 0:
                    imputed_value_for_client = transition_probabilities.loc[sc_m5_actual].idxmax()
            if pd.isna(imputed_value_for_client): 
                sc_m4_series = test_df[(test_df['id'] == client_id) & (test_df['month_num'] == 4)]['start_cluster']
                if not sc_m4_series.empty and pd.notna(sc_m4_series.iloc[0]):
                    sc_m4_actual = sc_m4_series.iloc[0]
                    if sc_m4_actual in transition_probabilities.index and transition_probabilities.loc[sc_m4_actual].sum() > 0:
                        predicted_sc_m5_temp = transition_probabilities.loc[sc_m4_actual].idxmax()
                        if predicted_sc_m5_temp in transition_probabilities.index and transition_probabilities.loc[predicted_sc_m5_temp].sum() > 0:
                             imputed_value_for_client = transition_probabilities.loc[predicted_sc_m5_temp].idxmax()
            if pd.isna(imputed_value_for_client) and train_start_cluster_mode is not None:
                imputed_value_for_client = train_start_cluster_mode
            test_df.loc[idx, 'start_cluster'] = imputed_value_for_client
    
    if train_start_cluster_mode is not None:
        test_df.loc[test_df['month_num'] == 6, 'start_cluster'] = test_df.loc[test_df['month_num'] == 6, 'start_cluster'].fillna(train_start_cluster_mode)
    test_df['start_cluster'] = test_df['start_cluster'].astype(str).astype(start_cluster_dtype) 
    logger.info(f"Test df month_6 start_cluster NaNs after advanced imputation: {test_df[test_df['month_num'] == 6]['start_cluster'].isna().sum()}")
    
    
    # --- 2.2. Combine train and test ---
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    # Sort by id and month_num for consistent lag/rolling features
    full_df = full_df.sort_values(['id', 'month_num']).reset_index(drop=True)
    
    # --- 2.3. Label Encode Target (Definition) ---
    all_possible_clusters_str = sorted(list(set(train_df_orig_loaded['end_cluster'].astype(str).unique()) | set(str(c) for c in sample_submission_df.columns[1:])))
    le_target.fit(all_possible_clusters_str)
    num_classes = len(le_target.classes_)
    logger.info(f"Number of target classes: {num_classes}")
    
    # --- 2.4. Feature Engineering (Lags, Diffs, Rolling) ---
    logger.info("Starting extended feature engineering...")
    cat_cols_original_names = [
        "channel_code", "city", "city_type", "okved", "segment",
        "start_cluster", "ogrn_month", "ogrn_year"
    ]
    final_cat_features_list = [] 
    for col in cat_cols_original_names:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype(str) 
            if col == 'start_cluster':
                full_df[col] = full_df[col].astype(start_cluster_dtype)
            else:
                unique_vals = full_df[col].dropna().unique()
                cat_dtype = pd.CategoricalDtype(categories=unique_vals, ordered=False)
                full_df[col] = full_df[col].astype(cat_dtype)
            final_cat_features_list.append(col)
    
    num_cols_for_fe = [ # Renamed for clarity, as it's used for more than just lags now
        'balance_amt_avg', 'balance_amt_max', 'balance_amt_min', 'balance_amt_day_avg',
        'sum_cred_h_oper_3m', 'sum_deb_h_oper_3m', 'cnt_cred_h_oper_3m', 'cnt_deb_h_oper_3m',
        # Add other numerical cols if you want to create rolling features for them
    ]
    numerical_cols = [] # This will store all generated numerical features
    
    for col in num_cols_for_fe:
        if col in full_df.columns:
            if not pd.api.types.is_numeric_dtype(full_df[col]): 
                 try:
                    full_df[col] = pd.to_numeric(full_df[col], errors='coerce') 
                 except Exception as e: continue # Skip if cannot convert
            
            # Lag 1
            full_df[f'{col}_lag1'] = full_df.groupby('id')[col].shift(1) # Keep NaNs for rolling, fill later
            # Diff 1
            full_df[f'{col}_diff1'] = full_df[col].fillna(0) - full_df[f'{col}_lag1'].fillna(0)
            
            # Rolling window features (e.g., over past 2 periods, including current)
            # min_periods=1 ensures we get a value even if only 1 period is available
            gb = full_df.groupby('id')[col]
            full_df[f'{col}_roll_mean_2'] = gb.transform(lambda x: x.rolling(window=2, min_periods=1).mean())
            full_df[f'{col}_roll_std_2'] = gb.transform(lambda x: x.rolling(window=2, min_periods=1).std())
            # full_df[f'{col}_roll_mean_3'] = gb.transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            # full_df[f'{col}_roll_std_3'] = gb.transform(lambda x: x.rolling(window=3, min_periods=1).std())
    
            if col not in numerical_cols: numerical_cols.append(col)
            if f'{col}_lag1' not in numerical_cols: numerical_cols.append(f'{col}_lag1')
            if f'{col}_diff1' not in numerical_cols: numerical_cols.append(f'{col}_diff1')
            if f'{col}_roll_mean_2' not in numerical_cols: numerical_cols.append(f'{col}_roll_mean_2')
            if f'{col}_roll_std_2' not in numerical_cols: numerical_cols.append(f'{col}_roll_std_2')
            # if f'{col}_roll_mean_3' not in numerical_cols: numerical_cols.append(f'{col}_roll_mean_3')
            # if f'{col}_roll_std_3' not in numerical_cols: numerical_cols.append(f'{col}_roll_std_3')
    
    # Fill NaNs that resulted from lags or rolling features (especially at the beginning of each group)
    # These specific columns are now in numerical_cols list
    for n_col in numerical_cols:
        if n_col in full_df.columns and full_df[n_col].isnull().any():
            if '_std_' in n_col: # Standard deviation can be 0 if constant
                full_df[n_col] = full_df[n_col].fillna(0) 
            else: # For means, lags, diffs, fill with 0 or a more sophisticated group-wise mean/median
                full_df[n_col] = full_df[n_col].fillna(0)
    
    
    if 'prev_month_start_cluster' in full_df.columns: # This should have been created before this loop by shift(1)
        pass # It was already processed.
    else: # Create if somehow missed
        full_df['prev_month_start_cluster'] = full_df.groupby('id')['start_cluster'].shift(1)
    
    if 'prev_month_start_cluster' in full_df.columns: # Now process it
        placeholder = "MISSING_LAG_SC"
        existing_sc_categories = list(full_df['start_cluster'].dtype.categories) 
        prev_month_categories = existing_sc_categories + [placeholder] if placeholder not in existing_sc_categories else existing_sc_categories
        prev_start_cluster_dtype = pd.CategoricalDtype(categories=prev_month_categories, ordered=False)
        
        full_df['prev_month_start_cluster'] = full_df['prev_month_start_cluster'].astype(str).fillna(placeholder).astype(prev_start_cluster_dtype)
        if 'prev_month_start_cluster' not in final_cat_features_list:
            final_cat_features_list.append('prev_month_start_cluster')
    logger.info("Extended feature engineering finished.")
    
    # --- Sections 3, 4, 5 (Data Splitting, Raw Feature Prep, Scaling/OHE) ---
    # ... (These sections are kept identical to the previous working version, 
    #      they will use the newly generated features in `numerical_cols` and `final_cat_features_list`) ...
    # --- 3. Split back into Train and Test & Process Target ---
    logger.info("\n--- 3. Splitting back and processing target... ---")
    train_processed_df = full_df[full_df['is_train'] == 1].copy()
    test_processed_df = full_df[full_df['is_train'] == 0].copy()
    
    if 'end_cluster' in train_processed_df.columns:
        train_processed_df = train_processed_df.drop(columns=['end_cluster'])
    
    train_df_for_target_merge = date_to_numeric(train_df_orig_loaded.copy())[['id', 'month_num', 'end_cluster']]
    train_processed_df = train_processed_df.merge(
        train_df_for_target_merge,
        on=['id', 'month_num'],
        how='left'
    )
    
    if 'end_cluster' not in train_processed_df.columns:
        raise KeyError("Column 'end_cluster' is missing after merge. Check merge logic.")
    
    train_processed_df.dropna(subset=['end_cluster'], inplace=True)
    train_processed_df['end_cluster_encoded'] = le_target.transform(train_processed_df['end_cluster'].astype(str))
    train_processed_df['end_cluster_encoded'] = train_processed_df['end_cluster_encoded'].astype(int)
    
    features_to_drop = ['id', 'date', 'month_num', 'is_train', 'end_cluster', 'end_cluster_encoded',
                        'index_city_code', 'ogrn_inn_connect_code'] 
    
    potential_features = [col for col in full_df.columns if col not in features_to_drop]
    
    numerical_features_final = [
        col for col in numerical_cols 
        if col in potential_features and col in train_processed_df.columns and pd.api.types.is_numeric_dtype(train_processed_df[col])
    ]
    categorical_features_final = [
        col for col in final_cat_features_list
        if col in potential_features and col in train_processed_df.columns and (pd.api.types.is_categorical_dtype(train_processed_df[col]) or pd.api.types.is_string_dtype(train_processed_df[col]) or pd.api.types.is_object_dtype(train_processed_df[col]))
    ]
    
    logger.info(f"Final Numerical Features ({len(numerical_features_final)}): {numerical_features_final[:10]}...") # Print more
    logger.info(f"Final Categorical Features ({len(categorical_features_final)}): {categorical_features_final[:5]}...")
    
    # --- 4. Validation Strategy (Time-based) ---
    logger.info("\n--- 4. Creating train/validation split... ---")
    train_val_df = train_processed_df[train_processed_df['month_num'].isin([1, 2])]
    valid_df = train_processed_df[train_processed_df['month_num'] == 3]
    
    if train_val_df.empty: raise RuntimeError("Error: train_val_df is empty after month filtering.")
    if valid_df.empty: raise RuntimeError("Error: valid_df is empty after month filtering.")
    
    X_train_val_raw = train_val_df[numerical_features_final + categorical_features_final].copy()
    y_train_val_int = train_val_df['end_cluster_encoded'].copy()
    X_valid_raw = valid_df[numerical_features_final + categorical_features_final].copy()
    y_valid_int = valid_df['end_cluster_encoded'].copy()
    
    test_month_6_df = test_processed_df[test_processed_df['month_num'] == 6]
    if test_month_6_df.empty:
        logger.warning("Warning: test_month_6_df is empty. Predictions for test will be based on an empty structure.")
        X_test_month_6_raw = pd.DataFrame(columns=numerical_features_final + categorical_features_final) 
        ids_test_month_6 = pd.Series(dtype='object', name='id') 
    else:
        X_test_month_6_raw = test_month_6_df[numerical_features_final + categorical_features_final].copy()
        ids_test_month_6 = test_month_6_df['id'].copy()
    
    logger.info(f"Raw X_train_val shape: {X_train_val_raw.shape}, y_train_val shape: {y_train_val_int.shape}")
    logger.info(f"Raw X_valid shape: {X_valid_raw.shape}, y_valid shape: {y_valid_int.shape}")
    logger.info(f"Raw X_test_month_6 shape: {X_test_month_6_raw.shape}, Test IDs shape: {ids_test_month_6.shape}")
    
    if X_train_val_raw.empty or X_valid_raw.empty: raise RuntimeError("Error: Raw train_val or valid set is empty.")
    if y_train_val_int.empty or y_valid_int.empty: raise RuntimeError("Error: Target for train or validation is empty.")
    
    # --- 5. NN Specific Preprocessing (Scaling & One-Hot Encoding) ---
    # ... (Identical to previous version, using MAX_OHE_CATEGORIES from top) ...
    logger.info("\n--- 5. NN Specific Preprocessing (Scaling & One-Hot Encoding) ---")
    scaler = StandardScaler()
    logger.info("Scaling numerical features...")
    if not X_train_val_raw[numerical_features_final].empty:
        X_train_val_num_scaled = scaler.fit_transform(X_train_val_raw[numerical_features_final].fillna(0))
    else: 
        X_train_val_num_scaled = np.empty((len(X_train_val_raw), 0)) 
    
    if not X_valid_raw[numerical_features_final].empty:
        X_valid_num_scaled = scaler.transform(X_valid_raw[numerical_features_final].fillna(0))
    else:
        X_valid_num_scaled = np.empty((len(X_valid_raw), 0))
    
    if not X_test_month_6_raw.empty and not X_test_month_6_raw[numerical_features_final].empty:
        X_test_m6_num_scaled = scaler.transform(X_test_month_6_raw[numerical_features_final].fillna(0))
    elif not X_test_month_6_raw.empty and X_test_month_6_raw[numerical_features_final].empty: 
        X_test_m6_num_scaled = np.empty((len(X_test_month_6_raw), 0))
    else: 
        X_test_m6_num_scaled = np.empty((0, len(numerical_features_final) if numerical_features_final else 0))
    
    X_train_val_num_scaled_df = pd.DataFrame(X_train_val_num_scaled, columns=numerical_features_final, index=X_train_val_raw.index)
    X_valid_num_scaled_df = pd.DataFrame(X_valid_num_scaled, columns=numerical_features_final, index=X_valid_raw.index)
    X_test_m6_num_scaled_df = pd.DataFrame(X_test_m6_num_scaled, columns=numerical_features_final, index=X_test_month_6_raw.index if not X_test_month_6_raw.empty else None)
    logger.info("Numerical features scaled.")
    
    logger.info("\nManaging cardinality and One-Hot Encoding categorical features...")
    X_train_val_cat_mod = X_train_val_raw[categorical_features_final].astype(str).copy()
    X_valid_cat_mod = X_valid_raw[categorical_features_final].astype(str).copy()
    X_test_m6_cat_mod = X_test_month_6_raw[categorical_features_final].astype(str).copy() if not X_test_month_6_raw.empty else pd.DataFrame(columns=categorical_features_final, dtype=str)
    
    learned_top_categories = {}
    for col in categorical_features_final:
        n_unique_train = X_train_val_cat_mod[col].nunique()
        logger.info(f"  Processing categorical column '{col}': {n_unique_train} unique values in training split.")
        if n_unique_train > MAX_OHE_CATEGORIES:
            top_n_minus_1 = MAX_OHE_CATEGORIES - 1
            # print(f"    '{col}' has high cardinality ({n_unique_train}). Applying top-{top_n_minus_1} categories + an 'OTHER' category.")
            top_cats = X_train_val_cat_mod[col].value_counts().nlargest(top_n_minus_1).index.tolist()
            learned_top_categories[col] = top_cats
            other_value_for_col = f'{col}_OTHER_RARE'
            X_train_val_cat_mod[col] = X_train_val_cat_mod[col].apply(lambda x: x if x in top_cats else other_value_for_col)
            X_valid_cat_mod[col] = X_valid_cat_mod[col].apply(lambda x: x if x in top_cats else other_value_for_col)
            if not X_test_m6_cat_mod.empty:
                 X_test_m6_cat_mod[col] = X_test_m6_cat_mod[col].apply(lambda x: x if x in top_cats else other_value_for_col)
            # print(f"    '{col}' after capping: {X_train_val_cat_mod[col].nunique()} unique values (including OTHER).")
        else:
            learned_top_categories[col] = X_train_val_cat_mod[col].unique().tolist()
            # print(f"    '{col}' has {n_unique_train} categories, no capping applied.")
    
    logger.info("\nApplying pd.get_dummies...")
    X_train_val_cat_ohe = pd.get_dummies(X_train_val_cat_mod, dummy_na=False, dtype=np.uint8)
    X_valid_cat_ohe_unaligned = pd.get_dummies(X_valid_cat_mod, dummy_na=False, dtype=np.uint8)
    if not X_test_m6_cat_mod.empty:
        X_test_m6_cat_ohe_unaligned = pd.get_dummies(X_test_m6_cat_mod, dummy_na=False, dtype=np.uint8)
    else: 
        X_test_m6_cat_ohe_unaligned = pd.DataFrame(dtype=np.uint8) 
    logger.info("pd.get_dummies applied.")
    
    train_ohe_cols = X_train_val_cat_ohe.columns
    logger.info(f"Number of OHE columns from training data after potential capping: {len(train_ohe_cols)}")
    
    logger.info("Reindexing OHE columns for validation and test sets...")
    X_valid_cat_ohe = X_valid_cat_ohe_unaligned.reindex(columns=train_ohe_cols, fill_value=0).astype(np.uint8)
    if not X_test_m6_cat_ohe_unaligned.empty:
        X_test_m6_cat_ohe = X_test_m6_cat_ohe_unaligned.reindex(columns=train_ohe_cols, fill_value=0).astype(np.uint8)
    else: 
        X_test_m6_cat_ohe = pd.DataFrame(columns=train_ohe_cols, index=X_test_m6_num_scaled_df.index, dtype=np.uint8).fillna(0)
    logger.info("OHE columns aligned.")
    
    logger.info("\nCombining numerical and categorical features...")
    X_train_val_final = pd.concat([X_train_val_num_scaled_df, X_train_val_cat_ohe], axis=1)
    X_valid_final = pd.concat([X_valid_num_scaled_df, X_valid_cat_ohe], axis=1)
    X_test_m6_final = pd.concat([X_test_m6_num_scaled_df, X_test_m6_cat_ohe], axis=1)
    logger.info("Final feature matrices created.")
    
    logger.info(f"X_train_val_final shape: {X_train_val_final.shape}")
    logger.info(f"X_valid_final shape: {X_valid_final.shape}")
    logger.info(f"X_test_m6_final shape: {X_test_m6_final.shape}")
    
    logger.info("\nOne-hot encoding target variable...")
    y_train_val_ohe = to_categorical(y_train_val_int, num_classes=num_classes)
    y_valid_ohe = to_categorical(y_valid_int, num_classes=num_classes)
    logger.info("Target variable one-hot encoded.")
    
    if X_train_val_final.empty or X_valid_final.empty:
        raise RuntimeError("Error: Training or validation set is empty after NN preprocessing.")
    if y_train_val_int.empty or y_valid_int.empty:
        raise RuntimeError("Error: Target for training or validation is empty.")
    if X_train_val_final.shape[1] == 0 and len(X_train_val_final) > 0:
        raise RuntimeError("Error: X_train_val_final has no columns but has rows!")
    elif X_train_val_final.shape[1] == 0 and len(X_train_val_final) == 0:
        logger.warning("Warning: X_train_val_final is completely empty.")
        if not (len(y_train_val_int) == 0 and len(y_train_val_ohe) == 0):
            raise RuntimeError("Error: X_train_val_final empty but y_train_val not.")

    return X_train_val_final, y_train_val_int, y_train_val_ohe, X_valid_final, y_valid_int, y_valid_ohe, X_test_m6_final, ids_test_month_6, num_classes


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

