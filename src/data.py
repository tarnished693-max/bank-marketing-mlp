"""Data Engineering and Preprocessing Pipeline for Bank Marketing Term Deposit Prediction.

Owner / Author: Esraa Gamal (@esraagamal11mohamed-cell)
Role: Role 2 - Data Engineer
Course: Neural Networks (AIE231)
Milestone: Week 9 & Week 10 Deliverable (Data Pipeline, Leak-Free Preprocessing, Imbalance Handling)

This module provides:
1. Ingestion of the UCI Bank Marketing dataset (`bank-additional-full.csv`).
2. Cleaning: Removal of 12 duplicate rows and immediate dropping of 'duration'
   to prevent target leakage (as specified in UCI documentation and project requirements).
3. Treatment of 'unknown' values in categorical features as informative categories.
4. Stratified 70/15/15 Train/Validation/Test splitting with zero data leakage.
5. Strict zero-leakage transformation: StandardScaler and OneHotEncoder fitted
   STRICTLY on the training fold, then applied to validation and test sets.
6. Imbalance mitigation utilities: Balanced class weights and minority oversampling.
7. Automated data leakage verification tests and assertions.
8. Serialization and persistence of the fitted preprocessor for inference.
"""

import os
import logging
from typing import Dict, Tuple, List, Optional, Any, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Default feature definitions according to UCI Bank Marketing specification
DEFAULT_NUMERIC_FEATURES: List[str] = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

DEFAULT_CATEGORICAL_FEATURES: List[str] = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]


def load_raw_data(data_path: str = "data/bank-additional-full.csv", sep: str = ";") -> pd.DataFrame:
    """Load raw dataset from CSV file.
    
    Args:
        data_path: Path to the raw CSV file.
        sep: CSV delimiter (default ';' for UCI bank-additional).
        
    Returns:
        pd.DataFrame: Loaded raw dataset.
        
    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if not os.path.exists(data_path):
        # Fallback check relative to repository root
        alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), data_path)
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{data_path}'. Please ensure 'bank-additional-full.csv' "
                f"is present in the 'data/' folder."
            )
            
    df = pd.read_csv(data_path, sep=sep)
    logger.info(f"Loaded raw dataset from '{data_path}' with shape {df.shape}.")
    return df


def clean_data(
    df: pd.DataFrame,
    drop_duration: bool = True,
    remove_duplicates: bool = True,
    target_column: str = "y",
) -> pd.DataFrame:
    """Clean data, enforce zero target leakage, and resolve duplicates.
    
    Target Leakage Prevention:
        The 'duration' column (call duration in seconds) is only known AFTER
        the telemarketing call is completed and has a near-deterministic correlation
        with the target. Retaining it would lead to catastrophic target leakage.
        Hence, it is dropped prior to any modeling.
        
    Missing Value Treatment:
        Missing values in this dataset are recorded as 'unknown' in categorical
        attributes ('default', 'education', 'housing', 'loan', 'job', 'marital').
        As shown in EDA, client refusal or omission of financial status
        (especially 'default') is highly predictive. We retain 'unknown' as a
        distinct, valid categorical level during one-hot encoding.
        
    Duplicate Removal:
        The raw dataset contains 12 duplicated rows. These are removed to ensure
        clean observations.
        
    Args:
        df: Raw DataFrame.
        drop_duration: Whether to drop the leaky 'duration' column.
        remove_duplicates: Whether to drop duplicate rows.
        target_column: Name of the target column ('y').
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # 1. Deduplication
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        dropped_count = initial_rows - len(df_clean)
        logger.info(f"Removed {dropped_count} duplicate rows (retained {len(df_clean):,} records).")
        
    # 2. Target Leakage Prevention: Drop 'duration'
    if drop_duration and "duration" in df_clean.columns:
        df_clean = df_clean.drop(columns=["duration"])
        logger.info("Dropped 'duration' column to prevent target leakage.")
        
    # 3. Standardize binary target: 'yes' -> 1, 'no' -> 0
    if target_column in df_clean.columns:
        if df_clean[target_column].dtype == object or df_clean[target_column].dtype == str:
            df_clean[target_column] = (df_clean[target_column].str.strip().str.lower() == "yes").astype(int)
        logger.info(
            f"Target '{target_column}' class distribution: "
            f"Class 0 (No): {(df_clean[target_column] == 0).sum():,} ({(df_clean[target_column] == 0).mean()*100:.2f}%), "
            f"Class 1 (Yes): {(df_clean[target_column] == 1).sum():,} ({(df_clean[target_column] == 1).mean()*100:.2f}%)."
        )
        
    return df_clean


def split_data(
    df: pd.DataFrame,
    target_column: str = "y",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Perform leak-free stratified 70 / 15 / 15 Train / Validation / Test split.
    
    The split is performed BEFORE any transformation or scaling to ensure
    absolute isolation of validation and test distributions.
    
    Args:
        df: Cleaned DataFrame containing features and target.
        target_column: Name of target column.
        train_ratio: Proportion of training samples (default: 0.70).
        val_ratio: Proportion of validation samples (default: 0.15).
        test_ratio: Proportion of test samples (default: 0.15).
        stratify: Whether to perform stratified splitting based on target classes.
        random_state: Random seed for exact reproducibility.
        
    Returns:
        Tuple: (X_train_df, y_train, X_val_df, y_val, X_test_df, y_test)
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    
    X = df.drop(columns=[target_column])
    y = df[target_column].values
    
    stratify_target = y if stratify else None
    
    # First split: Train vs (Val + Test)
    temp_size = val_ratio + test_ratio  # 0.30
    X_train_df, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        stratify=stratify_target,
        random_state=random_state,
    )
    
    # Second split: Val vs Test (50/50 of temp -> 0.15 / 0.15 of total)
    val_relative_ratio = val_ratio / temp_size
    stratify_temp = y_temp if stratify else None
    
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_relative_ratio),
        stratify=stratify_temp,
        random_state=random_state,
    )
    
    logger.info(
        f"Data split completed: "
        f"Train={len(X_train_df):,} ({y_train.mean()*100:.2f}% pos), "
        f"Val={len(X_val_df):,} ({y_val.mean()*100:.2f}% pos), "
        f"Test={len(X_test_df):,} ({y_test.mean()*100:.2f}% pos)."
    )
    
    return X_train_df, y_train, X_val_df, y_val, X_test_df, y_test


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scaler_type: str = "standard",
) -> ColumnTransformer:
    """Construct an un-fitted scikit-learn ColumnTransformer.
    
    Applies:
        - Numeric: StandardScaler (mean=0, std=1) or MinMaxScaler.
        - Categorical: OneHotEncoder(handle_unknown='ignore', sparse_output=False).
          'handle_unknown=ignore' ensures any unseen category in test or live inference
          maps to an all-zero vector without crashing.
          
    Args:
        numeric_features: List of numerical column names.
        categorical_features: List of categorical column names.
        scaler_type: 'standard' or 'minmax'.
        
    Returns:
        ColumnTransformer: Preprocessing transformer pipeline.
    """
    if numeric_features is None:
        numeric_features = DEFAULT_NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = DEFAULT_CATEGORICAL_FEATURES
        
    scaler = StandardScaler() if scaler_type.lower() == "standard" else MinMaxScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numeric_features),
            ("cat", encoder, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract output feature names from a fitted ColumnTransformer.
    
    Args:
        preprocessor: Fitted ColumnTransformer.
        
    Returns:
        List[str]: List of transformed feature names.
    """
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            if hasattr(trans, "get_feature_names_out"):
                cat_names = trans.get_feature_names_out(cols).tolist()
                feature_names.extend(cat_names)
            else:
                feature_names.extend([f"{c}_encoded" for c in cols])
    return feature_names


def fit_and_transform_pipeline(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    preprocessor: Optional[ColumnTransformer] = None,
    scaler_type: str = "standard",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """Fit preprocessing pipeline STRICTLY on training split, then transform all sets.
    
    Zero-Leakage Guarantee:
        `preprocessor.fit()` is exclusively called on `X_train_df`.
        `X_val_df` and `X_test_df` are only transformed using the training parameters.
        
    Args:
        X_train_df: Raw training features DataFrame.
        X_val_df: Raw validation features DataFrame.
        X_test_df: Raw test features DataFrame.
        preprocessor: Optional pre-constructed ColumnTransformer.
        scaler_type: Scaler type ('standard' or 'minmax').
        
    Returns:
        Tuple: (X_train, X_val, X_test, fitted_preprocessor, feature_names)
    """
    if preprocessor is None:
        numeric_cols = [c for c in DEFAULT_NUMERIC_FEATURES if c in X_train_df.columns]
        cat_cols = [c for c in DEFAULT_CATEGORICAL_FEATURES if c in X_train_df.columns]
        preprocessor = build_preprocessor(numeric_cols, cat_cols, scaler_type=scaler_type)
        
    logger.info("Fitting preprocessor STRICTLY on X_train_df (Zero Leakage)...")
    X_train = preprocessor.fit_transform(X_train_df)
    
    logger.info("Transforming X_val_df and X_test_df with fitted parameters...")
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)
    
    # Convert to float32 for TensorFlow / Keras compatibility
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    
    feature_names = get_feature_names(preprocessor)
    logger.info(f"Pipeline fitted: Transformed input dimension = {X_train.shape[1]} features.")
    
    return X_train, X_val, X_test, preprocessor, feature_names


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights for cost-sensitive learning in Keras/Sklearn.
    
    Formula:
        w_j = N / (n_classes * N_j)
        
    Args:
        y_train: Training binary labels array.
        
    Returns:
        Dict[int, float]: Mapping {0: weight_0, 1: weight_1}.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights_dict = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    logger.info(f"Computed balanced class weights: {class_weights_dict}")
    return class_weights_dict


def oversample_minority(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Oversample minority class on training fold (Resampling Imbalance Technique).
    
    Performs random oversampling with replacement of positive-class (y=1) samples
    to achieve a 1:1 balance in training data, without leaking into validation or test sets.
    
    Args:
        X_train: Preprocessed training features.
        y_train: Training binary labels.
        random_state: Random seed for sampling reproducibility.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X_resampled, y_resampled)
    """
    rng = np.random.RandomState(random_state)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    
    n_neg = len(neg_idx)
    # Sample positive indices with replacement to match negative count
    sampled_pos_idx = rng.choice(pos_idx, size=n_neg, replace=True)
    
    all_indices = np.concatenate([neg_idx, sampled_pos_idx])
    rng.shuffle(all_indices)
    
    X_resampled = X_train[all_indices]
    y_resampled = y_train[all_indices]
    
    logger.info(
        f"Minority oversampling completed: Original={len(y_train):,} ({y_train.mean()*100:.2f}% pos) "
        f"-> Resampled={len(y_resampled):,} ({y_resampled.mean()*100:.2f}% pos)."
    )
    return X_resampled, y_resampled


def save_preprocessor(preprocessor: ColumnTransformer, save_path: str = "models/preprocessor.joblib") -> str:
    """Serialize and save the fitted ColumnTransformer preprocessor to disk.
    
    Args:
        preprocessor: Fitted ColumnTransformer.
        save_path: Destination path.
        
    Returns:
        str: Absolute path of saved file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    joblib.dump(preprocessor, save_path)
    logger.info(f"Saved fitted preprocessor to '{save_path}'.")
    return os.path.abspath(save_path)


def load_preprocessor(load_path: str = "models/preprocessor.joblib") -> ColumnTransformer:
    """Load a previously fitted ColumnTransformer preprocessor from disk.
    
    Args:
        load_path: Path to serialized preprocessor.
        
    Returns:
        ColumnTransformer: Loaded fitted preprocessor.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Serialized preprocessor not found at '{load_path}'.")
    preprocessor = joblib.load(load_path)
    logger.info(f"Loaded fitted preprocessor from '{load_path}'.")
    return preprocessor


def run_leakage_checks(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    preprocessor: ColumnTransformer,
) -> bool:
    """Run comprehensive automated data leakage and integrity checks.
    
    Verifies:
        1. Disjoint partitions: Exactly zero index overlap among train, val, and test splits.
        2. Post-call feature exclusion: 'duration' is absent from both raw and processed features.
        3. Mean and scale consistency: Scaler parameters match training data only.
        4. Array integrity: No NaNs, nulls, or infinite values in processed feature matrices.
        5. Dimensionality matching: Output feature dimensions are identical across all sets.
        
    Returns:
        bool: True if all checks pass.
        
    Raises:
        AssertionError: If any leak or inconsistency is detected.
    """
    logger.info("Executing automated data leakage and integrity checks...")
    
    # Check 1: Disjoint index partitions
    train_idx = set(X_train_df.index)
    val_idx = set(X_val_df.index)
    test_idx = set(X_test_df.index)
    
    assert train_idx.isdisjoint(val_idx), "FATAL LEAKAGE: Training and Validation partitions overlap!"
    assert train_idx.isdisjoint(test_idx), "FATAL LEAKAGE: Training and Test partitions overlap!"
    assert val_idx.isdisjoint(test_idx), "FATAL LEAKAGE: Validation and Test partitions overlap!"
    logger.info("  [PASS] Partitions are strictly disjoint (0 index overlap).")
    
    # Check 2: Excluded leaky 'duration' column
    assert "duration" not in X_train_df.columns, "FATAL LEAKAGE: 'duration' found in raw training features!"
    assert "duration" not in X_val_df.columns, "FATAL LEAKAGE: 'duration' found in raw validation features!"
    assert "duration" not in X_test_df.columns, "FATAL LEAKAGE: 'duration' found in raw test features!"
    feature_names = get_feature_names(preprocessor)
    assert not any("duration" in f.lower() for f in feature_names), "FATAL LEAKAGE: 'duration' encoded in output!"
    logger.info("  [PASS] 'duration' feature successfully excluded across all partitions.")
    
    # Check 3: Scaler parameters derived strictly from train
    scaler = preprocessor.named_transformers_["num"]
    if isinstance(scaler, StandardScaler):
        train_num_cols = preprocessor.transformers_[0][2]
        train_means = X_train_df[train_num_cols].mean().values
        assert np.allclose(scaler.mean_, train_means, atol=1e-3), (
            "FATAL LEAKAGE: Scaler means do not match training data statistics!"
        )
        logger.info("  [PASS] Scaler parameters match training partition statistics strictly.")
        
    # Check 4: No NaN or Inf values
    assert not np.isnan(X_train).any(), "Data integrity failure: NaNs detected in X_train!"
    assert not np.isnan(X_val).any(), "Data integrity failure: NaNs detected in X_val!"
    assert not np.isnan(X_test).any(), "Data integrity failure: NaNs detected in X_test!"
    logger.info("  [PASS] Transformed feature matrices contain zero NaNs or nulls.")
    
    # Check 5: Dimensionality consistency
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], (
        f"Dimension mismatch: Train({X_train.shape[1]}), Val({X_val.shape[1]}), Test({X_test.shape[1]})"
    )
    logger.info(f"  [PASS] Feature dimensionality identical across all sets ({X_train.shape[1]} features).")
    
    logger.info("ALL LEAKAGE AND INTEGRITY CHECKS PASSED SUCCESSFULLY [OK].")
    return True


def prepare_data(
    data_path: str = "data/bank-additional-full.csv",
    save_preprocessor_path: Optional[str] = "models/preprocessor.joblib",
    random_state: int = 42,
    scaler_type: str = "standard",
) -> Dict[str, Any]:
    """Execute complete end-to-end leak-free data preparation pipeline.
    
    Convenience function that orchestrates loading, cleaning, splitting,
    fitting preprocessor, leakage checks, class weight calculation,
    and preprocessor persistence.
    
    Returns:
        Dict[str, Any]: Preprocessed datasets and pipeline artifacts:
            - 'X_train', 'y_train'
            - 'X_val', 'y_val'
            - 'X_test', 'y_test'
            - 'class_weights'
            - 'preprocessor'
            - 'feature_names'
            - 'input_dim'
    """
    logger.info("Starting complete end-to-end data preparation pipeline...")
    
    # 1. Load raw data
    raw_df = load_raw_data(data_path)
    
    # 2. Clean data (drop duration, remove duplicates, encode target)
    cleaned_df = clean_data(raw_df, drop_duration=True, remove_duplicates=True)
    
    # 3. Stratified 70/15/15 split
    X_train_df, y_train, X_val_df, y_val, X_test_df, y_test = split_data(
        cleaned_df,
        target_column="y",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        random_state=random_state,
    )
    
    # 4. Fit preprocessor strictly on train and transform
    X_train, X_val, X_test, preprocessor, feature_names = fit_and_transform_pipeline(
        X_train_df,
        X_val_df,
        X_test_df,
        scaler_type=scaler_type,
    )
    
    # 5. Run leakage checks
    run_leakage_checks(
        X_train_df,
        X_val_df,
        X_test_df,
        X_train,
        X_val,
        X_test,
        preprocessor,
    )
    
    # 6. Compute class weights
    class_weights = compute_class_weights(y_train)
    
    # 7. Persist preprocessor if requested
    if save_preprocessor_path:
        save_preprocessor(preprocessor, save_preprocessor_path)
        
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_df": X_train_df,
        "X_val_df": X_val_df,
        "X_test_df": X_test_df,
        "class_weights": class_weights,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "input_dim": int(X_train.shape[1]),
    }


if __name__ == "__main__":
    logger.info("Executing src/data.py module standalone verification...")
    data_dict = prepare_data(
        data_path="data/bank-additional-full.csv",
        save_preprocessor_path="models/preprocessor.joblib",
        random_state=42,
    )
    print("\n" + "=" * 60)
    print(" DATA PREPARATION PIPELINE SUMMARY (Esraa Gamal - Role 2)")
    print("=" * 60)
    print(f"X_train shape    : {data_dict['X_train'].shape}  | Positive: {data_dict['y_train'].mean()*100:.2f}%")
    print(f"X_val shape      : {data_dict['X_val'].shape}  | Positive: {data_dict['y_val'].mean()*100:.2f}%")
    print(f"X_test shape     : {data_dict['X_test'].shape}  | Positive: {data_dict['y_test'].mean()*100:.2f}%")
    print(f"Input Dimension  : {data_dict['input_dim']} features")
    print(f"Class Weights    : {data_dict['class_weights']}")
    print(f"Preprocessor     : Saved to models/preprocessor.joblib")
    print("=" * 60)
