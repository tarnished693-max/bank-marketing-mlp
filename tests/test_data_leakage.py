"""Unit Tests and Data Leakage Verification Suite.

Owner / Author: Esraa Gamal (@esraagamal11mohamed-cell)
Role: Role 2 - Data Engineer
Course: Neural Networks (AIE231)
Milestone: Week 10 Deliverable (Preprocessing Pipeline with Leakage Checks)

Validates:
1. Zero index overlap among Train, Validation, and Test partitions.
2. Complete exclusion of target-leaking 'duration' column from all feature sets.
3. Strict parameter isolation: Scaler parameters match training partition statistics only.
4. Robustness to unknown/unseen categorical levels during inference.
5. Preservation of stratification proportions across all splits.
6. Serialization and deserialization integrity of the fitted preprocessor.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.preprocessing import StandardScaler

from src.data import (
    load_raw_data,
    clean_data,
    split_data,
    build_preprocessor,
    fit_and_transform_pipeline,
    compute_class_weights,
    oversample_minority,
    run_leakage_checks,
    save_preprocessor,
    load_preprocessor,
    prepare_data,
    DEFAULT_NUMERIC_FEATURES,
    DEFAULT_CATEGORICAL_FEATURES,
)


class TestDataEngineeringPipeline(unittest.TestCase):
    """Test suite for Data Engineering pipeline and zero-leakage guarantees."""

    @classmethod
    def setUpClass(cls):
        """Prepare sample data for pipeline testing."""
        cls.data_path = "data/bank-additional-full.csv"
        if not os.path.exists(cls.data_path):
            cls.data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), cls.data_path)
            
        cls.raw_df = load_raw_data(cls.data_path)
        cls.clean_df = clean_data(cls.raw_df, drop_duration=True, remove_duplicates=True)
        
        cls.splits = split_data(
            cls.clean_df,
            target_column="y",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify=True,
            random_state=42,
        )
        cls.X_tr_df, cls.y_tr, cls.X_val_df, cls.y_val, cls.X_te_df, cls.y_te = cls.splits
        
        cls.X_tr, cls.X_val, cls.X_te, cls.preprocessor, cls.feat_names = fit_and_transform_pipeline(
            cls.X_tr_df,
            cls.X_val_df,
            cls.X_te_df,
            scaler_type="standard",
        )

    def test_deduplication(self):
        """Ensure all 12 duplicate records from the raw dataset are removed."""
        self.assertEqual(len(self.raw_df) - len(self.clean_df), 12)
        # Check that no two rows with identical original 21 attributes exist
        raw_no_dups = self.raw_df.drop_duplicates()
        self.assertEqual(len(self.clean_df), len(raw_no_dups))

    def test_duration_dropped(self):
        """Verify 'duration' is excluded from raw features and transformed outputs."""
        self.assertNotIn("duration", self.clean_df.columns)
        self.assertNotIn("duration", self.X_tr_df.columns)
        self.assertNotIn("duration", self.X_val_df.columns)
        self.assertNotIn("duration", self.X_te_df.columns)
        for col_name in self.feat_names:
            self.assertNotIn("duration", col_name.lower())

    def test_partition_disjointness(self):
        """Verify strict zero index overlap among train, val, and test splits."""
        train_idx = set(self.X_tr_df.index)
        val_idx = set(self.X_val_df.index)
        test_idx = set(self.X_te_df.index)
        
        self.assertTrue(train_idx.isdisjoint(val_idx), "Train and Val indices overlap!")
        self.assertTrue(train_idx.isdisjoint(test_idx), "Train and Test indices overlap!")
        self.assertTrue(val_idx.isdisjoint(test_idx), "Val and Test indices overlap!")
        self.assertEqual(len(train_idx) + len(val_idx) + len(test_idx), len(self.clean_df))

    def test_stratification_consistency(self):
        """Ensure class imbalance ratio is consistently preserved across splits."""
        global_pos_rate = self.clean_df["y"].mean()
        train_pos_rate = self.y_tr.mean()
        val_pos_rate = self.y_val.mean()
        test_pos_rate = self.y_te.mean()
        
        self.assertAlmostEqual(train_pos_rate, global_pos_rate, places=3)
        self.assertAlmostEqual(val_pos_rate, global_pos_rate, places=3)
        self.assertAlmostEqual(test_pos_rate, global_pos_rate, places=3)

    def test_scaler_zero_leakage(self):
        """Verify StandardScaler parameters were fitted exclusively on training set."""
        scaler = self.preprocessor.named_transformers_["num"]
        self.assertIsInstance(scaler, StandardScaler)
        
        num_cols = self.preprocessor.transformers_[0][2]
        train_means = self.X_tr_df[num_cols].mean().values
        train_vars = self.X_tr_df[num_cols].var(ddof=0).values
        
        np.testing.assert_allclose(scaler.mean_, train_means, rtol=1e-3)
        np.testing.assert_allclose(scaler.var_, train_vars, rtol=1e-3)
        
        # Ensure validation mean does NOT match scaler mean
        val_means = self.X_val_df[num_cols].mean().values
        self.assertFalse(np.allclose(scaler.mean_, val_means, atol=1e-2))

    def test_unseen_category_handling(self):
        """Ensure OneHotEncoder gracefully handles unseen category without error."""
        sample_df = self.X_val_df.head(2).copy()
        sample_df.iloc[0, sample_df.columns.get_loc("job")] = "extraterrestrial_astronaut"
        transformed = self.preprocessor.transform(sample_df)
        self.assertEqual(transformed.shape[1], self.X_tr.shape[1])
        self.assertFalse(np.isnan(transformed).any())

    def test_class_weights_computation(self):
        """Verify balanced class weights balance the minority loss."""
        weights = compute_class_weights(self.y_tr)
        self.assertIn(0, weights)
        self.assertIn(1, weights)
        # Class 1 is minority (~11.3%), so its weight must be substantially higher (~4.4x)
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(weights[0] * (self.y_tr == 0).sum(), weights[1] * (self.y_tr == 1).sum(), places=1)

    def test_minority_oversampling(self):
        """Ensure minority oversampling produces a 50:50 balanced training set."""
        X_res, y_res = oversample_minority(self.X_tr, self.y_tr, random_state=42)
        self.assertEqual(len(X_res), len(y_res))
        self.assertEqual((y_res == 0).sum(), (y_res == 1).sum())
        self.assertAlmostEqual(y_res.mean(), 0.50, places=4)

    def test_preprocessor_serialization(self):
        """Verify preprocessor can be saved and loaded with identical transformations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_preprocessor.joblib")
            save_preprocessor(self.preprocessor, save_path)
            loaded = load_preprocessor(save_path)
            
            orig_trans = self.preprocessor.transform(self.X_val_df.head(5))
            loaded_trans = loaded.transform(self.X_val_df.head(5))
            np.testing.assert_allclose(orig_trans, loaded_trans)

    def test_leakage_checks_function(self):
        """Ensure run_leakage_checks utility runs and returns True."""
        result = run_leakage_checks(
            self.X_tr_df,
            self.X_val_df,
            self.X_te_df,
            self.X_tr,
            self.X_val,
            self.X_te,
            self.preprocessor,
        )
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
