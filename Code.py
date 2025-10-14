"""
Amazon ML Challenge 2025 - EXACT 40 SMAPE with Binary Search
Team: G4od's Plan
Target: EXACTLY 40 SMAPE
"""

import pandas as pd
import numpy as np
import re
import warnings
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy import stats

# ============================================================================
# SMAPE Metric
# ============================================================================
def smape(y_true, y_pred):
    """Calculate SMAPE score"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.maximum(y_pred, 0.01)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    return np.mean(diff / denominator) * 100

# ============================================================================
# BINARY SEARCH FOR EXACT TARGET
# ============================================================================
def binary_search_for_target(y_true, y_pred_base, target_smape=40.0, tolerance=0.001):
    """Binary search to find exact scaling factor for target SMAPE"""
    
    low, high = 0.5, 2.0
    best_scale = 1.0
    best_diff = float('inf')
    best_smape = smape(y_true, y_pred_base)
    
    print(f"\n🔍 Binary Search for Target SMAPE = {target_smape}")
    print(f"Initial SMAPE: {best_smape:.6f}")
    
    for iteration in range(200):  # More iterations for precision
        mid = (low + high) / 2
        scaled_pred = y_pred_base * mid
        scaled_pred = np.maximum(scaled_pred, 0.1)
        
        current_smape = smape(y_true, scaled_pred)
        diff = abs(current_smape - target_smape)
        
        if diff < best_diff:
            best_diff = diff
            best_scale = mid
            best_smape = current_smape
        
        if diff < tolerance:
            print(f"✅ Found optimal scale: {mid:.6f}, SMAPE: {current_smape:.6f} (iteration {iteration})")
            return mid
        
        if current_smape < target_smape:
            # Current SMAPE too low, need to increase error (worse predictions)
            low = mid
        else:
            # Current SMAPE too high, need to decrease error (better predictions)
            high = mid
        
        if iteration % 50 == 0:
            print(f"  Iteration {iteration}: scale={mid:.6f}, SMAPE={current_smape:.6f}, diff={diff:.6f}")
    
    print(f"✅ Best scale found: {best_scale:.6f}, SMAPE: {best_smape:.6f}, diff: {best_diff:.6f}")
    return best_scale

def grid_search_calibration(y_true, y_pred, target=40.0):
    """Grid search for best scale and shift parameters"""
    
    print(f"\n🔍 Grid Search Calibration for Target = {target}")
    
    best_params = {'scale': 1.0, 'shift': 0.0}
    best_diff = float('inf')
    best_smape = 0
    
    # Coarse search
    for scale in np.linspace(0.80, 1.20, 50):
        for shift in np.linspace(-15, 15, 40):
            pred_calibrated = y_pred * scale + shift
            pred_calibrated = np.maximum(pred_calibrated, 0.1)
            
            current_smape = smape(y_true, pred_calibrated)
            diff = abs(current_smape - target)
            
            if diff < best_diff:
                best_diff = diff
                best_params = {'scale': scale, 'shift': shift}
                best_smape = current_smape
                
                if diff < 0.01:
                    print(f"✅ Found: scale={scale:.4f}, shift={shift:.2f}, SMAPE={current_smape:.6f}")
                    return best_params
    
    # Fine search around best
    scale_center = best_params['scale']
    shift_center = best_params['shift']
    
    for scale in np.linspace(scale_center - 0.05, scale_center + 0.05, 30):
        for shift in np.linspace(shift_center - 3, shift_center + 3, 30):
            pred_calibrated = y_pred * scale + shift
            pred_calibrated = np.maximum(pred_calibrated, 0.1)
            
            current_smape = smape(y_true, pred_calibrated)
            diff = abs(current_smape - target)
            
            if diff < best_diff:
                best_diff = diff
                best_params = {'scale': scale, 'shift': shift}
                best_smape = current_smape
    
    print(f"✅ Best: scale={best_params['scale']:.6f}, shift={best_params['shift']:.2f}, SMAPE={best_smape:.6f}")
    return best_params

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_brand(text):
    """Extract brand from item name"""
    try:
        match = re.search(r'Item Name:\s*([^\n]+)', text)
        if match:
            title = match.group(1).strip()
            title = re.sub(r'^(The|A|An|New|Best|Top|Premium)\s+', '', title, flags=re.IGNORECASE)
            words = title.split()
            if words:
                brand = re.sub(r'[^\w]', '', words[0]).upper()
                return brand if len(brand) > 1 else 'UNKNOWN'
        return 'UNKNOWN'
    except:
        return 'UNKNOWN'

def extract_weight_volume(text):
    """Extract weight and volume with unit conversion to oz/floz"""
    features = {}
    text_lower = text.lower()

    weight_patterns = {
        'oz': (r'(\d+\.?\d*)\s*(?:oz|ounce|ounces?)(?!\s*fl)', 1.0),
        'lb': (r'(\d+\.?\d*)\s*(?:lb|lbs|pound|pounds?)', 16.0),
        'g': (r'(\d+\.?\d*)\s*(?:g|gram|grams?)(?!\s*ml)', 0.035274),
        'kg': (r'(\d+\.?\d*)\s*(?:kg|kilogram|kilograms?)', 35.274),
        'mg': (r'(\d+\.?\d*)\s*(?:mg|milligram|milligrams?)', 0.000035274),
    }

    volume_patterns = {
        'floz': (r'(\d+\.?\d*)\s*(?:fl\.?\s*oz|fluid\s*ounce)', 1.0),
        'ml': (r'(\d+\.?\d*)\s*(?:ml|milliliter|milliliters?)', 0.033814),
        'l': (r'(\d+\.?\d*)\s*(?:l|liter|liters?)(?!\s*oz)', 33.814),
        'gal': (r'(\d+\.?\d*)\s*(?:gal|gallon|gallons?)', 128.0),
    }

    weights = []
    volumes = []

    for pattern, multiplier in weight_patterns.values():
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                weights.append(float(match) * multiplier)
            except:
                pass

    for pattern, multiplier in volume_patterns.values():
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                volumes.append(float(match) * multiplier)
            except:
                pass

    features['total_weight_oz'] = sum(weights) if weights else 0
    features['max_weight_oz'] = max(weights) if weights else 0
    features['total_volume_floz'] = sum(volumes) if volumes else 0
    features['max_volume_floz'] = max(volumes) if volumes else 0
    features['has_weight'] = int(len(weights) > 0)
    features['has_volume'] = int(len(volumes) > 0)

    return features

def extract_pack_size(text):
    """Extract pack size"""
    features = {}
    text_lower = text.lower()

    pack_patterns = [
        r'pack\s+of\s+(\d+)',
        r'(\d+)\s*[-–]\s*pack',
        r'(\d+)\s*pack(?!age)',
        r'\((\d+)\s*pack\)',
        r'(\d+)\s*count',
        r'(\d+)\s*ct\b',
        r'box\s+of\s+(\d+)',
    ]

    pack_sizes = []
    for pattern in pack_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                size = int(match)
                if 1 <= size <= 10000:
                    pack_sizes.append(size)
            except:
                pass

    if pack_sizes:
        features['pack_size'] = int(np.median(pack_sizes))
    else:
        features['pack_size'] = 1

    features['log_pack_size'] = np.log1p(features['pack_size'])
    features['is_multipack'] = int(features['pack_size'] > 1)
    features['is_bulk'] = int(features['pack_size'] >= 12)

    return features

def extract_ipq(text):
    """Extract Item Package Quantity"""
    value = 0.0
    unit = 'UNKNOWN'

    try:
        value_match = re.search(r'Value:\s*([\d.]+)', text)
        if value_match:
            value = float(value_match.group(1))

        unit_match = re.search(r'Unit:\s*([^\n]+)', text)
        if unit_match:
            unit = unit_match.group(1).strip().upper()
            unit_map = {
                'FL OZ': 'FLOZ', 'FLUID OUNCE': 'FLOZ', 'FLUID OUNCES': 'FLOZ',
                'OZ': 'OZ', 'OUNCE': 'OZ', 'OUNCES': 'OZ',
                'LB': 'LB', 'POUND': 'LB', 'POUNDS': 'LB',
                'G': 'GRAM', 'GRAM': 'GRAM', 'GRAMS': 'GRAM',
                'ML': 'ML', 'L': 'LITER',
                'COUNT': 'COUNT', 'CT': 'COUNT',
                'EACH': 'EACH'
            }
            unit = unit_map.get(unit, unit)
    except:
        pass

    return value, unit

def extract_text_features(text):
    """Extract text statistics"""
    features = {}

    features['text_length'] = len(text)
    features['text_length_log'] = np.log1p(len(text))

    words = text.split()
    features['word_count'] = len(words)
    features['unique_words'] = len(set(words))
    features['unique_ratio'] = features['unique_words'] / max(features['word_count'], 1)

    word_lengths = [len(w) for w in words]
    features['avg_word_len'] = np.mean(word_lengths) if word_lengths else 0
    features['max_word_len'] = max(word_lengths) if word_lengths else 0

    title_match = re.search(r'Item Name:\s*([^\n]+)', text)
    if title_match:
        title = title_match.group(1)
        features['title_length'] = len(title)
        features['title_words'] = len(title.split())
        features['title_capitals'] = sum(1 for c in title if c.isupper())
    else:
        features['title_length'] = 0
        features['title_words'] = 0
        features['title_capitals'] = 0

    bullets = re.findall(r'Bullet Point \d+:', text)
    features['num_bullets'] = len(bullets)
    features['has_bullets'] = int(len(bullets) > 0)

    features['num_digits'] = sum(c.isdigit() for c in text)
    features['num_commas'] = text.count(',')
    features['num_periods'] = text.count('.')
    features['digit_ratio'] = features['num_digits'] / max(len(text), 1)

    return features

def extract_numbers(text):
    """Extract numerical statistics"""
    features = {}

    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    if numbers:
        nums = [float(n) for n in numbers]
        features['num_count'] = len(nums)
        features['num_max'] = max(nums)
        features['num_min'] = min(nums)
        features['num_mean'] = np.mean(nums)
        features['num_median'] = np.median(nums)
        features['num_std'] = np.std(nums) if len(nums) > 1 else 0
        features['num_range'] = max(nums) - min(nums)
    else:
        features['num_count'] = 0
        features['num_max'] = 0
        features['num_min'] = 0
        features['num_mean'] = 0
        features['num_median'] = 0
        features['num_std'] = 0
        features['num_range'] = 0

    return features

def extract_categories(text):
    """Extract category features"""
    features = {}
    text_lower = text.lower()

    categories = {
        'food': ['food', 'snack', 'cookie', 'chip', 'candy', 'chocolate', 'cereal'],
        'beverage': ['drink', 'beverage', 'juice', 'water', 'soda', 'tea', 'coffee'],
        'supplement': ['vitamin', 'supplement', 'protein', 'mineral'],
        'health': ['health', 'organic', 'natural', 'gluten-free'],
        'baby': ['baby', 'infant', 'diaper', 'wipes'],
        'beauty': ['beauty', 'makeup', 'cosmetic', 'skin', 'hair'],
        'cleaning': ['clean', 'detergent', 'soap', 'wash'],
        'pet': ['pet', 'dog', 'cat', 'treat']
    }

    for cat, keywords in categories.items():
        count = sum(text_lower.count(kw) for kw in keywords)
        features[f'cat_{cat}'] = int(count > 0)
        features[f'cat_{cat}_score'] = min(count, 5)

    return features

def engineer_features(df, brand_stats=None, unit_stats=None, is_train=True):
    """Master feature engineering function"""
    print(f"\n{'='*80}")
    print(f"🔧 Feature Engineering: {'TRAIN' if is_train else 'TEST'}")
    print(f"{'='*80}")

    df = df.copy()
    all_features = []

    for i, text in enumerate(df['catalog_content']):
        if i % 10000 == 0:
            print(f"  Progress: {i:,}/{len(df):,} ({100*i/len(df):.1f}%)")

        features = {}

        features['brand'] = extract_brand(text)

        value, unit = extract_ipq(text)
        features['ipq_value'] = value
        features['ipq_unit'] = unit

        pack_info = extract_pack_size(text)
        features.update(pack_info)

        weight_vol = extract_weight_volume(text)
        features.update(weight_vol)

        text_feats = extract_text_features(text)
        features.update(text_feats)

        numbers = extract_numbers(text)
        features.update(numbers)

        categories = extract_categories(text)
        features.update(categories)

        # Derived features
        features['ipq_per_pack'] = features['ipq_value'] / max(features['pack_size'], 1)
        features['total_quantity'] = features['ipq_value'] * features['pack_size']
        features['log_ipq'] = np.log1p(features['ipq_value'])
        features['log_total_qty'] = np.log1p(features['total_quantity'])
        features['sqrt_ipq'] = np.sqrt(features['ipq_value'])

        features['weight_per_pack'] = features['total_weight_oz'] / max(features['pack_size'], 1)
        features['volume_per_pack'] = features['total_volume_floz'] / max(features['pack_size'], 1)

        all_features.append(features)

    print(f"  Progress: {len(df):,}/{len(df):,} (100.0%)")

    features_df = pd.DataFrame(all_features)
    df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    # Brand statistics
    if is_train:
        brand_stats = df.groupby('brand')['price'].agg(['mean', 'median', 'std', 'count']).reset_index()
        brand_stats.columns = ['brand', 'brand_price_mean', 'brand_price_median', 'brand_price_std', 'brand_count']
        brand_stats['brand_price_std'] = brand_stats['brand_price_std'].fillna(0)

        unit_stats = df.groupby('ipq_unit')['price'].agg(['mean', 'median', 'std', 'count']).reset_index()
        unit_stats.columns = ['ipq_unit', 'unit_price_mean', 'unit_price_median', 'unit_price_std', 'unit_count']
        unit_stats['unit_price_std'] = unit_stats['unit_price_std'].fillna(0)

        df = df.merge(brand_stats, on='brand', how='left')
        df = df.merge(unit_stats, on='ipq_unit', how='left')
    else:
        if brand_stats is not None:
            df = df.merge(brand_stats, on='brand', how='left')
            df['brand_price_mean'] = df['brand_price_mean'].fillna(df['brand_price_mean'].median())
            df['brand_price_median'] = df['brand_price_median'].fillna(df['brand_price_median'].median())
            df['brand_price_std'] = df['brand_price_std'].fillna(0)
            df['brand_count'] = df['brand_count'].fillna(1)

        if unit_stats is not None:
            df = df.merge(unit_stats, on='ipq_unit', how='left')
            df['unit_price_mean'] = df['unit_price_mean'].fillna(df['unit_price_mean'].median())
            df['unit_price_median'] = df['unit_price_median'].fillna(df['unit_price_median'].median())
            df['unit_price_std'] = df['unit_price_std'].fillna(0)
            df['unit_count'] = df['unit_count'].fillna(1)

    # Prepare text for TF-IDF
    def prepare_text(text):
        try:
            parts = []
            name_match = re.search(r'Item Name:\s*([^\n]+)', text)
            if name_match:
                parts.extend([name_match.group(1).lower()] * 3)
            bullets = re.findall(r'Bullet Point \d+:\s*([^\n]+)', text)
            parts.extend([b.lower() for b in bullets])
            result = ' '.join(parts)
            result = re.sub(r'[^\w\s]', ' ', result)
            result = re.sub(r'\s+', ' ', result).strip()
            return result
        except:
            return text.lower()

    df['clean_text'] = df['catalog_content'].apply(prepare_text)

    print(f"✅ Features: {df.shape[1]} columns")

    return df, brand_stats, unit_stats

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("\n" + "="*80)
    print("🚀 EXACT 40 SMAPE TARGET with Binary Search 🚀")
    print("="*80)
    print("Team: God's Plan")
    print("Target: EXACTLY 40 SMAPE")
    print("="*80)

    TARGET_SMAPE = 40.0

    # Load data
    print("\n📂 Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")

    # Engineer features
    train_df, brand_stats, unit_stats = engineer_features(train_df, is_train=True)
    test_df, _, _ = engineer_features(test_df, brand_stats=brand_stats, unit_stats=unit_stats, is_train=False)

    # TF-IDF
    print(f"\n{'='*80}")
    print("📝 TF-IDF Vectorization...")
    print(f"{'='*80}")

    tfidf = TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )

    train_tfidf = tfidf.fit_transform(train_df['clean_text'])
    test_tfidf = tfidf.transform(test_df['clean_text'])

    # SVD dimensionality reduction
    svd = TruncatedSVD(n_components=100, random_state=42)
    train_tfidf_reduced = svd.fit_transform(train_tfidf)
    test_tfidf_reduced = svd.transform(test_tfidf)

    tfidf_cols = [f'tfidf_{i}' for i in range(train_tfidf_reduced.shape[1])]
    train_tfidf_df = pd.DataFrame(train_tfidf_reduced, columns=tfidf_cols)
    test_tfidf_df = pd.DataFrame(test_tfidf_reduced, columns=tfidf_cols)

    print(f"✅ TF-IDF: {train_tfidf_reduced.shape[1]} features (after SVD)")

    # Encode categoricals
    print(f"\n🔢 Encoding...")

    le_brand = LabelEncoder()
    train_df['brand_encoded'] = le_brand.fit_transform(train_df['brand'].astype(str))
    test_df['brand_encoded'] = test_df['brand'].apply(
        lambda x: le_brand.transform([str(x)])[0] if str(x) in le_brand.classes_ else -1
    )

    le_unit = LabelEncoder()
    train_df['unit_encoded'] = le_unit.fit_transform(train_df['ipq_unit'].astype(str))
    test_df['unit_encoded'] = test_df['ipq_unit'].apply(
        lambda x: le_unit.transform([str(x)])[0] if str(x) in le_unit.classes_ else -1
    )

    # Select features
    feature_cols = [
        'ipq_value', 'log_ipq', 'sqrt_ipq', 'pack_size', 'log_pack_size',
        'ipq_per_pack', 'total_quantity', 'log_total_qty',
        'total_weight_oz', 'max_weight_oz', 'total_volume_floz', 'max_volume_floz',
        'has_weight', 'has_volume', 'weight_per_pack', 'volume_per_pack',
        'text_length', 'text_length_log', 'word_count', 'unique_words', 'unique_ratio',
        'avg_word_len', 'max_word_len', 'num_bullets', 'has_bullets',
        'title_length', 'title_words', 'title_capitals',
        'num_digits', 'num_commas', 'num_periods', 'digit_ratio',
        'num_count', 'num_max', 'num_min', 'num_mean', 'num_median', 'num_std', 'num_range',
        'brand_encoded', 'unit_encoded',
        'brand_price_mean', 'brand_price_median', 'brand_price_std', 'brand_count',
        'unit_price_mean', 'unit_price_median', 'unit_price_std', 'unit_count',
        'is_multipack', 'is_bulk',
    ] + [f'cat_{c}' for c in ['food', 'beverage', 'supplement', 'health', 'baby', 'beauty', 'cleaning', 'pet']] + \
        [f'cat_{c}_score' for c in ['food', 'beverage', 'supplement', 'health', 'baby', 'beauty', 'cleaning', 'pet']]

    # Combine features
    X_train = pd.concat([
        train_df[feature_cols].reset_index(drop=True),
        train_tfidf_df
    ], axis=1)

    X_test = pd.concat([
        test_df[feature_cols].reset_index(drop=True),
        test_tfidf_df
    ], axis=1)

    y_train = train_df['price'].values

    print(f"\n✅ Total Features: {X_train.shape[1]}")

    # Cross-validation
    print(f"\n{'='*80}")
    print("🔄 2-Fold Cross-Validation")
    print(f"{'='*80}")

    n_folds = 4
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    cat_idx = [feature_cols.index('brand_encoded'), feature_cols.index('unit_encoded')]

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n{'='*80}")
        print(f"📊 Fold {fold+1}/{n_folds}")
        print(f"{'='*80}")

        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # Log transform target
        y_tr_log = np.log1p(y_tr)
        y_val_log = np.log1p(y_val)

        # LightGBM
        print("\n🌟 LightGBM...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.015,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'max_depth': 8,
            'min_child_samples': 25,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'min_split_gain': 0.01,
            'verbose': -1,
            'n_jobs': -1
        }

        lgb_train = lgb.Dataset(X_tr, y_tr_log, categorical_feature=cat_idx)
        lgb_val = lgb.Dataset(X_val, y_val_log, reference=lgb_train)

        lgb_model = lgb.train(
            lgb_params, lgb_train,
            num_boost_round=10000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(300), lgb.log_evaluation(500)]
        )

        pred_lgb_val = np.expm1(lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration))
        pred_lgb_test = np.expm1(lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration))

        # XGBoost
        print("\n🚀 XGBoost...")
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': 7,
            'learning_rate': 0.015,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 5,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'gamma': 0.01,
            'n_jobs': -1,
            'tree_method': 'hist'
        }

        xgb_train = xgb.DMatrix(X_tr, label=y_tr_log)
        xgb_val_d = xgb.DMatrix(X_val, label=y_val_log)
        xgb_test_d = xgb.DMatrix(X_test)

        xgb_model = xgb.train(
            xgb_params, xgb_train,
            num_boost_round=10000,
            evals=[(xgb_val_d, 'val')],
            early_stopping_rounds=300,
            verbose_eval=500
        )

        pred_xgb_val = np.expm1(xgb_model.predict(xgb_val_d))
        pred_xgb_test = np.expm1(xgb_model.predict(xgb_test_d))

        # CatBoost
        print("\n🐱 CatBoost...")
        cat_model = CatBoostRegressor(
            iterations=10000,
            learning_rate=0.015,
            depth=8,
            l2_leaf_reg=5,
            min_data_in_leaf=25,
            loss_function='MAE',
            eval_metric='MAE',
            early_stopping_rounds=300,
            verbose=500,
            cat_features=cat_idx,
            random_seed=42
        )

        cat_model.fit(X_tr, y_tr_log, eval_set=(X_val, y_val_log), use_best_model=True)

        pred_cat_val = np.expm1(cat_model.predict(X_val))
        pred_cat_test = np.expm1(cat_model.predict(X_test))

        # Ensemble
        pred_val = (pred_lgb_val * 0.45 + pred_xgb_val * 0.30 + pred_cat_val * 0.25)
        pred_test = (pred_lgb_test * 0.45 + pred_xgb_test * 0.30 + pred_cat_test * 0.25)

        pred_val = np.maximum(pred_val, 0.1)
        pred_test = np.maximum(pred_test, 0.1)

        oof_preds[val_idx] = pred_val
        test_preds += pred_test / n_folds

        fold_smape = smape(y_val, pred_val)
        fold_scores.append(fold_smape)

        print(f"\n⭐ Fold {fold+1} SMAPE: {fold_smape:.4f}")

    overall_smape = smape(y_train, oof_preds)

    print(f"\n{'='*80}")
    print("🎯 INITIAL RESULTS (Before Calibration)")
    print(f"{'='*80}")
    for i, s in enumerate(fold_scores):
        print(f"Fold {i+1}: {s:.4f}")
    print(f"{'='*80}")
    print(f"Mean CV: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"⭐⭐⭐ OOF SMAPE: {overall_smape:.6f} ⭐⭐⭐")
    print(f"Gap to target: {abs(overall_smape - TARGET_SMAPE):.6f}")
    print(f"{'='*80}")

    # ============================================================================
    # BINARY SEARCH CALIBRATION TO HIT EXACT TARGET
    # ============================================================================
    
    print(f"\n{'='*80}")
    print("🎯 CALIBRATING TO EXACT TARGET SMAPE = 40.00")
    print(f"{'='*80}")

    # Method 1: Binary Search (Scale only)
    print("\n📍 Method 1: Binary Search (Scale Only)")
    optimal_scale = binary_search_for_target(y_train, oof_preds, TARGET_SMAPE, tolerance=0.001)
    
    oof_scaled = oof_preds * optimal_scale
    oof_scaled = np.maximum(oof_scaled, 0.1)
    test_preds_method1 = test_preds * optimal_scale
    test_preds_method1 = np.maximum(test_preds_method1, 0.1)
    
    method1_smape = smape(y_train, oof_scaled)
    print(f"✅ Method 1 Final OOF SMAPE: {method1_smape:.6f}")

    # Method 2: Grid Search (Scale + Shift)
    print("\n📍 Method 2: Grid Search (Scale + Shift)")
    params = grid_search_calibration(y_train, oof_preds, TARGET_SMAPE)
    
    oof_calibrated = oof_preds * params['scale'] + params['shift']
    oof_calibrated = np.maximum(oof_calibrated, 0.1)
    test_preds_method2 = test_preds * params['scale'] + params['shift']
    test_preds_method2 = np.maximum(test_preds_method2, 0.1)
    
    method2_smape = smape(y_train, oof_calibrated)
    print(f"✅ Method 2 Final OOF SMAPE: {method2_smape:.6f}")

    # Choose best method
    methods = {
        'Binary Search (Scale)': (test_preds_method1, method1_smape, optimal_scale),
        'Grid Search (Scale+Shift)': (test_preds_method2, method2_smape, params),
    }

    best_method_name = min(methods.items(), key=lambda x: abs(x[1][1] - TARGET_SMAPE))[0]
    final_predictions, final_smape, final_params = methods[best_method_name]

    print(f"\n{'='*80}")
    print("🏆 FINAL CALIBRATED RESULTS")
    print(f"{'='*80}")
    print(f"Best Method: {best_method_name}")
    print(f"Parameters: {final_params}")
    print(f"🎯 FINAL OOF SMAPE: {final_smape:.6f}")
    print(f"📊 Deviation from target: {abs(final_smape - TARGET_SMAPE):.6f}")
    print(f"{'='*80}")

    # Save submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': final_predictions
    })
    submission['price'] = submission['price'].round(2)
    submission.to_csv('test_out.csv', index=False)

    print(f"\n✅ Saved: test_out.csv")

    # Final status
    print(f"\n{'='*80}")
    deviation = abs(final_smape - TARGET_SMAPE)
    if deviation < 0.01:
        print("🏆🏆🏆 PERFECT! Within 0.01 of target 40! 🏆🏆🏆")
    elif deviation < 0.05:
        print("🔥🔥🔥 EXCELLENT! Within 0.05 of target! 🔥🔥🔥")
    elif deviation < 0.1:
        print("🎉🎉 GREAT! Within 0.1 of target! 🎉🎉")
    elif deviation < 0.5:
        print("✅ Good! Close to target!")
    else:
        print(f"⚠️  Deviation: {deviation:.4f}")
    
    print(f"Target SMAPE: {TARGET_SMAPE:.6f}")
    print(f"Achieved SMAPE: {final_smape:.6f}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
