
# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** G4od's Plan  
**Team Members:**Adithya, Gowtham kumar, Naren Kumar  
**Submission Date:** October 13, 2025

---

## 1. Executive Summary

We developed a text-only ensemble learning solution for product price prediction, achieving a SMAPE of **40.00** using three gradient boosting algorithms (LightGBM, XGBoost, CatBoost) combined with comprehensive feature engineering and binary search calibration. Our approach extracts 70+ structured features from catalog text and applies dual-method post-processing to precisely target the desired error metric.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

The challenge requires predicting product prices from textual catalog content without images. Key insights from EDA revealed that brand names, item package quantities, weight/volume measurements, pack sizes, and text statistics are strong price predictors with significant variance across product categories.

**Key Observations:**
- IPQ values and units strongly correlate with price  
- Brand-level price aggregations capture premium positioning  
- Pack size and bulk indicators show distinct pricing patterns  
- Text complexity metrics reflect product sophistication  

### 2.2 Solution Strategy

**Approach Type:** Ensemble with Binary Search Calibration  
**Core Innovation:** Dual calibration methodology (binary search + grid search) applied to out-of-fold predictions to achieve exact target SMAPE through iterative optimization, ensuring consistent metric performance across test sets.

---

## 3. Model Architecture

### 3.1 Architecture Overview

### 3.2 Model Components

**Text Processing Pipeline:**
- Preprocessing: Extract item names (3× weight), bullet points, normalize text, remove special characters  
- TF-IDF: `max_features=300`, `ngram_range=(1,3)`, `min_df=3`, `max_df=0.9`, `sublinear_tf=True`  
- Dimensionality Reduction: TruncatedSVD to 100 components  

**Feature Engineering (70+ features):**
- **IPQ Features:** value, log, sqrt, per_pack ratio, total_quantity  
- **Weight/Volume:** Regex extraction with unit normalization (oz, lb, g, kg, ml, l, gal)  
- **Pack Size:** Median aggregation from multiple patterns, log transformations, multipack/bulk flags  
- **Brand Extraction:** Regex from item name, encoded with price statistics (mean, median, std, count)  
- **Text Statistics:** length, word count, unique ratio, title metrics, bullet count, digit/punctuation ratios  
- **Numerical Features:** count, min, max, mean, median, std, range of numbers in text  
- **Category Detection:** Binary and scored features for 8 categories (food, beverage, health, beauty, etc.)  
- **Statistical Encodings:** Brand and unit-level price aggregations (mean, median, std, count)  

**Ensemble Models:**

**LightGBM (45% weight):**
- Parameters: `num_leaves=63`, `max_depth=8`, `lr=0.015`, `feature_fraction=0.7`, `bagging_fraction=0.7`, `reg_alpha/lambda=0.5`  
- Log-transformed target, MAE metric, `early_stopping=300`  

**XGBoost (30% weight):**
- Parameters: `max_depth=7`, `lr=0.015`, `subsample=0.7`, `colsample_bytree=0.7`, `min_child_weight=5`, `reg_alpha/lambda=0.5`  
- Log-transformed target, MAE metric, `early_stopping=300`  

**CatBoost (25% weight):**
- Parameters: `depth=8`, `lr=0.015`, `l2_leaf_reg=5`, `min_data_in_leaf=25`, `cat_features=[brand, unit]`  
- Log-transformed target, MAE loss, `early_stopping=300`  

---

## 4. Feature Engineering Techniques

### 4.1 Regex-Based Extraction
- **Weight/Volume:** Multi-pattern matching with conversion factors (lb→oz: 16×, g→oz: 0.035×, ml→floz: 0.034×)  
- **Pack Size:** 7 patterns including “pack of X”, “X-pack”, “X count”, median aggregation  
- **Brand:** First word extraction after removing prefixes (“The”, “A”, “An”, “New”, “Best”, “Premium”)  

### 4.2 Statistical Aggregations
- Brand-level target encoding: mean, median, std, count of prices  
- Unit-level target encoding with same statistics  
- Median imputation for unseen categories in test set  

### 4.3 Post-Processing Calibration

**Binary Search Method:**
- Iteratively finds optimal scale factor (`range: 0.5–2.0`, 200 iterations, `tolerance: 0.001`)  
- Adjusts predictions to hit exact target SMAPE on OOF predictions  

**Grid Search Method:**
- Two-stage search: coarse (50×40 grid) + fine (30×30 grid around best)  
- Optimizes both scale and shift parameters for bias correction  

Final method selected based on closest deviation from target SMAPE.

---

## 5. Model Performance

### 5.1 Validation Results
- **Pre-Calibration SMAPE:** 40.4–40.6 (2-Fold CV)  
- **Post-Calibration SMAPE:** **40.00** (Binary Search optimal scale: 0.989)  
- **Calibration Precision:** Within 0.01 of target  
- **Total Features:** 170 (70 structured + 100 TF-IDF)  
- **Training Time:** ~35 minutes on CPU  

**Individual Model Performance:**
- LightGBM: ~41.2 SMAPE  
- XGBoost: ~42.1 SMAPE  
- CatBoost: ~41.8 SMAPE  
- **Ensemble Improvement:** 1.5+ SMAPE reduction  

---

## 6. Conclusion

Our text-only ensemble approach successfully achieves exactly **40.00 SMAPE** through comprehensive feature engineering extracting structured information from unstructured catalog data, combined with optimally weighted gradient boosting models and precision calibration. The binary search methodology ensures consistent target metric achievement, demonstrating that sophisticated NLP and statistical techniques can match multimodal approaches without image features.

---

## Appendix

### A. Code Artifacts
Complete implementation available with binary search calibration, feature engineering pipeline, and ensemble training.

### B. Key Technologies
- **Libraries:** pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, scipy  
- **Runtime:** ~35 minutes for full pipeline  
- **Cross-Validation:** 2-Fold with `random_state=42`
