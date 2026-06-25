
# CarPrice-Predictor

> Predicting used car prices from raw, uncleaned, real-world Craigslist listings using classical ML regression techniques.

---

## What Makes This Different?

Most used car price prediction projects on the internet use pre-cleaned datasets and run a single model. This project is different. Here, the raw uncleaned Craigslist dataset is used and the following are explicitly addressed:

* Checking for **Heteroscedasticity**
* Testing **Linear Regression against regularized versions** (Ridge / Lasso / ElasticNet)
* Explaining **why regularization helped or didn't** in this specific project
* Engineering features that **measurably improved the model**

---

## Feature Selection Strategy

### Dropped Immediately — Zero Value Features

| Feature                              | Reason                                                                                                                |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `VIN`                              | Unique identifier per car (like a fingerprint). High cardinality — model can't learn any general trends. 100% noise. |
| `url`,`region_url`,`image_url` | Non-informative links                                                                                                 |
| `lat`,`long`                     | Redundant with region/state                                                                                           |
| `posting_date`                     | Not relevant to price                                                                                                 |
| `description`                      | Unstructured free text — requires NLP, out of scope                                                                  |
| `county`                           | Too granular, mostly empty                                                                                            |
| `paint_color`                      | Very low predictive power. Unless wildly unpopular color, a white Camry and silver Camry cost the same                |

---

### Redundant / Noisy — Recommended Drop

**`region` and `state`**

While geographic location can slightly affect car prices (e.g., 4WDs selling better in snowy states), having both is highly redundant. `region` contains hundreds of specific local areas causing high cardinality and overcomplicating the model. `state` was also dropped since shortform state names aren't interpretable as-is for this model.

---

### Proceed with Caution — Keep but Clean

**`model`**

Car models (e.g., "Camry", "Mustang", "F-150") are incredibly important for price. However, this column contains thousands of messy, misspelled, or overly-specific text entries (e.g., "Civic EX-L", "Civic LX"). Requires heavy text cleaning and target encoding. If too messy and causing overfitting, fall back to `manufacturer` + `type`.

---

### Gold Standard — Must Keep

These are the heavy hitters that directly drive used car price:

| Feature               | Why It Matters                                               |
| --------------------- | ------------------------------------------------------------ |
| `year`/`odometer` | Ultimate indicators of age and wear                          |
| `manufacturer`      | Brand directly affects resale value                          |
| `cylinders`         | Defines engine power class                                   |
| `fuel`              | Fuel type affects running costs and demand                   |
| `transmission`      | Automatic vs manual has market preference                    |
| `type`              | Sedan vs SUV vs truck affects price class                    |
| `condition`         | Directly signals depreciation level                          |
| `title_status`      | Critical — salvage/rebuilt titles lower price significantly |

---

## Understanding `title_status` Values

The title status tells the legal and history condition of the vehicle. Here's what each value means:

| Status         | Meaning                                                                                                                 | Price Impact                         |
| -------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `clean`      | Normal title. No major legal or insurance issues. Car was never declared total loss.                                    | Highest resale value                 |
| `rebuilt`    | Was previously damaged (salvage), repaired, and passed inspection. Now legally drivable.                                | Cheaper than clean                   |
| `lien`       | Active loan/financial claim on the car. Owner still owes money to a bank/lender. Transfer can have legal complications. | Moderate impact                      |
| `salvage`    | Heavily damaged, declared total loss by insurance (accident, flood, fire). Usually unsafe or expensive to repair.       | Very low resale value                |
| `missing`    | Title information unavailable or unknown. Could mean undocumented history.                                              | Treat carefully during preprocessing |
| `parts only` | Not intended for road use. Sold only for spare parts — usually non-functional or irreparable.                          | Extremely low prices                 |

> **Decision made:** `title_status` was dropped after EDA revealed `clean` dominates at **96.6%** of all rows — near-zero variance, contributing only noise to the model.

---

## Text Preprocessing — `model` Column

### Step 1 — String Standardization

Before encoding, clean typos and variations to reduce unique values. If `F-150`, `f150`, and `f-150 ` are treated as different things, cleaning solves half the problem.

```python
df['model'] = df['model'].astype(str).str.lower().str.strip()
df['model'] = df['model'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.strip()
df['model'] = df['model'].replace('', np.nan).fillna('other')
```

### Step 2 — Rare Category Grouping (Frequency Thresholding)

Thousands of models appear only once (e.g., `grand wagoneer`). A model cannot learn a price pattern from a car it only sees once. Group all models appearing fewer than 10 times into a catch-all `'other'` category — this drops unique count from 12,000+ to a few hundred.

```python
model_counts = X_train['model'].value_counts()
rare_models  = model_counts[model_counts < 10].index
X_train.loc[X_train['model'].isin(rare_models), 'model'] = 'other'
X_test.loc[X_test['model'].isin(rare_models), 'model']   = 'other'
```

---

## Encoding Strategy

### Target Encoding — `model` and `manufacturer`

For high-cardinality categorical columns, **Target Encoding** was chosen over the alternatives:

| Method                             | Why It Fails Here                                                                                               |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **One-Hot Encoding**         | Creates 500+ new columns → massive sparse matrix → curse of dimensionality → harms model performance         |
| **Label / Ordinal Encoding** | Assigns arbitrary integers (camry=1, f150=2) → model wrongly assumes numerical order → misleads the algorithm |
| **Target Encoding**          | Replaces each category with the mean target (price) for that category → compact, meaningful, order-free        |

#### Critical Rule — Prevent Target Leakage

Because Target Encoding uses `price` to calculate encoded values, it must be computed **only on training data** and then mapped onto test data:

```python
from category_encoders import TargetEncoder

TE_model        = TargetEncoder(cols=['model'])
TE_manufacturer = TargetEncoder(cols=['manufacturer'])

# Fit ONLY on X_train
X_train['model_Encoded']        = TE_model.fit_transform(X_train['model'], y_train)
X_train['manufacturer_Encoded'] = TE_manufacturer.fit_transform(X_train['manufacturer'], y_train)

# Transform X_test using the already-fitted encoder
X_test['model_Encoded']        = TE_model.transform(X_test['model'])
X_test['manufacturer_Encoded'] = TE_manufacturer.transform(X_test['manufacturer'])
```

If encoded on the full dataset before splitting, the model subtly "cheats" by seeing test set prices during training — leading to overly optimistic training scores but poor real-world performance.

---

### Ordinal Encoding — `condition`

`condition` is ordinal (values have a meaningful order), so it was encoded with an explicit order:

```
salvage → fair → good → excellent → like new → new
   0         1      2        3           4        5
```

### One-Hot Encoding — `fuel`, `type`, `transmission`

Nominal categories with no natural order. Applied using `ColumnTransformer` for efficiency:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CT = ColumnTransformer(
    transformers=[
        ('OHE_type',         OneHotEncoder(), ['type']),
        ('OHE_transmission', OneHotEncoder(), ['transmission'])
    ],
    remainder='passthrough'   # keeps all other columns untouched
)
```

> **Why `remainder='passthrough'` matters:** Without it, `ColumnTransformer` silently drops every column not explicitly listed. This single parameter tells it to process the specified columns and pass everything else through unchanged.

> **Why `index=X_train.index` matters during manual OHE:** Scikit-Learn's `fit_transform()` strips the Pandas index and returns an unindexed NumPy array starting at row 0. If rows were shuffled during splitting, missing `index=X_train.index` causes row misalignment when concatenating — filling the dataset with broken `NaN` values.

---

## Cylinders — Extracting Numbers from Text

```python
# "6 cylinders" → 6.0
df['cylinders'] = df['cylinders'].str.extract(r'(\d+)').astype(float)
```

| Regex Part         | Meaning                                                                |
| ------------------ | ---------------------------------------------------------------------- |
| `r`              | Raw string — backslashes treated literally                            |
| `\d`             | Any digit from 0–9                                                    |
| `+`              | One or more digits (captures`10`,`12`, not just single digits)     |
| `.astype(float)` | Converts extracted string`"6"`to actual number`6.0`for computation |

---

## Feature Transformations — `price`, `odometer`, `year`

> **The rule:** Log transform is for columns where values span multiple orders of magnitude (100 to 100,000). `price` and `odometer` qualify. `year` spans 1990–2024 — that's a timeline, not magnitude. Engineer timeline columns into age/duration instead.

### `price` — Remove Outliers First, Then Log Transform

Applying log transform before removing outliers does nothing useful — a `$1` or `$999,999` listing is still an extreme value after log transform. Order matters:

```python
# Step 1: Remove junk listings
df = df[(df['price'] >= 500) & (df['price'] <= 150000)]

# Step 2: Log transform the target
df['logged_price'] = np.log1p(df['price'])
```

| Before Log Transform                                                  | After Log Transform                                                   |
| --------------------------------------------------------------------- | --------------------------------------------------------------------- |
| ![1781934758242](https://claude.ai/chat/image/README/1781934758242.png) | ![1781934783261](https://claude.ai/chat/image/README/1781934783261.png) |

---

### `odometer` — Cap Extremes, Then Log Transform

A 250k mile car is real. A 999,999 mile car is a data entry error. Cap, don't remove:

```python
# Step 1: Cap extremes
df = df[(df['odometer'] >= 1000) & (df['odometer'] <= 300000)]

# Step 2: Log transform as a feature
df['logged_odometer'] = np.log1p(df['odometer'])
df.drop(columns=['odometer'], inplace=True)
```

---

### `year` — Engineer into `car_age` Instead

Log transforming `year` makes no mathematical sense — `log(2015)` is meaningless. `car_age` has a direct, interpretable relationship with price:

```python
df['car_age'] = 2026 - df['year']
df = df[(df['car_age'] >= 0) & (df['car_age'] <= 30)]
df.drop(columns=['year'], inplace=True)
```

#### Logged Price vs Car Age — Outlier Check

![1781935862124](https://claude.ai/chat/image/README/1781935862124.png)

The overall trend is clear — as `car_age` increases, `logged_price` decreases. This is exactly the relationship the model needs to learn. The log transform worked correctly. The scattered circles are **real listings** (low-mileage collectibles, luxury brands, flood-damaged cars) — not errors to be removed.

---

## Missing Value Analysis

![1781850768148](https://claude.ai/chat/image/README/1781850768148.png)

### Decision Table — Drop or Impute?

| Missing % | Action                              |
| --------- | ----------------------------------- |
| 0–5%     | Usually safe to impute              |
| 5–20%    | Impute carefully after analysis     |
| 20–40%   | Depends on feature importance       |
| 40–60%   | Usually drop unless highly valuable |
| 60–70%   | Mostly drop                         |
| > 85–90% | Almost always drop                  |

### Key Observations

* **`condition` and `cylinders`** have ~60% missing values. Not dropped because they contribute significantly to price prediction. Imputed after confirming feature importance.
* **`price`, `year`, `model`, `state`** are almost fully complete — these are the core features.
* **Missing values are not uniformly distributed across rows** — certain listing types systematically lack information, suggesting missingness is not completely random.
* **`type`, `title_status`, `odometer`** have moderate missing values — imputed since they are business-important features.
* **Some rows have multiple missing columns simultaneously** — likely lower-quality or incomplete seller entries.

### Handling `manufacturer`, `fuel`, `transmission` (< 5% missing)

The shape of the distribution for these columns remained identical after Complete Case Analysis (CCA) — confirming the missing data is  **completely at random (MCAR)** . Therefore CCA (dropping those rows directly) was performed instead of imputation:

![1782381850616](https://claude.ai/chat/image/README/1782381850616.png)

```python
df = df.dropna(subset=['manufacturer', 'fuel', 'transmission'])
```

---

## Categorical Column Distributions

Checked for dominant categories across all categorical features. A column where a single category fills 90%+ has near-zero variance and is practically useless for regression.

### Dominance Check Code

```python
print("=== Dominance Check ===")
for col in df.select_dtypes(include='object').columns:
    top_pct = df[col].value_counts(normalize=True).iloc[0] * 100
    if top_pct >= 90:
        print(f"  {col}: top category = {top_pct:.1f}%  <- Dominating")
    else:
        print(f"  {col}: top category = {top_pct:.1f}%  balanced enough")
```

> **`iloc[0]` explained:** `value_counts(normalize=True)` returns proportions for every category as a sorted list. `.iloc[0]` isolates only the top row (most frequent category). Without it, the `if` statement would crash trying to compare an entire list against a single number.

### Results

| Feature          | Top Category %  | Status                          |
| ---------------- | --------------- | ------------------------------- |
| `manufacturer` | 17.0%           | Balanced                        |
| `model`        | 1.9%            | Balanced                        |
| `condition`    | 50.3%           | Balanced                        |
| `cylinders`    | 39.1%           | Balanced                        |
| `fuel`         | 84.1%           | Balanced                        |
| `title_status` | **96.6%** | **Dominating — Dropped** |
| `transmission` | 78.7%           | Balanced                        |
| `type`         | 26.4%           | Balanced                        |

#### Distribution Charts

**manufacturer** (17.0% — balanced)
![1782110991188](https://claude.ai/chat/image/README/1782110991188.png)

**model** (1.9% — balanced)
![1782111006354](https://claude.ai/chat/image/README/1782111006354.png)

**condition** (50.3% — balanced)
![1782111026239](https://claude.ai/chat/image/README/1782111026239.png)

**cylinders** (39.1% — balanced)
![1782111035421](https://claude.ai/chat/image/README/1782111035421.png)

**fuel** (84.1% — balanced)
![1782111046940](https://claude.ai/chat/image/README/1782111046940.png)

**title_status** (96.6% — Dominating, dropped)
![1782111056235](https://claude.ai/chat/image/README/1782111056235.png)

**transmission** (78.7% — balanced)
![1782111066900](https://claude.ai/chat/image/README/1782111066900.png)

**type** (26.4% — balanced)
![1782111016226](https://claude.ai/chat/image/README/1782111016226.png)

---

## Pipeline Order (Industry Standard)

```
1. Basic Data Cleaning          → drop irrelevant cols, fix dtypes
2. Safe Feature Engineering     → row-level transforms (car_age, log_price, log_odometer)
3. Train-Test Split             ← THE WALL. Nothing that "learns" crosses this.
4. Unsafe Feature Engineering   → rare model grouping (computed from X_train only)
5. Imputation                   → fit on X_train, transform both
6. Encoding                     → fit on X_train, transform both
7. Scaling                      → fit on X_train, transform both
8. Model Training
```

> **The golden rule:** If a transformation needs to compute any statistic from the dataset to work — it goes after the split, fitted on `X_train` only, applied to both `X_train` and `X_test`.

---

## Model Results

| Model                   | R²              | MAE              | RMSE             |
| ----------------------- | ---------------- | ---------------- | ---------------- |
| Linear Regression       | 0.6786           | 0.3163           | 0.4969           |
| Ridge                   | 0.6786           | 0.3163           | 0.4969           |
| Lasso                   | 0.6785           | 0.3164           | 0.4970           |
| **Random Forest** | **0.8562** | **0.1476** | **0.3324** |
| Random Forest (CV)      | **0.8400** | —               | —               |

> **Key finding:** Ridge and Lasso performed identically to plain Linear Regression because multicollinearity was already addressed during the VIF check — there was nothing left for regularization to fix. The large gap between linear models (R²=0.67) and Random Forest (R²=0.856) confirms that **used car pricing relationships are significantly nonlinear** — features interact with each other in ways a straight line cannot capture.
>
