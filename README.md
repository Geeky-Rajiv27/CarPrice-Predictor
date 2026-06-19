# CarPrice-Predictor

    Here , i am gonna build a linear model using a raw uncleaned real world data to predict the price of the used car
    NOTE : Q: why this is different then other's thousands of car price prediction proeject ?

    Ans : Here i am gonna , check for :

    a) Heteroscedasticity

    b) test linear regression against regularized versions (Ridge/Lasso/ElasticNet)

    c) I am going to explain why regularization helped me or didn't in this project.

    d) Probably, i am gonna try to engineered such feature that could measureably improved the model .


# Features - I need and i don't need


### 🛑 The Absolute Unwanted (Drop Immediately)

* **`VIN` (Vehicle Identification Number)**
  * **Why:** This is a completely unique identifier for every single car (like a fingerprint or social security number). It has high cardinality (nearly every row will have a unique value), meaning a machine learning model can't learn any general trends from it. It's 100% noise for price prediction.

### ⚠️ The Redundant / Noise Columns (Highly Recommended to Drop)

* **`region` and `state`**
  * **Why:** While geographic location *can* slightly affect car prices (e.g., 4WDs sell better in snowy states), having both is highly redundant. `region` usually contains hundreds of specific local areas, which creates too many categories (high cardinality) and overcomplicates your model.
  * **Recommendation:** Drop `region` entirely. If you want to keep geographical context, keep only `state`, or drop both if you want a generalized country-wide model.

### 🔍 The "Proceed with Caution" Columns (Keep, but clean up)

* **`model`**
  * **Why it's tricky:** Car models (e.g., "Camry", "Mustang", "F-150") are *incredibly* important for price. However, this column usually contains thousands of messy, misspelled, or overly specific text entries (e.g., "Civic EX-L", "Civic LX").
  * **Recommendation:** Don't drop it immediately, but it will require heavy text cleaning or target encoding. If it's too messy and your model is overfitting, you might rely purely on `manufacturer` and `type`.
* **`paint_color`**
  * **Why it's borderline:** Unless a car is a wildly unpopular color, a white Camry and a silver Camry generally cost the same. It has very low predictive power and mostly adds unnecessary complexity.
  * **Recommendation:** Safe to drop if you want a leaner model, or group rare colors into an "Other" category.

### The Gold Standard (Definitely Keep)

For context, these are your heavy hitters that you **must** keep for an accurate price prediction:

* **`year` and `odometer`:** The ultimate indicators of age and wear.
* **`manufacturer`, `cylinders`, `fuel`, `transmission`, `type`:** These define the core mechanics and class of the vehicle.
* **`condition` and `title_status`:** Critical for identifying depreciated assets (like a "salvage" title or a car listed in "fair" condition).


## About Feature [title_status]


The **title status** tells the legal/history condition of the vehicle.

Here’s what each means:

---

# 1. `clean`

A normal vehicle title.

Meaning:

* No major legal or insurance issues recorded.
* Car was not declared total loss.

This is usually the most desirable condition.

---

# 2. `rebuilt`

The car was previously damaged badly and declared a total loss (`salvage`), but later repaired and inspected for road use.

Meaning:

* Car had serious damage before.
* Someone repaired it.
* Now legally drivable again.

Usually cheaper than clean-title cars.

---

# 3. `lien`

There is still a loan/legal financial claim on the car.

Meaning:

* Owner may still owe money to a bank/lender.
* Transfer/sale can have legal complications.

---

# 4. `salvage`

The vehicle was heavily damaged and declared a total loss by insurance.

Possible causes:

* accident,
* flood,
* fire,
* severe damage.

Usually:

* unsafe or expensive to repair,
* very low resale value.

Important feature for price prediction.

---

# 5. `missing`

Title information is unavailable or unknown.

Could mean:

* missing data,
* undocumented history.

You may treat this carefully during preprocessing.

---

# 6. `parts only`

Vehicle is not intended for road use anymore.

Meaning:

* sold only for spare parts,
* usually non-functional or irreparable.

These cars often have extremely low prices.

# **TEXT PREPROCESSING :**

### Step 1: Clean the Strings (Standardization)

Before encoding, reduce the sheer number of unique values by cleaning up the typos and variations. If `F-150`, `f150`, and `f-150 ` are treated as different things, cleaning them solves half your problem.


### Step 2: Group the "Rare" Models (Frequency Thresholding)

Look at your output: thousands of models only appear **1 time** (like `gand wagoneer`). A machine learning model cannot learn a price pattern from a car it only sees once.

Group all models that appear less than a certain threshold (e.g., less than 10 or 20 times) into a catch-all category called `'other'`. This will instantly plummet your unique count from 12,000+ to a few hundred.


## Target Encoding 

#### 1) df['model'] :


To understand why Target Encoding is best, look at why the alternatives fall short for this specific column:

#### 1. One-Hot Encoding (Bad Choice)

* **What it does:** Creates a separate column for every single unique car model (1 or 0).
* **Why it fails here:** Even after cleaning, you likely have hundreds of unique car models left. Creating 500+ new columns creates a massive, sparse matrix that consumes excessive memory, slows down training, and triggers the "curse of dimensionality," severely harming model performance.

#### 2. Label Encoding / Ordinal Encoding (Bad Choice)

* **What it does:** Assigns a random arbitrary integer to each model (e.g., `camry = 1`, `f150 = 2`, `accord = 3`).
* **Why it fails here:** Algorithms naturally assume that numbers have a sequence or order (i.e., **3**>**2**>**1**). Label encoding forces your model to assume that an Accord is mathematically "greater than" a Camry and twice as large as a F-150, which makes no sense and misleads the machine learning algorithm.

### ⚠️ One Critical Rule: Prevent Target Leakage!

Because Target Encoding uses the `price` (the dependent variable) to calculate the encoded values, you must compute the encoding **only on your Training Split** and then map those calculated values onto your Test Split.

If you encode the entire dataset before splitting, your model will subtly "cheat" by looking at the test set prices during training, leading to overly optimistic training scores but terrible performance on real-world data. The `category_encoders` library handles this smoothly when you use `.fit_transform()` on training data and `.transform()` on testing data.


### 2) df['Manucfacturer']

##### Extract only the numbers from the text (e.g., "6 cylinders" -> 6)

    code: 	df['cylinders'] = df['cylinders'].str.extract(r'(\d+)').astype(float)

Explanation  : 

a) r = this represent raw strings

b) \d = represents any digit from 0 to 9

c) +  = represent one or more . For eg : if given 10 or 12 cylinders then it will extract both 10 and 12 not just beginning one only

d)astype(float) :

Even though the text `"6 cylinders"` has been cut down to `"6"`, Python still thinks it is a text string (like a word), not a number. You cannot calculate averages or correlations with text.

* `.astype(float)` converts that text string `"6"` into an actual decimal number: `6.0`.

## **Important note : [during OneHotEncoding]**

💡 Why `index=X_train.index` is critical

When Scikit-Learn outputs `fit_transform()`, it completely strips away your Pandas index and returns a raw, unindexed NumPy array starting back at row `0`. If your `X_train` rows have been shuffled or split, missing the `index=X_train.index` step will cause the rows to misalign when you run `pd.concat()`, filling your dataset with broken `NaN` values!


## **NOTE : Implementation of Column transformer to reduce code** 

💡 Why this is much better

1. **Massive Code Reduction:** Your manual code requires about 10 lines of matrix manipulation per column. If you added `'type'` and `'transmission'` manually, you would have written nearly 30 lines of error-prone code. The `ColumnTransformer` handles all of them simultaneously.
2. **`remainder='passthrough'` is the Secret Sauce:** Without this parameter, `ColumnTransformer` would drop any column you didn't explicitly encode. Adding `remainder='passthrough'` tells it to process the categorical columns and leave your numeric columns (like `year` or `odometer`) completely alone, gluing everything back together automatically.
