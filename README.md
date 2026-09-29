# Barclays Customer Churn Analysis

## 📌 Project Overview
Barclays Bank experienced high customer churn rates and needed to replace outdated methodologies with modern, real-time analytics and predictive modeling. This project consolidates raw customer data, cleans and analyzes behavioral trends, and prepares feature sets for machine learning models to predict future churn rates.

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Wrangling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (`LogisticRegression`, `SVC`, `DecisionTreeClassifier`, `RandomForestClassifier`, `KNeighborsClassifier`, `GradientBoostingClassifier`, `AdaBoostClassifier`)
- **Utilities:** `pickle`, `warnings`

---

## 📁 Repository Structure
barclays-churn-analysis/
├── BARCLAYS_CUSTOMER_CHURN_ANALYSIS.ipynb  # Main Google Colab Jupyter Notebook
├── README.md                               # Project documentation and Day 1 progress summary
└── models/                                 # Serialized ML model weights (.pkl)


---

## 📅 Day 1 Outcomes & Progress

### 1. Data Ingestion & Integration
- Loaded **7 relational datasets** from external sources: `ActiveCustomer`, `CreditCard`, `ExitCustomer`, `Gender`, `Geography`, `Bank_Churn`, and `CustomerInfo`.
- Merged the datasets on key identifiers (`CustomerId`, `GeographyID`, `GenderID`) to form a unified dataset (`final_df`) consisting of **10,000 rows and 17 columns**.

### 2. Data Cleaning & Feature Engineering
- Identified and removed missing/null values (`dropna()`).
- Dropped irrelevant identifier columns like `RowNumber`.
- Converted `Bank DOJ` (Date of Joining) to proper datetime types and extracted temporal attributes (`Month`, `Day`).
- Mapped categorical features to numeric representations for machine learning readiness:
  - `GenderCategory` $\rightarrow$ Male: 0, Female: 1
  - `GeographyLocation` $\rightarrow$ France: 0, Spain: 1, Germany: 2
  - `Month` $\rightarrow$ Integer mapping (0–11) using Python's `calendar` library.

### 3. Exploratory Data Analysis (EDA)
- **Correlation Analysis:** Identified key predictors correlated with customer churn (`Exited`):
  - **Positive Correlation:** `Age` (+0.285), `GeographyLocation` (+0.154), `Balance` (+0.119), `GenderCategory` (+0.107).
  - **Negative Correlation:** `IsActiveMember` (-0.156), `NumOfProducts` (-0.048).
- **Univariate Analysis:** Analyzed value distributions across textual features (`Surname`, `GeographyLocation`, `GenderCategory`).
- **Bivariate Analysis:** Analyzed churn patterns across joining dates, months, and days of the week.

### 4. Machine Learning Data Preparation
- Selected features and dropped raw metadata columns (`CustomerId`, `GeographyID`, `GenderID`, `Bank DOJ`, `Surname`, `Day`).
- Separated feature matrix ($X$) and target vector ($y$) resulting in shapes $X$: `(10000, 11)` and $y$: `(10000,)`.
- Applied `train_test_split` with an **80/20 train-test ratio** (`random_state=42`):
  - `X_train`: (8000, 11) | `y_train`: (8000,)
  - `X_test`: (2000, 11) | `y_test`: (2000,)

---

## 🚀 How to Run
1. Open [`BARCLAYS_CUSTOMER_CHURN_ANALYSIS.ipynb`](BARCLAYS_CUSTOMER_CHURN_ANALYSIS.ipynb) in Google Colab or Jupyter Notebook.
2. Execute cells sequentially to load datasets directly from remote URLs, run EDA, and generate training splits.
