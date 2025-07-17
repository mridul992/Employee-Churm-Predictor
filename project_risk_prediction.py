
# 1. Import Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 2. Load the dataset
df = pd.read_csv('/mnt/data/dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()

# 3. Exploratory Data Analysis
print(df.info())
print(df.describe())
print(df['Attrition'].value_counts())
sns.countplot(data=df, x='Attrition')
plt.show()

# 4. Data Cleaning
## Removing Irrelevant Columns
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

## Keeping Only Consistent Data (e.g., dropping if any rows missing)
df.dropna(inplace=True)

## Plotting Correlation Matrix
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.show()

# 5. Data Preprocessing and Encoding
## Label Encoding binary fields
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])  # Yes=1, No=0

## Define features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

## Categorical columns
cat_cols = X.select_dtypes(include='object').columns.tolist()

## ColumnTransformer for encoding
ct = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(drop='first'), cat_cols)
], remainder='passthrough')

# 6. Model Building (through pipelines)

# 7. Prepare Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. ML Algorithms

models = {
    'Decision Tree': DecisionTreeRegressor(),
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Support Vector Machine': SVR(),
    'XGBoost': XGBRegressor()
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('transformer', ct),
        ('regressor', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = (mse, r2)
    print(f"\n{name} -> MSE: {mse:.4f}, R2: {r2:.4f}")

# 9. Best Model
best_model_name = max(results.items(), key=lambda x: x[1][1])[0]
print(f"\n✅ Best Model: {best_model_name}")

# 10. Predictions (Test 1 to Test 4)
final_pipe = Pipeline([
    ('transformer', ct),
    ('regressor', models[best_model_name])
])
final_pipe.fit(X, y)

print("\nTest Predictions:")
print(final_pipe.predict(X.iloc[0:4]))

# 11. Export Model
joblib.dump(final_pipe, '/mnt/data/project_risk_model.pkl')
