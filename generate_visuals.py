import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import os

# Set style
sns.set_theme(style="whitegrid")

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

# Load data
df = pd.read_csv('CarPrice.csv')

# Preprocessing
if 'car_ID' in df.columns:
    df.drop('car_ID', axis=1, inplace=True)
df['Brand'] = df['CarName'].apply(lambda x: str(x).split(' ')[0].lower())
brand_mapping = {
    'maxda': 'mazda', 'porcshce': 'porsche', 'toyouta': 'toyota', 
    'vokswagen': 'volkswagen', 'vw': 'volkswagen', 'nissan': 'nissan',
    'alfa-romero': 'alfa-romeo'
}
df['Brand'] = df['Brand'].replace(brand_mapping)
df.drop('CarName', axis=1, inplace=True)

# Visual 1: Distribution of Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True, color='teal')
plt.title('Distribution of Car Prices', fontsize=15)
plt.xlabel('Price ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('outputs/price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Visual 2: Brand vs Average Price
plt.figure(figsize=(12, 6))
brand_price = df.groupby('Brand')['price'].mean().sort_values(ascending=False)
sns.barplot(x=brand_price.index, y=brand_price.values, palette='viridis')
plt.title('Average Car Price by Brand', fontsize=15)
plt.xticks(rotation=45)
plt.xlabel('Brand', fontsize=12)
plt.ylabel('Average Price ($)', fontsize=12)
plt.savefig('outputs/brand_price_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Visual 3: Predicted vs Actual (Need to train a quick model)
X = df.drop('price', axis=1)
y = df['price']
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='darkorange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs Predicted Car Prices', fontsize=15)
plt.xlabel('Actual Price ($)', fontsize=12)
plt.ylabel('Predicted Price ($)', fontsize=12)
plt.savefig('outputs/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visuals generated successfully in 'outputs/' folder.")
