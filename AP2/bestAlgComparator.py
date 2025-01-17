import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("tourism_dataset.csv")

df = df.drop(columns=['Location'])

label_encoder_country = LabelEncoder()
label_encoder_category = LabelEncoder()

df['Country'] = label_encoder_country.fit_transform(df['Country'])
df['Category'] = label_encoder_category.fit_transform(df['Category'])

df['Accommodation_Available'] = df['Accommodation_Available'].map({'Yes': 1, 'No': 0})

df['RpV'] = df['Revenue'] / df['Visitors']

scaler = StandardScaler()
numerical_columns = ["Visitors", "Revenue", "Rating", "RpV"]
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

correlation_matrix = df.corr().abs()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

X = df.drop(["Revenue"], axis=1)
y = df["Revenue"]

def evaluate_models(train_sizes):
    results = {}
    
    for train_size in train_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42)

        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(objective="reg:squarederror", random_state=42),
            'AdaBoost': AdaBoostRegressor(random_state=42),
            'Linear Regression': LinearRegression(),
            'ID3 (Decision Tree)': DecisionTreeRegressor(random_state=42),
        }
        
        model_rmse = []
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            model_rmse.append(rmse)
            
        results[train_size] = model_rmse
    
    return results

train_sizes = [0.7, 0.75, 0.8, 0.85, 0.9]

results = evaluate_models(train_sizes)

models = ['Random Forest', 'XGBoost', 'AdaBoost', 'Linear Regression', 'ID3 (Decision Tree)']
results_df = pd.DataFrame(results, index=models)

plt.figure(figsize=(10, 6))
for model in results_df.index:
    plt.plot(train_sizes, results_df.loc[model], label=model, marker='o')

plt.xlabel('Train Size')
plt.ylabel('RMSE')
plt.title('Model Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()
