import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
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

def evaluate_country_ranking(country_to_analyze):
    df_country = df[df['Country'] == country_to_analyze]

    X = df_country[['Category', 'Visitors', 'Rating', 'Accommodation_Available', 'RpV']] 
    y = df_country['Revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    revenue_index = numerical_columns.index('Revenue')
    revenue_scale = scaler.scale_[revenue_index]
    revenue_mean = scaler.mean_[revenue_index]

    y_test_original = y_test * revenue_scale + revenue_mean
    y_pred_original = y_pred * revenue_scale + revenue_mean

    df_test = X_test.copy()
    df_test['Actual_Revenue'] = y_test_original
    df_test['Predicted_Revenue'] = y_pred_original
    df_test['Predicted_RpV'] = df_test['Predicted_Revenue'] / df_test['Visitors']

    category_ranking_revenue = df_test.groupby('Category')['Predicted_Revenue'].mean().sort_values(ascending=False)
    category_ranking_rpv = df_test.groupby('Category')['Predicted_RpV'].mean().sort_values(ascending=False)

    category_names = label_encoder_category.inverse_transform(category_ranking_revenue.index)

    ranking_revenue = pd.DataFrame({
        'Category': category_names,
        'Predicted_Revenue': category_ranking_revenue.values
    })
    
    ranking_rpv = pd.DataFrame({
        'Category': category_names,
        'Predicted_RpV': category_ranking_rpv.values
    })

    return ranking_revenue, ranking_rpv

for country_code in df['Country'].unique():
    country_name = label_encoder_country.inverse_transform([country_code])[0]
    print(f"\nIerarhia categoriilor pentru țara '{country_name}', pe baza veniturilor prezise și a veniturilor per vizitator:")

    category_ranking_revenue, category_ranking_rpv = evaluate_country_ranking(country_code)

    print("\nRanking pe baza veniturilor prezise:")
    print(category_ranking_revenue)

    print("\nRanking pe baza veniturilor per vizitator (RpV) prezise:")
    print(category_ranking_rpv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(category_ranking_revenue['Category'], category_ranking_revenue['Predicted_Revenue'], color='skyblue')
    ax1.set_xlabel('Categorie')
    ax1.set_ylabel('Venit Presupus (Predicted Revenue)')
    ax1.set_title(f'Ierarhia Categoriilor pentru {country_name} (Venit Presupus)')
    ax1.set_xticklabels(category_ranking_revenue['Category'], rotation=45)

    ax2.bar(category_ranking_rpv['Category'], category_ranking_rpv['Predicted_RpV'], color='salmon')
    ax2.set_xlabel('Categorie')
    ax2.set_ylabel('Venit per Vizitator Presupus (Predicted RpV)')
    ax2.set_title(f'Ierarhia Categoriilor pentru {country_name} (Venit per Vizitator Presupus)')
    ax2.set_xticklabels(category_ranking_rpv['Category'], rotation=45)

    plt.tight_layout()
    plt.show()
