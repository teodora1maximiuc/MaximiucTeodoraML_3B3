import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# incarcam datele din fisier
def load_data(file_path):
    data = pd.read_excel(file_path)
    data['Data'] = pd.to_datetime(data['Data'], format="%d-%m-%Y %H:%M:%S")  #convertire date
    data['Month'] = data['Data'].dt.month
    data['Year'] = data['Data'].dt.year
    data['Day'] = data['Data'].dt.day
    data['Hour'] = data['Data'].dt.hour
    data['Minute'] = data['Data'].dt.minute
    data['Second'] = data['Data'].dt.second
    return data

file_path = "C:\\Users\\Raluci\\OneDrive\\Desktop\\python\\ML-AP1\\MAXIMIUCTEODORAML_3B3\\AP1\\Grafic_SENwithDec.xlsx"
df = load_data(file_path)

features = ['Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]',
            'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]', 'Hour', 'Day', 'Month']
target = 'Sold[MW]' #targetul dupa care vrem sa predicitia

#set antrenament
train_df = df[df['Month'] != 12 & (df['Year'] != 2024)]
#set testare (luna decembrie)
test_df = df[(df['Month'] == 12) & (df['Year'] == 2024)]

X_train = train_df[features]  #caracteristici antrenament
y_train = train_df[target]  #tinta antrenament
#la fel pt test
X_test = test_df[features]
y_test = test_df[target]

#normalizare date (intre 0 si 1)
scaler = MinMaxScaler()

# scalare date
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# antrenare id3 si bayes
model_id3 = DecisionTreeRegressor(random_state=42, max_depth=6) 
model_id3.fit(X_train_scaled, y_train)

model_nb = GaussianNB()
model_nb.fit(X_train_scaled, y_train)

# predictii decembrie 2024
y_pred_id3 = model_id3.predict(X_test_scaled)
y_pred_nb = model_nb.predict(X_test_scaled)

# RMSE, MAE si R-squared
rmse_id3 = np.sqrt(mean_squared_error(y_test, y_pred_id3)) 
mae_id3 = mean_absolute_error(y_test, y_pred_id3) 

rmse_nb = np.sqrt(mean_squared_error(y_test, y_pred_nb))
mae_nb = mean_absolute_error(y_test, y_pred_nb) 

# calc. sold total si sold total prezis
total_sold = y_test.sum()
total_predicted_sold_id3 = y_pred_id3.sum() 
total_predicted_sold_nb = y_pred_nb.sum() 

# R-squared
r2_id3 = r2_score(y_test, y_pred_id3)
r2_nb = r2_score(y_test, y_pred_nb)

print(f'ID3 - RMSE: {rmse_id3}, MAE: {mae_id3}, R-squared: {r2_id3}')
print(f'Bayes naiv - RMSE: {rmse_nb}, MAE: {mae_nb}, R-squared: {r2_nb}')
print(f'Sold total: {total_sold}')
print(f'Sold total prezis (ID3): {total_predicted_sold_id3}')
print(f'Sold total prezis (Naive Bayes): {total_predicted_sold_nb}')

# acuratete
accuracy_id3 = 100 - (abs(total_sold - total_predicted_sold_id3) / total_sold) * 100 
accuracy_nb = 100 - (abs(total_sold - total_predicted_sold_nb) / total_sold) * 100

print(f'Acuratete ID3: {accuracy_id3:.2f}%')
print(f'Acuratete Bayes naiv: {accuracy_nb:.2f}%')

december_dates = test_df['Data'] 
y_pred_id3_december = y_pred_id3 
y_pred_nb_december = y_pred_nb
y_test_december = y_test 

plt.figure(figsize=(10, 6))
plt.plot(december_dates, y_test_december, label='Actual Sold [MW]', color='blue', linestyle='-', marker='o') 
plt.plot(december_dates, y_pred_id3_december, label='Sold prezis (ID3) [MW]', color='red', linestyle='--', marker='x') 
plt.plot(december_dates, y_pred_nb_december, label='Sold prezis (Bayes naiv) [MW]', color='green', linestyle='--', marker='s')

plt.title('Sold real si prezis pentru Decembrie 2024')
plt.xlabel('Data')
plt.ylabel('Sold [MW]')
plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
