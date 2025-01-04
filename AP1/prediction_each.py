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
target_consumption = 'Consum[MW]'  #target pt consum
target_production = 'Productie[MW]'  #target pt productie
# !soldul este diferenta dintre consum si productie, deci vom prezice acestea doua iar rezultatul ne va ajuta sa determinam soldul!

# set de antrenament (fara dec 2024)
train_df = df[df['Month'] != 12 & (df['Year'] != 2024)]
# set testare (dec 2024)
test_df = df[(df['Month'] == 12) & (df['Year'] == 2024)]

X_train = train_df[features]  #caracteristici antrenament
y_train_consumption = train_df[target_consumption]  # tinta pt consum
y_train_production = train_df[target_production]  # tinta pt productie
# La fel pentru test
X_test = test_df[features]
y_test_consum = test_df[target_consumption]
y_test_productie = test_df[target_production]

# normalizare date (intre 0 si 1)
scaler = MinMaxScaler()

# scalare date
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Antrenare ID3 si Bayes naiv
model_id3_consum = DecisionTreeRegressor(random_state=42, max_depth=5) 
model_id3_consum.fit(X_train_scaled, y_train_consumption)

model_id3_productie = DecisionTreeRegressor(random_state=42, max_depth=5) 
model_id3_productie.fit(X_train_scaled, y_train_production)

model_nb_consum = GaussianNB()
model_nb_consum.fit(X_train_scaled, y_train_consumption)

model_nb_productie = GaussianNB()
model_nb_productie.fit(X_train_scaled, y_train_production)

#predictii decembrie 2024
y_pred_id3_consum = model_id3_consum.predict(X_test_scaled)
y_pred_id3_productie = model_id3_productie.predict(X_test_scaled)

y_pred_nb_consum = model_nb_consum.predict(X_test_scaled)
y_pred_nb_productie = model_nb_productie.predict(X_test_scaled)

# RMSE, MAE si R-squared
rmse_id3_consum = np.sqrt(mean_squared_error(y_test_consum, y_pred_id3_consum)) 
mae_id3_consum = mean_absolute_error(y_test_consum, y_pred_id3_consum)

rmse_id3_productie = np.sqrt(mean_squared_error(y_test_productie, y_pred_id3_productie))
mae_id3_productie = mean_absolute_error(y_test_productie, y_pred_id3_productie)

rmse_nb_consum = np.sqrt(mean_squared_error(y_test_consum, y_pred_nb_consum))
mae_nb_consum = mean_absolute_error(y_test_consum, y_pred_nb_consum)

rmse_nb_productie = np.sqrt(mean_squared_error(y_test_productie, y_pred_nb_productie))
mae_nb_productie = mean_absolute_error(y_test_productie, y_pred_nb_productie)

# R-squared (cat de bine prezice valorile)
r2_id3_consum = r2_score(y_test_consum, y_pred_id3_consum)
r2_id3_productie = r2_score(y_test_productie, y_pred_id3_productie)

r2_nb_consum = r2_score(y_test_consum, y_pred_nb_consum)
r2_nb_productie = r2_score(y_test_productie, y_pred_nb_productie)

print(f'ID3 - Consum RMSE: {rmse_id3_consum}, MAE: {mae_id3_consum}, R-squared: {r2_id3_consum}')
print(f'ID3 - Productie RMSE: {rmse_id3_productie}, MAE: {mae_id3_productie}, R-squared: {r2_id3_productie}')
print(f'Naive Bayes - Consum RMSE: {rmse_nb_consum}, MAE: {mae_nb_consum}, R-squared: {r2_nb_consum}')
print(f'Naive Bayes - Productie RMSE: {rmse_nb_productie}, MAE: {mae_nb_productie}, R-squared: {r2_nb_productie}')

# soldul calculat in functie de consum si productie
sold_id3 = y_pred_id3_consum - y_pred_id3_productie  # ID3
sold_nb = y_pred_nb_consum - y_pred_nb_productie  #Bayes naiv

# soldul real
real_sold = y_test_consum - y_test_productie

# acuratetea sold
def calculate_accuracy(real_sold, predicted_sold):
    difference = np.abs(real_sold - predicted_sold)
    
    # cazul daca prezicerea e mai mare
    accuracy = 100 - ((difference / real_sold) * 100)
    
    # pt acuratete negativa
    accuracy[accuracy > 100] = 100
    accuracy[accuracy < 0] = 0
    
    return np.mean(accuracy)

accuracy_sold_id3 = calculate_accuracy(real_sold, sold_id3)
accuracy_sold_nb = calculate_accuracy(real_sold, sold_nb)

print(f'Acuratete sold ID3: {accuracy_sold_id3:.2f}%')
print(f'Acuratete sold Bayes naiv: {accuracy_sold_nb:.2f}%')

# grafic date
december_dates = test_df['Data']

plt.figure(figsize=(10, 6))
plt.plot(december_dates, real_sold, label='Sold real [MW]', color='blue', linestyle='-', marker='o')
plt.plot(december_dates, sold_id3, label='Sold prezis (ID3) [MW]', color='red', linestyle='--', marker='x')
plt.plot(december_dates, sold_nb, label='Sold prezis (Bayes naiv) [MW]', color='green', linestyle='--', marker='s')

plt.title('Sold real si prezis pt Decembrie 2024')
plt.xlabel('Data')
plt.ylabel('Sold [MW]')
plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
