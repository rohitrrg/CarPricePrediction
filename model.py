import pandas as pd
from xgboost import XGBRegressor
import pickle

data = pd.read_csv('dataset/car data.csv')

# The column car name doesn't seem to add much value to our analysis and hence dropping the column
data = data.drop('Car_Name', axis=1)

# It's important to know how many years old the car is.
data['Car_age'] = 2020-data['Year']
data.drop('Year', axis=1, inplace=True)

fuel = pd.get_dummies(data['Fuel_Type'])
transmission = pd.get_dummies(data['Transmission'])
seller = pd.get_dummies(data['Seller_Type'])

data.drop(['Fuel_Type', 'Transmission', 'Seller_Type'], axis=1, inplace=True)

data_final = pd.concat([data, fuel, transmission, seller], axis=1)


X = data_final.iloc[:, 1:]
y = data_final.iloc[:, 0]


model = XGBRegressor()
model.fit(X.values, y.values)

pickle.dump(model, open('model.pkl', 'wb'))
