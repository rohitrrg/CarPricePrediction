from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    present_price = float(request.form['present_price'])
    kms_driven = int(request.form['kms_driven'])
    year = int(request.form['year'])
    age = 2020 - year
    owners = int(request.form['owners'])

    fuel_type = request.form['fuel_type']
    if fuel_type == 'diesel':
        fuel_type = [0, 1, 0]
    elif fuel_type == 'petrol':
        fuel_type = [0, 0, 1]
    elif fuel_type == 'cng':
        fuel_type = [1, 0, 0]

    seller_type = request.form['seller_type']
    if seller_type == 'dealer':
        seller_type = [1, 0]
    elif seller_type == 'individual':
        seller_type = [0, 1]

    gear_transmission = request.form['gear_transmission']
    if gear_transmission == 'manual':
        gear_transmission = [0, 1]
    elif gear_transmission == 'automatic':
        gear_transmission = [1, 0]

    data = [present_price, kms_driven, owners, age]+fuel_type+gear_transmission+seller_type
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    output = round(prediction[0])

    return render_template('home.html', prediction_text='The Selling Price of the car is ₹ {} Lakhs only.'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
