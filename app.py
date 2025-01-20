from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['POST'])
def form():
    model = pickle.load(open('model.pkl', 'rb'))
    try:            
        # Guardar las variables ingresadas en un diccionario variable:valor ingresado
        features = [
            'loan_amnt', 'term', 'int_rate', 'installment', 'sub_grade', 'emp_length',
            'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
            'revol_bal', 'revol_util', 'collections_12_mths_ex_med', 'acc_now_delinq',
            'tot_coll_amt', 'tot_cur_bal', 'home_ownership', 'verification_status', 'purpose'
        ]

        form_data = {feature: request.form.get(feature) for feature in features}

        for feature in features:
            if feature not in ['sub_grade','home_ownership', 'verification_status', 'purpose']:
                form_data[feature] = float(form_data[feature].replace(',', '.'))
        
        # Transformación de sub_grade
        sub_grade = request.form.get('sub_grade')

        letter_value = {
            'A': 7,
            'B': 6,
            'C': 5,
            'D': 4,
            'E': 3,
            'F': 2,
            'G': 1
        }

        number_value = {
            '1': 0.8,
            '2': 0.6,
            '3': 0.4,
            '4': 0.2,
            '5': 0.0
        }
        
        letter = sub_grade[0].upper()
        number = sub_grade[1]

        sub_grade = letter_value[letter] + number_value[number]

        form_data['sub_grade'] = sub_grade

        # Transformación de las variables que pasan por One Hot 
        categorical_features = {
            'home_ownership': ['MORTGAGE', 'NONE', 'OTHER', 'OWN', 'RENT'],
            'verification_status': ['Source Verified', 'Verified'],
            'purpose': [
                'credit_card', 'debt_consolidation', 'educational', 'home_improvement', 'house',
                'major_purchase', 'medical', 'moving', 'other', 'renewable_energy', 'small_business',
                'vacation', 'wedding'
            ]
        }

        encoded_features = []

        # La opción elegida en cada categoría se codifica como 1 y el resto como 0
        for feature, value in categorical_features.items():
            for category in value:
                encoded_features.append(1 if form_data[feature] == category else 0)

        # Guardar valores para la predicción
        numerical_features = [
            form_data['loan_amnt'], form_data['term'], form_data['int_rate'], form_data['installment'],
            form_data['sub_grade'], form_data['emp_length'], form_data['annual_inc'], form_data['dti'],
            form_data['delinq_2yrs'], form_data['inq_last_6mths'], form_data['open_acc'], form_data['pub_rec'],
            form_data['revol_bal'], form_data['revol_util'], form_data['collections_12_mths_ex_med'],
            form_data['acc_now_delinq'], form_data['tot_coll_amt'], form_data['tot_cur_bal']
        ]

        X = np.array(numerical_features + encoded_features).reshape(1, -1)

        try:
            prediction = model.predict(X)[0]
            return jsonify(prediction=prediction)
        except Exception as e:
            return jsonify(error="Prediction error: " + str(e))

    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run()