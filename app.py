from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['POST'])
def form():

    """
    0   loan_amnt                            887379 non-null  float64
    1   term                                 887379 non-null  int64  
    2   int_rate                             887379 non-null  float64
    3   installment                          887379 non-null  float64
    4   sub_grade                            887379 non-null  float64
    5   emp_length                           887379 non-null  int32  
    6   annual_inc                           887379 non-null  float64
    7   loan_status                          887379 non-null  int32  
    8   dti                                  887379 non-null  float64
    9   delinq_2yrs                          887379 non-null  float64
    10  inq_last_6mths                       887379 non-null  float64
    11  open_acc                             887379 non-null  float64
    12  pub_rec                              887379 non-null  float64
    13  revol_bal                            887379 non-null  float64
    14  revol_util                           887379 non-null  float64
    15  collections_12_mths_ex_med           887379 non-null  float64
    16  acc_now_delinq                       887379 non-null  float64
    17  tot_coll_amt                         887379 non-null  float64
    18  tot_cur_bal                          887379 non-null  float64
    19  home_ownership_MORTGAGE              887379 non-null  float64
    20  home_ownership_NONE                  887379 non-null  float64
    21  home_ownership_OTHER                 887379 non-null  float64
    22  home_ownership_OWN                   887379 non-null  float64
    23  home_ownership_RENT                  887379 non-null  float64
    24  verification_status_Source Verified  887379 non-null  float64
    25  verification_status_Verified         887379 non-null  float64
    26  purpose_credit_card                  887379 non-null  float64
    27  purpose_debt_consolidation           887379 non-null  float64
    28  purpose_educational                  887379 non-null  float64
    29  purpose_home_improvement             887379 non-null  float64
    30  purpose_house                        887379 non-null  float64
    31  purpose_major_purchase               887379 non-null  float64
    32  purpose_medical                      887379 non-null  float64
    33  purpose_moving                       887379 non-null  float64
    34  purpose_other                        887379 non-null  float64
    35  purpose_renewable_energy             887379 non-null  float64
    36  purpose_small_business               887379 non-null  float64
    37  purpose_vacation                     887379 non-null  float64
    38  purpose_wedding                      887379 non-null  float64
    """

    model = pickle.load(open('model.pkl', 'rb'))
    try:            
        features = [
            'loan_amnt', 'term', 'int_rate', 'installment', 'sub_grade', 'emp_length',
            'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
            'revol_bal', 'revol_util', 'collections_12_mths_ex_med', 'acc_now_delinq',
            'tot_coll_amt', 'tot_cur_bal', 'home_ownership', 'verification_status', 'purpose'
        ]

        # Guardar las variables ingresadas en un diccionario variable:valor ingresado
        form_data = {feature: request.form.get(feature) for feature in features}

        for feature in features:
            if feature not in ['home_ownership', 'verification_status', 'purpose']:
                form_data[feature] = float(form_data[feature])

        # Transformación de las variables sub_grade y las que pasan por One Hot 
        categorical_features = {
            'sub_grade': ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 
                        'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 
                        'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 
                        'G1', 'G2', 'G3', 'G4', 'G5'],
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

        data = np.array([list(form_data.values())[:17] + encoded_features])

        prediction = model.predict(data)[0]

        return jsonify(prediction=prediction)

    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run()