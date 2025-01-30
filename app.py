from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from model import ScorecardModel # Necesario para deserializar el modelo
from PIL import Image, ImageDraw

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['POST'])
def form():
    
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    try:            
        # Guardar las variables ingresadas en un diccionario variable:valor ingresado
        features = [
            'loan_amnt', 'term', 'int_rate', 'installment', 'sub_grade', 'emp_length',
            'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
            'revol_bal', 'revol_util', 'tot_coll_amt', 'tot_cur_bal', 'home_ownership', 
            'verification_status', 'purpose'
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
                encoded_features.append(1 if form_data.get(feature) == category else 0)

        # Guardar valores para la predicción
        numerical_features = [
            form_data['loan_amnt'], form_data['term'], form_data['int_rate'], form_data['installment'],
            form_data['sub_grade'], form_data['emp_length'], form_data['annual_inc'], form_data['dti'],
            form_data['delinq_2yrs'], form_data['inq_last_6mths'], form_data['open_acc'], form_data['pub_rec'],
            form_data['revol_bal'], form_data['revol_util'], form_data['tot_coll_amt'], form_data['tot_cur_bal']
        ]

        # Escalar los valores numéricos
        numerical_features_scaled = scaler.transform([numerical_features])[0]

        X = np.array(numerical_features_scaled.tolist() + encoded_features).reshape(1, -1)
        
        try:
            prediction = model.predict(X)
            score, risk_category, decision = prediction
            
            # Llamamos a la función para guardar la imagen con el puntaje
            save_score_on_image(score[0][0])

            # Pasar los resultados a la plantilla
            return render_template('result.html', score=score[0][0], risk_category=risk_category, decision=decision)
        
        except Exception as e:
            return render_template('result.html', error="Prediction error: " + str(e))

    except Exception as e:
        return render_template('result.html', error=str(e))

# Función para guardar la imagen con el punto
def save_score_on_image(user_score):
    image_path = 'static/images/scorecard_plot.png'
    img = Image.open(image_path)
    image_width, image_height = img.size

    # Definir el rango del puntaje (por ejemplo, 300-850)
    min_score = 300
    max_score = 850

    # Márgenes izquierdo y derecho
    margin_left = 198  # Ancho del margen izquierdo en píxeles
    margin_right = 113  # Ancho del margen derecho en píxeles

    # Calcular el ancho útil, excluyendo los márgenes izquierdo y derecho
    useful_width = image_width - margin_left - margin_right

    # Calcular la posición del puntaje
    score_range = max_score - min_score
    score_position_x = int((user_score - min_score) / score_range * useful_width ) + margin_left

    # Mover el punto hacia arriba ajustando la posición 'y'
    point_y_position = int(image_height * 0.6)

    # Crear el objeto para dibujar sobre la imagen
    draw = ImageDraw.Draw(img)
    point_color = (255, 0, 0)  # Color rojo
    point_radius = 10  # Radio del punto

    # Dibujar el punto sobre la imagen
    draw.ellipse((score_position_x - point_radius, point_y_position - point_radius, 
                  score_position_x + point_radius, point_y_position + point_radius), 
                 fill=point_color)

    # Guardar la imagen con el punto
    output_image_path = 'static/scorecard_with_point.png'
    img.save(output_image_path)

if __name__ == '__main__':
    app.run()