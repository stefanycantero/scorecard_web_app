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
        if request.method == 'POST':
            c1 = float(request.form.get('c1'))
            c2 = float(request.form.get('c2'))
            c3 = float(request.form.get('c3'))
            c4 = float(request.form.get('c4'))
            c5 = float(request.form.get('c5'))
            c6 = float(request.form.get('c6'))
            c7 = float(request.form.get('c7'))

            data = np.array([[c1, c2, c3, c4, c5, c6, c7]])
            prediction = model.predict(data)[0][0]
            return jsonify(prediction=prediction)
    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run()