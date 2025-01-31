# Aplicativo Web para Cálculo de Puntaje Crediticio y Aprobación de Créditos

Este repositorio contiene una aplicación web desarrollada con Flask, HTML, CSS y JavaScript que permite a los usuarios calcular su puntaje crediticio y evaluar la posibilidad de obtener un crédito. La aplicación utiliza un modelo de Machine Learning con TensorFlow y redes neuronales, entrenado con el dataset credit-risk-dataset de Kaggle, para realizar predicciones sobre el riesgo crediticio y la aprobación de créditos. 

Equipo 01 conformado por:

Verónica Pérez Zea
Sebastian Aguinaga Velasquez
Stefany Cantero Cárdenas
María Del Pilar Mira Londoño

## Funcionalidades

*   **Cálculo de Puntaje Crediticio:** El usuario ingresa información personal y financiera relevante, como monto del crédito, plazo, tasa de interés, ingresos, historial de empleo y deudas, entre otros. La aplicación calcula el puntaje crediticio utilizando un modelo entrenado para calcular la probabilidad de incumplimiento del pago del crédito.
*   **Evaluación de Riesgo Crediticio:** Basándose en el puntaje crediticio la aplicación determina la categoría de riesgo del cliente (bajo, moderado, alto).
*   **Decisión de Crédito:** La aplicación muestra una recomendación sobre la aprobación o rechazo del crédito (bajo, en revisión, moderado, rechazado).
*   **Visualización Gráfica:** Se presenta una gráfica que muestra la ubicación del cliente en relación con la población en términos de puntaje crediticio, lo que permite una mejor comprensión de su situación financiera.

## Tecnologías Utilizadas

*   **Frontend:** HTML, CSS, JavaScript
*   **Backend:** Flask (Python)
*   **Modelo de Machine Learning:** TensorFlow, Keras (redes neuronales)
*   **Librerías de Preprocesamiento y Entrenamiento:**
    *   NumPy
    *   Pandas
    *   Matplotlib
    *   Seaborn
    *   Scikit-learn
*   **Dataset:** credit-risk-dataset de Kaggle

## Estructura del Proyecto
├── app.py          # Archivo principal de la aplicación Flask  

├── templates/      # Archivos HTML para las plantillas web  

│   ├── index.html  

│   └── result.html  

├── static/         # Archivos CSS e imágenes   

│   ├── images

│   ├── style.css

│   └── result_style.css

├── requirements.txt # Lista de dependencias del proyecto

├── model.pkl        # Archivo .pkl con el modelo entrenado

├── scaler.pkl       # Archivo .pkl con el scaler del modelo

├── model.py         # Archivo .py con la clase que permite cambiar la escala del resultado

└── README.md       # Este archivo
