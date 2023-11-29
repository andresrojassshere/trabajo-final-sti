from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from IPython.display import clear_output
import ipywidgets as widgets
from IPython.display import display

app = Flask(__name__)


tiendas_df = pd.read_csv('tiendas.csv')


# Normalizar características de la db
max_values = tiendas_df.iloc[:, 1:].max()
tiendas_df.iloc[:, 1:] /= max_values


def calcular_similitud(usuario_features, database_data):
    # Normalizar características de usuario
    usuario_features_normalized = {key: value / max_values[key] for key, value in usuario_features.items()}

    # Agregar usuario actual a la matriz de la db
    usuario_features_array = [usuario_features_normalized[key] for key in database_data.columns[1:]]
    database_features = np.vstack([database_data.iloc[:, 1:].values, usuario_features_array])


    similarity_scores = cosine_similarity(database_features)
    user_similarity = similarity_scores[-1, :-1]


    # Indices de las tiendas
    top_indices = np.argsort(user_similarity)[::-1][:3]

    # Devolver las tiendas mas similares y porcentajes
    similar_stores = []
    for index in top_indices:
        similarity_percentage = user_similarity[index] * 100
        store_name = database_data['nombre'][index]
        similar_stores.append((store_name, similarity_percentage))

    return similar_stores

# Ruta para los sliders
@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('sliders.html')

# Ruta para manejar la solicitud de resultados
@app.route('/resultados', methods=['POST'])
def resultados():
    # Obtener datos de los sliders
    color = int(request.form['color'])
    estilo = int(request.form['estilo'])
    textura = int(request.form['textura'])
    ajuste = int(request.form['ajuste'])
    tendencia = int(request.form['tendencia'])

    # Características del usuario
    usuario_features = {
        'color_preferido': color,
        'estilo_preferido': estilo,
        'textura_preferida': textura,
        'ajuste_ropa': ajuste,
        'importancia_tendencia': tendencia
    }

    
    resultados_similares = calcular_similitud(usuario_features, tiendas_df)

    # Renderizar pantalla de resultados
    return render_template('resultados.html', resultados=resultados_similares)

if __name__ == '__main__':
    app.run(debug=True)
