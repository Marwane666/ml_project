from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Charger le modèle KMeans pré-entraîné
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# Charger le scaler pré-entraîné
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Charger le PCA pré-entraîné
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

# Définir les colonnes attendues
expected_columns = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium',
                    'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols',
                    'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']

def load_and_predict(input_data):
    """
    Prédit les clusters pour de nouvelles données en utilisant le modèle pré-entraîné.
    """
    # Vérifier les colonnes et réorganiser
    if not all(column in input_data.columns for column in expected_columns):
        missing_cols = set(expected_columns) - set(input_data.columns)
        raise ValueError(f"Les données d'entrée doivent contenir les colonnes suivantes : {missing_cols}")

    input_data = input_data[expected_columns]

    # Prétraiter les données d'entrée
    input_data_scaled_array = scaler.transform(input_data)
    input_data_scaled = pd.DataFrame(input_data_scaled_array, columns=expected_columns)

    # Appliquer le PCA
    pca_input_data_array = pca.transform(input_data_scaled)
    pca_columns = [f'c{i+1}' for i in range(pca.n_components_)]
    pca_input_data = pd.DataFrame(pca_input_data_array, columns=pca_columns)

    # Prédire les clusters
    cluster_labels = kmeans_model.predict(pca_input_data)

    return cluster_labels

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        data = {
            'Alcohol': [float(request.form['Alcohol'])],
            'Malic_Acid': [float(request.form['Malic_Acid'])],
            'Ash': [float(request.form['Ash'])],
            'Ash_Alcanity': [float(request.form['Ash_Alcanity'])],
            'Magnesium': [float(request.form['Magnesium'])],
            'Total_Phenols': [float(request.form['Total_Phenols'])],
            'Flavanoids': [float(request.form['Flavanoids'])],
            'Nonflavanoid_Phenols': [float(request.form['Nonflavanoid_Phenols'])],
            'Proanthocyanins': [float(request.form['Proanthocyanins'])],
            'Color_Intensity': [float(request.form['Color_Intensity'])],
            'Hue': [float(request.form['Hue'])],
            'OD280': [float(request.form['OD280'])],
            'Proline': [float(request.form['Proline'])]
        }

        input_data = pd.DataFrame(data)

        try:
            cluster_labels = load_and_predict(input_data)
            cluster_label = cluster_labels[0]
            return render_template('result.html', cluster_label=cluster_label)
        except ValueError as e:
            error_message = str(e)
            return render_template('form.html',expected_columns=expected_columns, error=error_message)
    else:
        return render_template('form.html',expected_columns=expected_columns)

if __name__ == '__main__':
    app.run(debug=True)
