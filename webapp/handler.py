import pandas as pd
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann
import pickle
import os

# Carregando model
model = pickle.load(open('model/model_xgb_tuned.pkl', 'rb'))


# Inicializando o API
app = Flask(__name__)  # instanciando o Flask

# Criar o endpoint


@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json()

    if test_json:  # Tem os dados
        if isinstance(test_json, dict):  # Veio único, uma linha - Unique Example
            test_raw = pd.DataFrame(test_json, index=[0])

        else:  # Multiples Examples
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

    else:
        return Response('{}', status=200, mimetype='application/json')

    # Instanciando a Classe Rossmann
    pipeline = Rossmann()

    # Limpeza de dados
    df1 = pipeline.data_cleaning(test_raw)

    # Crianção de Features
    df2 = pipeline.feature_engeneering(df1)

    # Prepatação dos dados
    df3 = pipeline.data_preparation(df2)

    # Predição dos dados
    df_response = pipeline.get_prediction(model, test_raw, df3)

    return df_response


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)  # Indica o local hold
