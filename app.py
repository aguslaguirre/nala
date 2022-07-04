from flask import Flask, make_response, request
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle
import json





app = Flask(__name__)

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def form():
    return """
        <html>
                <h2>Modelo de detección de Fraudes NALA.</h2>
            <body>
                <p> Comparta su archivo CSV.</p>
                <form action="/transform" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" class="btn btn-block"/>
                    </br>
                    </br>
                    <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
                </form>
            </body>
        </html>
    """
@app.route('/transform', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    #print("file contents: ", file_contents)
    #print(type(file_contents))
    print(csv_input)
    for row in csv_input:
        print(row)

    stream.seek(0)
    result = transform(stream.read())

    df = pd.read_csv(StringIO(result), sep=";")
  
    df.fraude.replace(True,1,inplace=True)
    df.fraude.replace(False,0,inplace=True)
    df.is_prime.replace(True,1,inplace=True)
    df.is_prime.replace(False,0,inplace=True)
    df.genero.replace('F',0,inplace=True)
    df.genero.replace('M',1,inplace=True)
    df.genero.replace('--',0,inplace=True)

    df.dispositivo = df.dispositivo.str.replace(';', ',', regex = False)
    df.dispositivo = df.dispositivo.str.replace("'", '"', regex = False)
    y = json.loads(df.dispositivo[1])
    type(y)
    df.dispositivo = df.dispositivo.apply(lambda  x : json.loads(x) if x != np.nan else x)


    df['dispositivo_model'] = df.dispositivo.apply(lambda x : x['model'])
    df['dispositivo_divice_score'] = df.dispositivo.apply(lambda x : x['device_score'])
    df['dispositivo_os'] = df.dispositivo.apply(lambda x : x['os'])

    df1=df
    df1['fecha']=df['fecha']
    df1['fecha'] =  pd.to_datetime(df1['fecha'], infer_datetime_format=True)
    df['dia'] = df1['fecha'].apply(lambda x: x.day_name())


    if(df['monto'].dtypes=='object'):
        df.monto = df.monto.str.replace(',', '.', regex = False) 
        df['monto'] = df['monto'].astype(float) 
    if(df['dcto'].dtypes=='object'):
        df.dcto = df.dcto.str.replace(',', '.', regex = False) 
        df['dcto'] = df['dcto'].astype(float) 
    if(df['cashback'].dtypes=='object'):
        df.cashback = df.cashback.str.replace(',', '.', regex = False) 
        df['cashback'] = df['cashback'].astype(float) 
    if(df['ID_USER'].dtypes!='object'):
        df['ID_USER'] = df['ID_USER'].astype(int) 
    if(df['fecha'].dtypes!='object'):
        df['fecha'] = df['fecha'].astype(str) 
        
    df= df.drop(['dispositivo','fecha'], axis=1)
    df['fraude'].value_counts(normalize = True)
    df = pd.get_dummies(df, columns=None)

    scaler = MinMaxScaler()
    dat = scaler.fit_transform(df.values)
    scaled = pd.dfFrame(dat, columns=df.columns)
# There are now x features, since we broke down categorical vars

    fraude = df['fraude']
    scaled_df = scaled.drop(columns=['fraude']) 

    # load the model from disk
    filename = "./model.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    df['Prediction_is_fraud'] = loaded_model.predict(scaled_df)

    

    response = make_response(df.to_csv())
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response

if __name__ == "__main__":
    app.run(debug=True)