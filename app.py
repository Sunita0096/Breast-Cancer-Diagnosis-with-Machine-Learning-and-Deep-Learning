import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('FinalModel.pkl', 'rb'))

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  input_features = [int(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
       'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
       'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
       'symmetry_se', 'fractal_dimension_se', 'smoothness_worst',
       'compactness_worst', 'symmetry_worst', 'fractal_dimension_worst']

  df = pd.DataFrame(features_value, columns=features_name)
  output = model.predict(df)
  
  
  print(output)

  if output == 4:
      res_val = "Breast cancer"
  else:
      res_val = "no Breast cancer"


  return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
  app.run()
