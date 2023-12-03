import pandas as pd

import csv
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# def load_data(filename):
#    myList=[]
#    with open(filename) as numbers:
#        numbers_data=csv.reader(numbers, delimiter=',')
#        next(numbers_data)
#        for row in numbers_data:
#           myList.append(row)
#        return myList
#
# new_list=load_data('delhidataset.csv')
# for row in new_list:
## print(row)
data = pd.read_csv('delhidataset.csv')


# print(data.Locality)

pipe=pickle.load(open("model.pkl", 'rb'))
@app.route('/locations')
def index():
    locations = sorted(data['Locality'].unique())
    # print(data.Locality)
    return render_template('index.html', locations=locations)



@app.route('/predict',methods=['post'])
def predict():
    Locality=request.form.get('locality')
    bhk= request.form.get('bhk')
    bath= request.form.get('bath')
    sqft=request.form.get('sqft')
    print(Locality, bhk, bath, sqft)
    input =pd.DataFrame([[Locality, sqft, bath, bhk]], Columns=['Locality', 'total_sqft', 'bath', 'bhk'])
    # prediction= pipe.predict(input)[0]
    return str(np.round(prediction,2))

if __name__ == "__main__":
    app = Flask(__name__, template_folder='index.html')
    app.run(debug=True, port=5001)

#
#
#
#
# import pandas as pd
# from flask import Flask, render_template, request
# import pickle
# import numpy as np
#
# app = Flask(__name__)
#
# # Load data from CSV
# data = pd.read_csv('delhidataset.csv')
#
# # Load the machine learning model
# # with open('model.pkl', 'rb') as model_file:
#     # pipe = pickle.load(model_file)
#
# @app.route('/locations')
# def index():
#     locations = sorted(data['Locality'].unique())
#     return render_template('index.html', locations=locations, data=data.to_dict(orient='records'))
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     location = request.form.get('location')
#     bhk = request.form.get('bhk')
#     bath = request.form.get('bath')
#     sqft = request.form.get('sqft')
#
#     # Create a DataFrame from user input
#     input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
#
#     # Make predictions using the loaded model
#     # prediction = pipe.predict(input_data)[0]
#
#     return render_template('index.html', locations=sorted(data['Locality'].unique()), data=data.to_dict(orient='records'), prediction=np.round(predict, 2))
#
# if __name__ == "__main__":
#     app.run(debug=True, port=5001)
