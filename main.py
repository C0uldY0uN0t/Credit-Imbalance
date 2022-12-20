import numpy as np 
import pandas as pd
from flask import Flask, redirect, url_for, request, render_template
import pickle as pickle

rank = np.genfromtxt('test.out', delimiter=',', dtype=None, names=('model', 'accuracy', 'auc_score', 'fscore', 'recall', 'precision', 'fpr'))

ret = []
for i in range(len(rank)):
    app = []
    for j in range(len(rank[0])):
        app.append(rank[i][j])
    ret.append(app)


modelList = pd.DataFrame.from_records(ret)
cols = ['model', 'accuracy', 'auc_score', 'fscore', 'recall', 'precision', 'fpr']
modelList.columns = cols
fpr_first = modelList.sort_values(by=['fpr', 'accuracy', 'auc_score'], ascending=[True, False, False])

print(fpr_first.head())

print('We\'ll use HYBRID of SMOTE and RandomUnderSampling\n\t')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello Sammy!'

@app.route('/home', methods=['POST', 'GET'])
def getvals():
    if request.method == 'POST':
        user = request.form['array[]']
        print(f'user: {user}')
        return redirect(url_for('prediction', vels = user))
    else:
        return render_template('vis.html')
    

@app.route('/prediction/<vels>')
def prediction(vels):
    print(f'Vels: {vels}')
    pipe = pickle.load(open('SMOTERUS.pkl', 'rb'))
    out = pipe.predict([[vels]])
    if out == [1]:
        return 'Fraud'
    else:
        return 'Not Fraud'

if __name__ == '__main__':
    app.run(debug=True)