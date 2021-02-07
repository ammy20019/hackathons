from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features = [[int(x) for x in request.form.values()]]
    final = np.array(int_features)
    print(final)
    col=np.array(['age', 'pay_schedule', 'home_owner', 'income', 'months_employed',
    'current_address_year', 'personal_account_m', 'personal_account_y',
    'has_debt', 'amount_requested', 'risk_score', 'inquiries_last_month'])
    df=pd.DataFrame(final,columns=col)
    prediction = model.predict(df)

#print the output
    if prediction==1:
        return render_template('index.html',pred='Credit Baba Says Your Loan will be Approved')
    else:
        return render_template('index.html',pred='Credit Baba Says Your Loan will not be Approved')

if __name__ == '__main__' :
    app.run(debug = True)
    #app.debug = True
    #app.run(host = '127.0.0.0', port = 5000)

