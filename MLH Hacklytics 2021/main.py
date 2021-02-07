# Implementing xgboost
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# reading the files
df=pd.read_csv("https://raw.githubusercontent.com/ammy20019/hackathons/master/MLH%20Hacklytics%202021/financial_data.csv")
df.head()

df=df.drop(columns = ['entry_id',])
df.head()

le = LabelEncoder()
df["pay_schedule"] =le.fit_transform(df["pay_schedule"])
df=df.drop(["risk_score_4","risk_score_5","risk_score_2","risk_score_3","ext_quality_score_2","years_employed","ext_quality_score"],axis=1)


sc_X = StandardScaler()
x=df.drop("e_signed",axis=1)
y=df["e_signed"]
X_train, X_test, y_train, y_test = train_test_split(x,y ,test_size=0.2, random_state=0)

# Fitting and transforming our data
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Initializing the model XGBoost
xgb_clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=200,random_state=100,max_depth=5,max_features='auto'),n_estimators=200,random_state = 100)
xgb_clf.fit(X_train, y_train)

pickle.dump(xgb_clf, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

# score = xgb_clf.score(X_test, y_test)
# print(score)
