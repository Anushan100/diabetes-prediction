import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
df=pd.read_csv('diabetes2.csv')
df['Insulin']=df['Insulin'].replace(0,120.89)
df['Glucose']=df['Glucose'].replace(0,79.79)
df['SkinThickness']=df['SkinThickness'].replace(0,20.53)
df['BMI']=df['BMI'].replace(0,31.99)
df['BloodPressure']=df['BloodPressure'].replace(0,69)
x=df.drop(columns=['Outcome'])
y=df['Outcome']
rob=RobustScaler()
X_scaled=rob.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.5,random_state=3)
parameters = {
    'penalty' : ['l1','l2'],
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}
logreg = LogisticRegression()
clf = GridSearchCV(logreg,
                   param_grid = parameters,
                   scoring='accuracy',
                   cv=15,verbose=0,n_jobs=30,refit=True)
model=clf.fit(x_train,y_train)
import pickle
pickle.dump(clf,open('finalmodel.pkl','wb'))
loadmodel=pickle.load(open('finalmodel.pkl','rb'))