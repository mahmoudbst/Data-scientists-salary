

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('salary_data_cleaned.csv')
df.columns

df_model1 = df[['Avg Salary(K)','Rating','Size','Type of ownership', 'Industry', 'Sector', 'Revenue','Num_competitors','Python', 'SQL','Hourly', 'Employer_Provided_Salary','State','Job_Desc_Len','Company_Age_Years']]

#dummy data
dum = pd.get_dummies(df_model1)

#train test split

from sklearn.model_selection import train_test_split
X = dum.drop('Avg Salary(K)', axis = 1)
y = dum['Avg Salary(K)'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model building
## linear regression and lasso

from sklearn.linear_model import LinearRegression , Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train,y_train)

np.mean(cross_val_score(lm,X_train, y_train, scoring = 'neg_mean_absolute_error' , cv = 4))

lm_l=Lasso(alpha = 0.12)
lm_l.fit(X_train, y_train)
np.mean(cross_val_score(lm_l,X_train, y_train, scoring = 'neg_mean_absolute_error' , cv = 4))

alpha=[]
error=[]
for i in range(1,100):
    alpha.append(i/100)
    lml=Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train, y_train, scoring = 'neg_mean_absolute_error' , cv = 4)))
plt.plot(alpha,error)
err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err,columns=['alpha','error'])
df_err[df_err.error == max(df_err.error)]

##random forest

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 4))

#models tuning

from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,100,10), 'criterion':('mse', 'mae'), 'max_features':('auto','sqrt','log2')}
gs = GridSearchCV(rf, parameters, scoring = 'neg_mean_absolute_error', cv = 4)
gs.fit(X_train, y_train)
gs.best_score_
gs.best_params_

#testing
from sklearn.metrics import mean_absolute_error
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_ref = gs.best_estimator_.predict(X_test)

s={'score for linear regression' : tpred_lm, 'score for lasso': tpred_lml, 'score for random forest' : tpred_ref}

for k,v in s.items(): 
    print(k,mean_absolute_error(y_test, v))
    
import pickle
pickl = {'model' : gs.best_estimator_}
pickle.dump( pickl, open('model_file'+'.p', 'wb') )
