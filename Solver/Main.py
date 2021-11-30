import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from Utils import select_features

mats_train = pd.read_csv('train.csv')
mats_test = pd.read_csv('test.csv')

print('imputing... \n')
imputer = KNNImputer(n_neighbors=500)
numpied_mats_train = imputer.fit_transform(mats_train.to_numpy())
numpied_mats_test = imputer.fit_transform(mats_test.to_numpy())
print('done imputing \n')

mats_train = pd.DataFrame(numpied_mats_train, columns=mats_train.columns)
mats_test = pd.DataFrame(numpied_mats_test, columns=mats_test.columns)

# Solution
X_train = mats_train.drop('critical_temp', axis=1)
y_train = mats_train['critical_temp']
X_test = mats_test.drop('index', axis=1)

print('scaling... \n')
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print('done scaling \n')

print('selecting best features... \n')
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
print('dont selecting best features \n')

print('training and predicting... \n')
rfc = RandomForestRegressor(n_estimators=1000, min_samples_split=5, min_samples_leaf=2, max_features='sqrt',
                            max_depth=100, bootstrap=False, n_jobs=-1)

rfc.fit(X_train_fs, y_train)
pred_rfc = (rfc.predict(X_test_fs))
new_pred = []
for pred in pred_rfc:
    new_pred.append(float(pred))

print('done training and predicting \n')
print('exporting... \n')

np.savetxt("results_v11.csv", np.dstack((np.arange(0, len(new_pred)), new_pred))[0], "%d,%d",
           header="index,critical_temp")
print('done exporting')

# pred_r2_score = r2_score(y_test, pred_rfc)
# print('result: '+str(pred_r2_score))
# with open('scores.txt','a') as f:
#  f.write(str(pred_r2_score)+'\n')
