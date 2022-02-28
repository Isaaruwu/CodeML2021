import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from Utils import select_features

mats_train = pd.read_csv('Test&Train/train.csv')

print('imputing... \n')
imputer = KNNImputer(n_neighbors=500)
numpied_mats_train = imputer.fit_transform(mats_train.to_numpy())
print('done imputing \n')

mats_train = pd.DataFrame(numpied_mats_train, columns=mats_train.columns)

X = mats_train.drop('critical_temp', axis=1)
y = mats_train['critical_temp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=420)

print('scaling... \n')
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('done scaling \n')

X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

print('training and predicting... \n')
rfc = RandomForestRegressor(n_estimators=50, min_samples_split=5, max_features='sqrt', min_samples_leaf=2,
                            max_depth=100, bootstrap=False, n_jobs=-1)
# {'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 100, 'bootstrap': False}

rfc.fit(X_train_fs, y_train)
pred_rfc = rfc.predict(X_test_fs)
print('done training and predicting \n')
print('exporting... \n')

print('done exporting')
pred_r2_score = r2_score(y_test, pred_rfc)
print('result: ' + str(pred_r2_score))
with open('scores.txt', 'a') as f:
    f.write(str(pred_r2_score) + '\n')
