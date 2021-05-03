import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split as tts

dane = pd.read_csv('dataset.csv',index_col=0)
dane = dane.sort_index(axis = 'index')
do_usuniecia = ['x3','x20','x27','x30','x35','x40']
dane = dane.drop(do_usuniecia, axis = 1)

for kat in ['a', 'b', 'c', 'd']:
    nazwa = 'x6_' + kat
    dane[nazwa] = pd.Series(dane['x6'] == kat).astype(int)
dane = dane.drop('x6', axis = 1)

for kat in ['a', 'b', 'c']:
    nazwa = 'x8_' + kat
    dane[nazwa] = pd.Series(dane['x8'] == kat).astype(int)
dane = dane.drop('x8', axis = 1)

dane.replace({'x13': {'a':0,'b':1}},inplace=True)

for kat in ['a', 'b', 'c', 'd']:
    nazwa = 'x19_' + kat
    dane[nazwa] = pd.Series(dane['x19'] == kat).astype(int)
dane = dane.drop('x19', axis = 1)

dane.replace({'x39': {'a':0,'b':1}},inplace=True)

for kat in ['a', 'b', 'c']:
    nazwa = 'x48_' + kat
    dane[nazwa] = pd.Series(dane['x48'] == kat).astype(int)
dane = dane.drop('x48', axis = 1)

dane.replace({'label': {'negative':0,'positive':1}},inplace=True)

nazwy_pred = list(dane.keys())
nazwy_pred.remove('patient_number')
nazwy_pred.remove('label')
predyktory = dane[nazwy_pred]
cel = dane['label']

pred_ucz, pred_test, cel_ucz, cel_test = tts(predyktory, cel, test_size=0.3, random_state = 2021)

xg_class = xgb.XGBClassifier(max_depth = 5,  n_estimators = 10)
xg_class.fit(pred_ucz, cel_ucz)
xgb.plot_tree(xg_class)



