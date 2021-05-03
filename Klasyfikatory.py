from sklearn.ensemble import RandomForestClassifier as RF_C
from sklearn.tree import DecisionTreeClassifier as crt
from sklearn import svm
from sklearn.model_selection import train_test_split as tts
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier as ET
import pandas as pd
from jakosc_modelu import jakosc_modelu
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import bayes
import pickle
##############################################Import zbioru danych i usunięcie obserwacji z brakiem w label
dane = pd.read_csv('uzupelniony_dataset.csv',index_col=0)
dane = dane.fillna(value=np.nan)
dane = dane.dropna()
##############################################Zmiana wartosci zmiennych jakosciowych
dane.replace({'x6': {'a':0,'b':1, 'c':2, 'd':3}},inplace=True)
dane.replace({'x8': {'a':0,'b':1, 'c':2}},inplace=True)
dane.replace({'x13': {'a':0,'b':1}},inplace=True)
dane.replace({'x19': {'a':0,'b':1, 'c':2, 'd':3}},inplace=True)
dane.replace({'x39': {'a':0,'b':1}},inplace=True)
dane.replace({'x48': {'a':0,'b':1, 'c':2}},inplace=True)

##############################################Podział zbioru na uczący i testowy
nazwy_pred = list(dane.keys())
nazwy_pred.remove('patient_number')
nazwy_pred.remove('label')
predyktory = dane[nazwy_pred]
cel = dane['label']
pred_ucz, pred_test, cel_ucz, cel_test = tts(predyktory, cel, test_size=0.3, random_state = 2021, stratify=cel)
##############################################Transformacja rozkładu i standaryzacja
pred_uczZ = pred_ucz.copy()
pred_uczZ['x1'] = np.log(pred_uczZ['x1'])/np.log(6)
pred_uczZ['x4'] = np.log(pred_uczZ['x4'])/np.log(1.3)
pred_uczZ['x9'] = np.log(pred_uczZ['x9'])/np.log(1.5)
pred_uczZ['x11'] = np.log(pred_uczZ['x11'])/np.log(6)
pred_uczZ['x14'] = np.log(pred_uczZ['x14'])/np.log(1.2)
pred_uczZ['x16'] = np.log(pred_uczZ['x16'])/np.log(6)
pred_uczZ['x17'] = np.log(pred_uczZ['x17'])/np.log(6)
pred_uczZ['x18'] = np.log(pred_uczZ['x18'])/np.log(1.1)
pred_uczZ['x22'] = np.log(pred_uczZ['x22'])/np.log(6)
pred_uczZ['x28'] = np.log(pred_uczZ['x28'])/np.log(2)
pred_uczZ['x32'] = np.log(pred_uczZ['x32'])/np.log(6)
pred_uczZ['x36'] = np.log(pred_uczZ['x36'])/np.log(6)
pred_uczZ['x38'] = np.log(pred_uczZ['x38'])/np.log(1.1)
pred_uczZ['x44'] = np.log(pred_uczZ['x44'])/np.log(6)
pred_uczZ['x49'] = np.log(pred_uczZ['x49'])/np.log(1.1)

standaryzacja = StandardScaler()
standaryzacja.fit(pred_uczZ)
pred_uczZ = standaryzacja.transform(pred_uczZ)

pred_testZ = pred_test.copy()
pred_testZ['x1'] = np.log(pred_testZ['x1'])/np.log(6)
pred_testZ['x4'] = np.log(pred_testZ['x4'])/np.log(1.3)
pred_testZ['x9'] = np.log(pred_testZ['x9'])/np.log(1.5)
pred_testZ['x11'] = np.log(pred_testZ['x11'])/np.log(6)
pred_testZ['x14'] = np.log(pred_testZ['x14'])/np.log(1.2)
pred_testZ['x16'] = np.log(pred_testZ['x16'])/np.log(6)
pred_testZ['x17'] = np.log(pred_testZ['x17'])/np.log(6)
pred_testZ['x18'] = np.log(pred_testZ['x18'])/np.log(1.1)
pred_testZ['x22'] = np.log(pred_testZ['x22'])/np.log(6)
pred_testZ['x28'] = np.log(pred_testZ['x28'])/np.log(2)
pred_testZ['x32'] = np.log(pred_testZ['x32'])/np.log(6)
pred_testZ['x36'] = np.log(pred_testZ['x36'])/np.log(6)
pred_testZ['x38'] = np.log(pred_testZ['x38'])/np.log(1.1)
pred_testZ['x44'] = np.log(pred_testZ['x44'])/np.log(6)
pred_testZ['x49'] = np.log(pred_testZ['x49'])/np.log(1.1)

pred_testZ = standaryzacja.transform(pred_testZ)


##############################################Poszukiwanie hiperparametrów
parametry = open('parametry', 'a')

parsvc = bayes.bayes_svc(pred_ucz, cel_ucz)
print(parsvc.best_params_)

parcrt = bayes.bayes_crt(pred_ucz, cel_ucz)
print(parcrt.best_params_)


pickle.dump(parsvc, parametry)

parametry.close()
##############################################Klasyfikator - Lasy losowe
las = RF_C()
las.fit(pred_ucz, cel_ucz)

las_jakosc = jakosc_modelu(las, pred_test, cel_test)
las_jakosc.utworz_wskazniki()
Mx = las_jakosc.mx_pomylek
las_jakosc.wskazniki
las_jakosc.roc_auc('positive')
las_jakosc.auc


##############################################Klasyfikator - SVC
s = svm.LinearSVC(C = 1.8389187129249611, loss = 'hinge', tol = 0.00017694327306794636)
s.fit(pred_ucz, cel_ucz)
s = CalibratedClassifierCV(s)
s.fit(pred_ucz, cel_ucz)

svm_jakosc = jakosc_modelu(s, pred_test, cel_test)
svm_jakosc.utworz_wskazniki()
Mx2 = svm_jakosc.mx_pomylek
svm_jakosc.wskazniki
svm_jakosc.roc_auc('positive')
svm_jakosc.auc

##############################################Klasyfikator - Drzewo decyzyjne
drzewo = crt(criterion = 'entropy', ccp_alpha = 0.0035, class_weight = 'balanced')
drzewo.fit(pred_ucz, cel_ucz)

drzewo_jakosc = jakosc_modelu(drzewo, pred_test, cel_test)
drzewo_jakosc.utworz_wskazniki()
Mx3 = drzewo_jakosc.mx_pomylek
drzewo_jakosc.wskazniki
drzewo_jakosc.roc_auc('positive')
drzewo_jakosc.auc
##############################################Klasyfikator - ExtraTrees
et = ET()
et.fit(pred_ucz, cel_ucz)

et_jakosc = jakosc_modelu(et, pred_test, cel_test)
et_jakosc.utworz_wskazniki()
Mx4 = et_jakosc.mx_pomylek
et_jakosc.wskazniki
et_jakosc.roc_auc('positive')
et_jakosc.auc

