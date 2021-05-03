import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

class jakosc_modelu():
    def __init__(self, model, predyktory, cel, prog_prob = False):
        self.model = model
        self.prog_prob = prog_prob
        self.predyktory = predyktory
        self.cel = cel
        self.nazwy_predyktorow = list(predyktory.keys())
        self.klasyfikacja()        
    def klasyfikacja(self):
        if self.prog_prob:
            self.przewidywane = self.model.predict_proba(self.predyktory)[:,1] >= self.prog_prob
            self.przewidywane = pd.DataFrame({'label': self.przewidywane })
            self.przewidywane.replace({'label': {True:'positive', False: 'negative'}},inplace=True)
        else:
            self.przewidywane = self.model.predict(self.predyktory)
        self.prawdop = self.model.predict_proba(self.predyktory)
    def utworz_wskazniki(self):
        self.mx_pomylek = CM(self.cel, self.przewidywane)
        TN_U, FP_U, FN_U, TP_U = self.mx_pomylek.ravel()
        self.wskazniki = {}
        self.wskazniki['Trafność'] = round((TP_U+TN_U)/(TP_U+FP_U+TN_U+FN_U),3)
        self.wskazniki['Całkowity współczynnik błędu'] = round((FN_U+FP_U)/(TP_U+FP_U+TN_U+FN_U),3)
        self.wskazniki['Czułość'] = round(TP_U/(FN_U+TP_U),3)
        self.wskazniki['Specyficzność'] = round(TN_U/(TN_U+FP_U),3)
        self.wskazniki['Wskaźnik fałszywie negatywnych'] = round(FN_U/(FN_U+TP_U),3)
        self.wskazniki['Wskaźnik fałszywie pozytywnych'] = round(FP_U/(TN_U+FP_U),3)
        self.wskazniki['Precyzja'] = round(TP_U/(FP_U+TP_U),3)
        self.wskazniki['Proporcja prawdziewie negatywnych'] = round(TN_U/(TN_U+FN_U),3)
        self.wskazniki['Proporcja fałszywie pozytywnych'] = round(FP_U/(FP_U+TP_U),3)
        self.wskazniki['Proporcja fałszywie negatywnych'] = round(FN_U/(TN_U+FN_U),3)
    def roc_auc(self, nazwa_cel):
        fpr, tpr, thr = roc_curve(self.cel, self.prawdop[:,1], pos_label=nazwa_cel)
        plt.ylabel('Czułość')
        plt.xlabel('1 - Swoistość')
        plt.plot([0, 1], [0, 1], color= 'red', linestyle='--')
        plt.plot(fpr, tpr, color='blue') 
        plt.scatter(1 - self.wskazniki['Specyficzność'], self.wskazniki['Czułość'], color = 'y', alpha = 1)
        plt.show()
        self.auc = roc_auc_score(self.cel, self.prawdop[:,1])
        self.prawdop = pd.DataFrame({'fpr': fpr, 'tpr,':tpr, 'thr':thr})