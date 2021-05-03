from skopt import BayesSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier as crt
from sklearn import svm
def bayes_crt(pred, cel):
    '''Szukanie najlepszych hipierparametrów modelu opartego o drzewo decyzyjne'''
    opt = BayesSearchCV(crt(), {
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced'],
        'ccp_alpha': (0.00000001, 0.05)
        }, n_iter = 200, cv = 3, scoring='roc_auc')
    opt.fit(pred, cel)
    cv_scores = cross_val_score(opt, pred, cel)
    return opt

def bayes_svc(pred, cel):
    '''Szukanie najlepszych hipierparametrów modelu opartego o liniowy SVM'''
    opt = BayesSearchCV(svm.LinearSVC(), {
        'loss': ['hinge', 'squared_hinge'],
        'C': (0.000001, 1000),
        'tol': (0.000001,1)
        }, n_iter = 100, cv = 3, scoring='roc_auc')
    opt.fit(pred, cel)
    cv_scores = cross_val_score(opt, pred, cel)
    return opt

