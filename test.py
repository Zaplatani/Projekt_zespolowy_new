import pickle
import pandas as pd
nazwa_pliku = 'imputacja_modele'
plik = open(nazwa_pliku, 'rb')
sl = pickle.load(plik)
sl2 = pickle.load(plik)
plik.close()
