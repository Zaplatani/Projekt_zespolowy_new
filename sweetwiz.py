import pandas as pd
from scipy import stats as st
import sweetviz
dane = pd.read_csv('dataset.csv',index_col=0)
pozytywy = dane[dane.label == 'positive']
negatywy = dane[dane.label == 'negative']
rep = sweetviz.compare([pozytywy, 'pozytywne'], [negatywy, 'negatywne'])
rep.show_html('raport.html')

