import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu as mw
import seaborn as sns
import networkx as nx

def statystyki(zmienne_ciagle):
    ''' Tworzy macierz statystyk podanych zmiennych jakościowych'''
    statystyki = zmienne_ciagle.describe().T
    statystyki= statystyki.drop('count',axis = 1)
    nazwy_stat = ['Średnia', 'Odchyl. std', 'Min', '1Q', 'Mediana', '3Q', 'Max']
    statystyki.columns = nazwy_stat
    skosnosc = zmienne_ciagle.skew()
    kurtoza = zmienne_ciagle.kurtosis()
    statystyki['Skośność'] = skosnosc
    statystyki['Kurtoza'] = kurtoza
    return statystyki

def korelacje(dane, prog_silnej_korelacji = 0.7):
    '''Przeszukuje górną częsć macierzy i wypisuje wszytskie korelacje powyżej pewnego progu'''
    korelacje = []
    for i, x in enumerate(dane[1:]):
        for j, y in enumerate(dane[:1]):
            if i > j:
                if np.abs(dane[x][y]) >= prog_silnej_korelacji:
                    korelacje.append([x, y])
    return korelacje

def kolowy(wartosci, opis, kolory):
    plt.pie(wartosci, labels = opis,  colors = kolory, autopct = '%1.2f%%')

def stat(pozyt, negat, zmienna, granica = None):
    ilosc_obserwacji = 85959
    ramka = pd.DataFrame({'pozytywne': pozyt[zmienna], 'negatywne': negat[zmienna]})
    print('-> Test Manna-Whitneya\n\tH0: Mediana dla zmiennej', zmienna, 'równa dla obserwacji pozytywnych i negatywnych' )
    test = mw(ramka['pozytywne'].dropna(), ramka['negatywne'].dropna())
    pwartosc = round(test[1],5)
    if pwartosc < 0.05:
        print('\tp-wartość wynosi', str(pwartosc), '- odrzucamy hipotezę zerową')
    else:
        print('\tp-wartość wynosi', str(pwartosc), '- brak podstaw do odrzucenia hipotezy zerowej')
    print('-> Test Shapiro dla przypadków pozytywnych\n\tH0: Zmienna', zmienna, 'dla przypadków pozytywnych ma rozkład normalny')
    test2 = st.shapiro(ramka['pozytywne'].dropna())
    pwartosc2 = round(test2[1],5)
    if pwartosc2 < 0.05:
        print('\tp-wartość wynosi', str(pwartosc2), '- odrzucamy hipotezę zerową')
    else:
        print('\tp-wartość wynosi', str(pwartosc2), '- brak podstaw do odrzucenia hipotezy zerowej')
    print('-> Test Shapiro dla przypadków negatywnych\n\tH0: Zmienna', zmienna, 'dla przypadków negatywnych ma rozkład normalny')
    test3 = st.shapiro(ramka['negatywne'].dropna())
    pwartosc3 = round(test3[1],5)
    if pwartosc3 < 0.05:
        print('\tp-wartość wynosi', str(pwartosc3), '- odrzucamy hipotezę zerową')
    else:
        print('\tp-wartość wynosi', str(pwartosc3), '- brak podstaw do odrzucenia hipotezy zerowej')
    brakip = pozyt[zmienna].isnull().sum()
    brakin = negat[zmienna].isnull().sum()
    ilosc_poz = pozyt.shape[0]; ilosc_neg = negat.shape[0]
    iloscp = ilosc_poz - brakip
    iloscn = ilosc_neg - brakin
    print('\n-> Liczność braków danych dla przypadków pozytywnych: ' + str(brakip) + ' (' + str(round(brakip/ilosc_poz*100,2)) + '%)')
    print('-> Procent przypadków pozytynych w całym zbiorze ' + str(round(iloscp/ilosc_obserwacji*100,2))+'%')
    
    print('\n-> Liczność braków danych dla przypadków negatywnych: ' + str(brakin) + ' (' + str(round(brakin/ilosc_neg*100,2)) + '%)')
    print('-> Procent przypadków negatywnych w całym zbiorze ' + str(round(iloscn/ilosc_obserwacji*100,2))+'%')
    
    print('\nTABELA PODSTAWOWYCH STATYSTYK')
    st_poz = statystyki(pozyt)[['Średnia', 'Mediana', 'Odchyl. std', 'Kurtoza', 'Skośność']][zmienna:zmienna]
    st_poz = st_poz.rename(index = {zmienna: 'Pozytywne'})
    st_neg = statystyki(negat)[['Średnia', 'Mediana', 'Odchyl. std', 'Kurtoza', 'Skośność']][zmienna:zmienna]
    st_neg = st_neg.rename(index = {zmienna: 'Negatywne'})
    stat_poz_neg = pd.concat([st_poz, st_neg])
    print(stat_poz_neg)
    
    ramka.hist(figsize = (20,7), bins = 30)
    plt.show()
    kolory = ['red', 'green']
    if granica:
        ramka.plot.density(figsize = (20,7), xlim = granica, color=kolory)
    else:
        ramka.plot.density(figsize = (20,7),color=kolory)
    plt.title('Wykres gęstości')
    plt.grid(True)
    plt.show()
    plt.figure(figsize = (20,7))
    plt.subplot(1, 2, 1)
    ramka.boxplot(vert=False)
    plt.title('Wykres pudełkowy zmiennej {}'.format(zmienna))
    plt.subplot(1, 2, 2)
    do_kolowego = [iloscp, iloscn]
    plt.pie(do_kolowego, labels = ['pozytywne', 'negatywne'],  colors = ['red', 'green'], autopct = '%1.2f%%')
    tyt = 'Procentowa ilość przypadków pozytywnych i negatywnych dla cechy {}'.format(zmienna)
    plt.title(tyt)

def stat_jakosciowe(pozyt, negat, zmienna, kolejnosc = None):
    b1 = pozyt[zmienna].value_counts()
    b2 = negat[zmienna].value_counts()
    if kolejnosc:
        plt.subplot(1, 2, 1)
        b1[kolejnosc].plot(kind = 'bar', figsize = (20,5), title = 'pozytywne', rot = 0)
        plt.subplot(1, 2, 2)            
        b2[kolejnosc].plot(kind = 'bar', title = 'negatywne', rot = 0)
        czestosc = pd.concat([round(b1[kolejnosc]/sum(b1)*100,2), round(b2[kolejnosc]/sum(b2)*100,2)], axis = 1)
        czestosc.columns = ['pozytywne', 'negatywne']
        print('Odsetek wartości [%]\n', czestosc)
    else:
        plt.subplot(1, 2, 1)
        b1.plot(kind = 'bar', figsize = (20,5), title = 'pozytywne', rot = 0)
        plt.subplot(1, 2, 2)
        b2.plot(kind = 'bar', title = 'negatywne', rot = 0)
        
def braki_danych(pozytywy, negatywy):
    nazwy_zm = list(pozytywy.keys())
    nazwy_zm.remove('patient_number')
    nazwy_zm.remove('label')
    braki = []
    len_poz = len(pozytywy['x1'])
    len_neg = len(negatywy['x1'])
    for zm in nazwy_zm:
        bp = pozytywy[zm].isnull().sum()
        bn = negatywy[zm].isnull().sum()
        braki.append((zm, round(bp*100/len_poz,2), round(bn*100/len_neg,2)))
    
    braki = sorted(braki, key = lambda x: x[1])
    braki.reverse()
    print('ILOŚĆ BRAKÓW DANYCH\nzmienna\t\tpozytywne\tnegatywne')
    for i, brak in enumerate(braki):
        w = brak[0] + '\t\t' + str(brak[1]) + '%\t\t' + str(brak[2]) + '%'
        print(w)
    return braki


def slupkowy_braki(braki):
    zm = []; poz = []; neg = []
    for brak in braki:
        zm.append(brak[0]); poz.append(brak[1]); neg.append(brak[2])
    s1 = np.arange(len(zm))
    dlugosc = 0.25
    s2 = [x + dlugosc for x in s1]
    plt.figure(figsize = (20,50))
    plt.grid(True)
    plt.barh(s1, poz, height = dlugosc, label = 'positive', color = 'red')
    plt.barh(s2, neg, height = dlugosc, label = 'negative', color = 'green')
    plt.yticks([x + dlugosc for x in range(len(zm))], zm)
    plt.show()
    
def sasiedzi(v, kraw, przyp):
    s = []
    for w1, w2 in kraw:
        if v == przyp[w1]:
            s.append(przyp[w2])
        elif v == przyp[w2]:
            s.append(przyp[w1])
    return s

def skladowe(krawedzie):
    k = [] #krawędzie
    w = [] #wierzchołki
    for i, kraw in enumerate(krawedzie):
        w1, w2 = kraw  
        k.append((w1, w2))
        if w1 not in w:
            w.append(w1)
        if w2 not in w:
            w.append(w2)
    w.sort()
    przypisanie = {}
    for i, wierz in enumerate(w):
        przypisanie[wierz] = i
    n = len(w)
    S = []
    C = [0 for i in range(n)]
    cn = 0
    for i in range(n):
        if C[i] > 0:
            continue
        cn += 1
        S.append(i)
        C[i] = cn
        while len(S)>0:
            v = S.pop(-1)
            sasiad = sasiedzi(v, k, przypisanie)
            for u in sasiad:
                if C[u] >0:
                    continue
                S.append(u)
                C[u] = cn

    for i in range(1, cn+1):
        print('\nNumer grupy: ', i)
        for j in range(n):
            if C[j] == i:
                print('Zmienna: ', end ='')
                for klucz, wartosc in przypisanie.items():
                    if wartosc == j:
                        print(klucz)

def grafy_korelacji(korelacje, zmienne):
    graf = []
    for e in korelacje:
        w1, w2 = e
        if w1 in zmienne and w2 in zmienne and (w2,w1) not in graf:
            graf.append((w1,w2))
    G=nx.Graph()
    G.add_edges_from(graf)
    nx.draw(G, with_labels = True, node_color = 'white', font_size = 8)