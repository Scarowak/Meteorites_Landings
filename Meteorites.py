# -*- coding: utf-8 -*-

# Wczytuje biblioteki
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import geopandas
from cartopy import crs as ccrs

pd.options.mode.chained_assignment = None  # default='warn' 


#Wczytuje dane o meteorytach 
df=pd.read_csv("meteorite-landings.csv", delimiter=',')

#Usuwam rekordy z brakujacymi danymi
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)



"""
Dziele meteoryty wedlug masy:
    a - najlzejsze meteoryty - 0 - 10^-7 % masy maksymalnej 
    b - 10^-7 % < masa <= 10^-6 %
    c - 10^-6 % < masa <= 10^-5 %
    ...
    j - najciezsze meteoryty, 10% - 100% masy maksymalnej
"""

df['Procent_masy_max']=df['mass']*100/max(df['mass']) #Oblicza % masy dla wszystkich rekordow

#Lista kategorii
kat_list=[10**-7,10**-6 ,10**-5 ,10**-4 ,10**-3 ,10**-2 ,10**-1 ,10**0 ,10**1,10**2]

#Przydziela kategorie poszczegolnym meteorytom
df['Kategoria_masy']=np.select([(df['Procent_masy_max']< kat_list[0]),
                                (df['Procent_masy_max']< kat_list[1]),
                                 (df['Procent_masy_max']< kat_list[2]),
                                  (df['Procent_masy_max']< kat_list[3]),
                                   (df['Procent_masy_max']< kat_list[4]),
                                    (df['Procent_masy_max']< kat_list[5]),
                                     (df['Procent_masy_max']< kat_list[6]),
                                      (df['Procent_masy_max']< kat_list[7]),
                                       (df['Procent_masy_max']< kat_list[8]),
                                        (df['Procent_masy_max']<= kat_list[9])], ["a", "b", "c", "d", "e", "f","g", "h", "i", "j"], default=np.nan)


#Liczy ilosc meteorytow dla poszczegolnych kategorii wagowych
a_count=sum(df['Kategoria_masy']=='a')
b_count=sum(df['Kategoria_masy']=='b')
c_count=sum(df['Kategoria_masy']=='c')
d_count=sum(df['Kategoria_masy']=='d')
e_count=sum(df['Kategoria_masy']=='e')
f_count=sum(df['Kategoria_masy']=='f')
g_count=sum(df['Kategoria_masy']=='g')
h_count=sum(df['Kategoria_masy']=='h')
i_count=sum(df['Kategoria_masy']=='i')
j_count=sum(df['Kategoria_masy']=='j')

#Tworzy liste z suma poszczegolnych kategorii
suma_kat=[a_count, b_count, c_count, d_count, e_count, f_count, g_count, h_count, i_count, j_count]


#Tworzenie wykresu

plt.title('Rozład masy mateorytów dla poszczególnych klas: Świat') #Tytul wykresu
plt.xticks(rotation=310) #Rotacja opisu wartosci osi X
plt.xlabel("Kategoria masy meteorytu") #Opis osi X 
plt.ylabel("Ilość wystapień") #Opis osi Y


for i in range(len(suma_kat)): #Petla tworzy adnotacje dla poszczegolnych kolumn wykresu
    plt.annotate(suma_kat[i], xy=(i,suma_kat[i]), ha='center', va='bottom') 


plt.bar([str(kat_list[x]) for x in range(len(kat_list))],suma_kat) #tworzy wykres slupkowy
plt.plot(suma_kat, color='green') #tworzy wykres liniowy

plt.savefig('Rozklad masy mateorytow.png') #zapisuje wykres
plt.show() #wyswietla wykres


""" Tworzenie tabel dla wartosci skrajnych """

limes=0.003/2  #wartosc graniczna

ilosc_skrajnych=round(len(df)*limes)


df.sort_values('mass', inplace=True) #sortowanie wartosci wedlug masy
df.reset_index(inplace=True, drop=True) #resetowanie indeksow

df_max=df[len(df)-ilosc_skrajnych:] #wybor najciezszych meteorytow
df_max.reset_index(inplace=True, drop=True ) #resetuje indexy

df_min=df[:ilosc_skrajnych] #wybor najlzejszych meteorytow 



df_max.to_csv('meteoryty_najciezsze.csv', index=False) #Zapisuje dataframy
df_min.to_csv('meteoryty_najlzejsze.csv', index=False)

df.to_csv('meteoryty_po_obrobce.csv', index=False)



""" Sprawdzenie licznosci poszczegolnych klass najciezszych meteorytow """

df_max_serries=df_max.groupby(['recclass'])['recclass'].count() #grupowanie klass

df_max_count=df_max_serries.to_frame(name=None) #zmiana typu danych z serii na data_frame

df_max_count.rename(columns = {'recclass':'Count'}, inplace = True) #zmiana nazw kolumn
df_max_count.reset_index(inplace=True) #reset indexu


df_iron=pd.DataFrame([['Iron',0]],columns=['recclass','Count'])  #tworzenie pustych dataframow dla podobnych klass meteorytow
df_pallasite=pd.DataFrame([['Pallasite',0]],columns=['recclass','Count'])

others_recclass=[]#listy orzechowujace dane o reszcie klas 
others_count=[]

for i in range(len(df_max_count)): #petla sortujaca klasy
    
    if df_max_count['recclass'][i][0:4]=='Iron': #jezeli pierwsze litery klasy to Iron
        df_iron['Count'][0]=df_max_count['Count'][i]+df_iron['Count'][0] #zlicza ilosc meteorytow zelaznych wszystkich rodzajow
        
    
    elif df_max_count['recclass'][i][0:4]=='Pall': #jezeli pierwsze litery klasy to Pall
        df_pallasite['Count'][0]=df_max_count['Count'][i]+df_pallasite['Count'][0] #zlicza ilosc meteorytow Pallasite

    else: #przechowuje dane o reszcie meteorow

        others_recclass.append(df_max_count['recclass'][i]) 
        others_count.append(df_max_count['Count'][i])

others = {'recclass':others_recclass,
          'Count':others_count} #tworzy slownik z danynymi o reszcie meteorytow

df_others=pd.DataFrame(others) #przeksztalca slownik w dataframe 

df_max_count=pd.concat([df_iron,df_pallasite,df_others]) #laczy dane 


df_max_count.rename(columns={'Count': 'Ilość'}, inplace=True) #zmienia nazwe kolumny count na ilosc
df_max_count.sort_values('Ilość', inplace=True)  #sortuje dane wedlug ilosci wystapien w danej serii
df_max_count.reset_index(inplace=True, drop=True) #resetuje indeksy


""" Tworzenie wykresu wystpapien poszczegolnych rodzajow najciezszych meteorytow """

plot_max_classes = df_max_count.plot.bar(title = 'Ilość wystąpień meteorytów poszczególnych rodzajów: Świat', x='recclass', y='Ilość', rot=330) #Tworzy wykres slupkowy

for container in plot_max_classes.containers: #petla dodajaca etykiety ilosci do slupkow
    plot_max_classes.bar_label(container) 
  
plt.xlabel("Rodzaj") #opis osi x i y
plt.ylabel("Ilość wystąpień")    

plot_max_classes.figure.savefig('max_classes.jpg',bbox_inches="tight", dpi=300) #zapisuje wykres

""" Tworzenie wykresu dla najcięzszych meteteorytów """

df_max['mass']=df_max['mass']/1000   #zmienia jednostke na kg w kolumnie masa

plot_max = df_max.plot.scatter(title = 'Najcięższe meteoryty: Świat', x='name', y='mass', rot=90, figsize=(12,5), grid=True, color= "green", marker='d') #Tworzy wykres


plt.xlabel("Meteoryt") #Oznaczenia osi X i Y
plt.ylabel("Masa [kg]")

plot_max.figure.savefig('max_mass.jpg',bbox_inches="tight", dpi=300) #zapisuje wykres


""" MAPA ŚWIATA """

path = geopandas.datasets.get_path('naturalearth_lowres')
world = geopandas.read_file(path)

world['gdp_pp'] = world['gdp_md_est'] / world['pop_est']
ax=world.plot()
world_map=df_max.plot(kind='scatter',y='reclat', x='reclong', color='r', s=5, ax=ax)

for k, v in df_max[['reclong','reclat']].iterrows():
    world_map.annotate(k, v, fontsize=6)

world_map.figure.savefig('world_map_max_mass.jpg',bbox_inches="tight", dpi=600) #zapisuje wykres



""" NAJWIĘKSZE METEORYTY W DANEGO RODZAJU """


biggest=df.sort_values('mass', ascending=False).drop_duplicates(['recclass'])
biggest.to_csv('najwieksze_klasy.csv', index=False)
    