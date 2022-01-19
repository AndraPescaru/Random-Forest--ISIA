# Importam librariile de care avem nevoie:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, tree, ensemble
from sklearn.ensemble import BaggingRegressor


# Importam datele din fisierul forestfires.csv

datasets = pd.read_csv('forestfires.csv')

# Inlocuim zilele saptamanii din  datasets cu numere de la 1 - 7
day_map = {"mon": 1,"tue" : 2 ,"wed" : 3,"thu" : 4 ,"fri" : 5, "sat" : 6, "sun" : 7}
datasets['day'] = datasets['day'].map(day_map)

# Inlocuim lunile anului din ddatasets cu numere de la 1 - 12
month_map = {"jan": 1,"feb" : 2 ,"mar" : 3,"apr" : 4 ,"may" : 5, "jun" : 6, "jul" : 7,
             "aug" : 8,"sep" : 9, "oct" : 10, "nov" : 11, "dec" : 12}
datasets['month'] = datasets['month'].map(month_map)

# Vizualizare date dupa inlocuirea cuvintelor:
#print(datasets)

#Bagam datele intr-o matrice
datasets = datasets.to_numpy()
datasets = np.array(datasets)

# Setul de date este compus din 517 esantioane
# Primele 387 (75%) de esantioane sunt folosite pentru antrenare
data_train = datasets[: 387, 0 : 13]
etichete_train = datasets[0 : 387, 12]

# Restul de 130 (25%) de esantioane sunt folosite pentru testare
data_test = datasets[387 : , 0 : 13]
etichete_test = datasets[387 : , 12]

# Vizualizam date:
#print(data_train)
#print(etichete_train)
#print(data_test)
#print(etichete_test)

in_bag = [0.25, 0.5, 0.85 ];
dimensiuni_nod = [0.1, 0.5, 0.8];

# Creare si antrenare arbore de regresie:
for i in range(3):
    for j in range(3):
        regr = BaggingRegressor(n_estimators = 10, max_samples = in_bag[i], max_features = dimensiuni_nod[j], random_state=0)
        regr_fit = regr.fit(data_train, etichete_train)
        
        predictii = regr.predict(data_test);
        suma = 0;
        for k in range(0,len(etichete_test)):
            suma += (predictii[k] - etichete_test[k]) ** 2;
        
        print('Eroarea patratica medie pentru in_bag ', in_bag[i], ' si dimensiunea nodului de', dimensiuni_nod[j], ' este ', suma/len(etichete_test), '\n');
    
