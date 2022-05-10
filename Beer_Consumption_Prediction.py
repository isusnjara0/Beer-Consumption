# -*- coding: utf-8 -*-
"""

Predviđanje potrošnje pive

@author: Ivan Šušnjara
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from scipy import stats


data = pd.read_excel('potrosnja pive.xlsx')

print(data.head(20))

data.Vikend.replace('DA',1, inplace=True)


data['Srednja temperatura (C)'].fillna( method ='ffill', inplace = True)
data['Minimalna temperatura (C)'].fillna( method ='ffill', inplace = True)

data.Vikend = data.Vikend.astype('bool')

data = data[(abs(stats.zscore(data.iloc[:,6].values)) <= 3  )] #brišemo retke gdje je potrošnja bila izvan očekivanja
data = data[(abs(stats.zscore(data.iloc[:,4].values)) <= 3  )] #brišemo retke gdje su padaline bile izvan očekivanja

print(data.head(20))

fg, (axs1, axs2) = plt.subplots(nrows = 1, ncols=2, figsize=(10,5))

axs1.scatter(data['Maksimalna temperatura (C)'], data['Potrošnja pive (litara)'], marker='o',c = data['Maksimalna temperatura (C)'], cmap = 'coolwarm')
axs1.set_title('Potrošnja pive zavisno o maksimalnoj dnevnoj temperaturi')

axs2.bar(data.Datum.dt.month, data['Potrošnja pive (litara)'], color='y')
axs2.set_title('Potrošnju pive po mjesecima (1-12)')

plt.show()

minmax = MinMaxScaler()
data.iloc[:,1:-1] = minmax.fit_transform(data.iloc[:,1:-1])


X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,1:-1], data.iloc[:,6], test_size=0.4, random_state=0)


model = linear_model.LinearRegression()
cross = cross_val_score(model, X_train, y_train, cv=4)
model = model.fit(X_train, y_train)  
Y_pred = model.predict(X_test)

print('\nPouzdanost:')
print(model.score(X_test, y_test))     # pouzdanost predviđanja
print(f'Pouzdanost cross_val: {cross.mean() * 100:.2f}% - (+/- {(cross.std() * 2) * 100:.2f}%)')
