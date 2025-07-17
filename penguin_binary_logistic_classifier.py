
"""
Logistic Regression on Binary Penguin Classification Dataset.
Author: [Armaan Siddiqui]
Dataset: penguins_binary_classification.csv
"""

# Penguin Binary Classification

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ====== Load Dataset ======
df=pd.read_csv('/storage/emulated/0/ml_projects/penguins_binary_classification.csv')

#columns: species,  island, bill_lenght_mm, bill_depth_mm, flipper_lenght_mm, body_mass_g, year


# ====== Initial Checks ======
#print(df.iloc[:,[0,1,2,3]])
#print(df.iloc[:,[4,5,6]])
#print(df.head())
#print(df.shape)
#print(df.describe())
#print(df.info())
#print(df.isnull().sum())
#print(df.duplicated().sum())

#sns.boxplot(df)
#plt.show()

sns.pairplot(df, hue='species')
#plt.savefig('/storage/emulated/0/ml_projects/penguin_binary_lairplot.png')
plt.show()

# ====== Feature Scaling (Numerical Features) ======
non_categorical_data=df.iloc[:, 2:-1]

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(non_categorical_data)
scaled_data=ss.transform(non_categorical_data)
df.iloc[:, 2:-1]=scaled_data

x = df[['island']]


# ====== One-Hot Encoding (island) ======
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first')
x_encoded = ohe.fit_transform(x)
encoded_df = pd.DataFrame(x_encoded.toarray(), columns=ohe.get_feature_names_out())
df = df.drop('island', axis=1)
df = pd.concat([df, encoded_df], axis=1)


# ====== Label Encoding (species) ======
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])


# After Encoding:
# Columns: ['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
#           'body_mass_g', 'year', 'island_Biscoe', 'island_Dream']


x=df.iloc[:, 1:]
y=df['species']

#sns.pairplot(df, hue='species')
#plt.show()


# ====== Train-Test Split ======
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)


# ====== Logistic Regression ======
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)


# ====== Accuracy ======
train_acc = lr.score(x_train, y_train) * 100
test_acc = lr.score(x_test, y_test) * 100
print(f"Training Accuracy: {train_acc:.2f}%")
print(f"Testing Accuracy: {test_acc:.2f}%")