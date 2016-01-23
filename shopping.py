import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report


shopping_raw = pd.read_csv('shopping/shopping.csv', sep=';')
shopping_raw.head(2)
shopping_raw.info()

shopping_data = pd.DataFrame()
label_encoders = {}

for column in shopping_raw.columns:
    if shopping_raw[column].dtype == 'object':
        label_encoders[column] = preprocessing.LabelEncoder()
        shopping_data[column] = label_encoders[column].fit_transform(shopping_raw[column])
    else:
        shopping_data[column] = shopping_raw[column]

################################


xcols = [col for col in shopping_data.columns if col != 'y']

X = shopping_data[xcols].values
y = shopping_data['y'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

################################

clf = KNeighborsClassifier()

################################

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print clf.predict([
			[56, 10, 1, 0, 0, 3391, 0, 0, 0, 21, 0, 243, 1, -1, 0, 3],
			# 56;"unemployed";"married";"primary";"no";3391;"no";"no";"cellular";21;"apr";243;1;-1;0;"unknown";"yes"
			[46, 1, 1, 1, 0, 668, 1, 0, 2, 15, 8, 1263, 2, -1, 0, 3],
			# 46;"blue-collar";"married";"secondary";"no";668;"yes";"no";"unknown";15;"may";1263;2;-1;0;"unknown";"yes"
			[ 25, 9, 2, 1, 0, 505, 0, 1, 0, 17, 9, 386, 2, -1, 0, 3],
        	# 25;"technician";"single";"secondary";"no";505;"no";"yes";"cellular";17;"nov";386;2;-1;0;"unknown";"yes"
			])


print classification_report(y_test, y_pred)

################################

clf = RandomForestClassifier()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print classification_report(y_test, y_pred)
