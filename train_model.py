import pandas as pd
from sklearn.ensemble  import RandomForestClassifier
import pickle

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)