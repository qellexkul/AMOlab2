from sklearn import datasets
import pandas as pd

data = datasets.fetch_openml(name='titanic', version=1)

df = pd.DataFrame(data.data, columns=data.feature_names)

df['survived'] = data.target


df.to_csv('titanic.csv', index=False)
