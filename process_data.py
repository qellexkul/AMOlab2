import pandas as pd
from sklearn.model_selection import train_test_split

titanic_df = pd.read_csv('titanic.csv')

# Data preprocessing
titanic_df['sex'] = titanic_df['sex'].map({'female': 0, 'male': 1})
titanic_df['embarked'] = titanic_df['embarked'].fillna('S')
titanic_df['embarked'] = titanic_df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
titanic_df['age'] = titanic_df['age'].fillna(titanic_df['age'].median())
titanic_df['fare'] = titanic_df['fare'].fillna(titanic_df['fare'].median())

# Select features and target variable
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

X_train, X_test, y_train, y_test = train_test_split(titanic_df[features], titanic_df[target], test_size=0.2, random_state=42)

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)