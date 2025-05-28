from sklearn import datasets
import pandas as pd

# Загрузка данных
data = datasets.fetch_openml(name='titanic', version=1)

# Преобразование данных в DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Добавление целевой переменной (target) в DataFrame
df['survived'] = data.target


# Сохранение данных в CSV-файл
df.to_csv('titanic.csv', index=False)
