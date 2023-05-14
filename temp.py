import pandas as pd

# Загрузка исходного файла csv
df = pd.read_csv('D:\Chillzone\study\KNN\Identifying-cars-by-their-look\\val.csv', low_memory=True)

# Разделение на две части в соотношении 10 к 90
train_data = df.sample(frac=0.9, random_state=1)
test_data = df.drop(train_data.index)

# Сохранение разделенных данных в отдельные файлы csv
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)