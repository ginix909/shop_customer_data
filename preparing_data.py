import pandas as pd
import numpy as np


pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

df = pd.read_csv('Customers.csv')
# удалим 'CustomerID' так как id каждой записи у нас уже есть
df = df.drop(['CustomerID'], axis=1)

# проверим минимальное значение колонки 'Family Size'
# print(df['Family Size'].min())
# ответ 1, все в рамках логики

# проверим сколько всего профессий в колонке 'Profession'
# print(df['Profession'].nunique())
# ответ 9

# среднее значение по колонке 'Annual Income'
# print(df['Annual Income'].mean())

# сколько колонок во фрейме
# print(len(list(df.columns.values)))
# print(f'{df.shape[1]} - колонок')
# print(f'{df.shape[0]} - строк')

# выведем численные колонки (численные признаки)
# print(df.select_dtypes(include='integer').columns.values.tolist())

# выведем численные колонки (численные признаки)
# print(df.select_dtypes(include='object').columns.values.tolist())

# построим график распределения целевой переменной
np.random.seed(1)
fact_target_variable = df['Spending Score']
# fact_target_variable_filter = fact_target_variable. loc [ lambda x : x <300000]
# plt.hist(fact_target_variable, edgecolor='black', bins=20)
# plt.hist(fact_target_variable_filter, edgecolor='black', bins=20)
# plt.show() #унимодально, но есть правый хвост. плохо видно, лучше бы построить линию а не гистограмму и убрать хвост.


from scipy import stats

# sample data generation
np.random.seed(42)
data = sorted(stats.lognorm.rvs(s=0.5, loc=1, scale=1000, size=1000))
# print(data)
target_variable_list = sorted(fact_target_variable.tolist())
# print(target_variable_list)
# fit lognormal distribution
shape, loc, scale = stats.lognorm.fit(target_variable_list, loc=0)
pdf_lognorm = stats.lognorm.pdf(target_variable_list, shape, loc, scale)

# fit normal distribution
mean, std = stats.norm.fit(target_variable_list, loc=0)
pdf_norm = stats.norm.pdf(target_variable_list, mean, std)

# fit weibull distribution
shape, loc, scale = stats.weibull_min.fit(target_variable_list, loc=0)
pdf_weibull_min = stats.weibull_min.pdf(target_variable_list, shape, loc, scale)
# visualize

# fig, ax = plt.subplots(figsize=(8, 4))
# ax.hist(target_variable_list, bins='auto', density=True)
# ax.plot(target_variable_list, pdf_lognorm, label='lognorm')
# ax.plot(target_variable_list, pdf_norm, label='normal')
# ax.plot(target_variable_list, pdf_weibull_min, label='Weibull_Min')
# ax.set_xlabel('Значения Оценки расходов (Spending Score)')
# ax.set_ylabel('Доля оценки расходов в ГС ')
# ax.legend()
# plt.show()
'''значения распределены ненормально, но есть однозначная тенденция распределения, данные плоские. То есть каждая 
оценка представлена практически в равной доли... и как тут и что предсказывать - посмотрим...
логика может быть такой - все данные категориально ровные, по этому результаты ровные 
условно у всех врачей целевой показатель примерно 50000, у всех инженеров 10000 и группы примерно равны
интересно посмотреть на корреляцию. тогда все признаки должны примерно одинаково коррелировать с целевой переменной
либо коррелировать должен только один признак...'''

''' исследуем категориальные колонки на пропущенные значения'''
categorical_data = df.select_dtypes(include='object')  # колонки ['Gender', 'Profession']
categorical_missing_data_columns = categorical_data.isnull().sum()  # Gender - 0, Profession - 35
categorical_missing_data_columns = categorical_missing_data_columns.sort_values(ascending=False)
# print(categorical_missing_data_columns)

# Что будем делать с пропущенными значениями в колонке Профессия... я бы удалил 35 строк если честно
df = df[df['Profession'].notna()]
# print(df.shape)

''' исследуем численные колонки на пропущенные значения'''
numeric_data = df.select_dtypes(include=['integer', 'float'])  # ['Age', 'Annual income', Work Experience, Family size]
# numeric_missing_data_columns = numeric_data.isnull().sum()                              # все по нулям

'''устранили пропущенные значения'''

# переходим к инжинирингу данных
# разобьем колонку пол на 2 колонки с 0 и 1
# dummy_values = pd.get_dummies(df['Gender'], prefix='', drop_first=True).rename(columns={'_Male':'Gender'})
df['Gender'] = pd.get_dummies(df['Gender'], prefix='', drop_first=True).rename(columns={'_Male': 'Gender'})
# print(df.head())

# перейдем к отбору признаков feature selection
# посмотрим корреляцию между признаками и признаков с целевой переменной
# numerical_cols = cleaning_data.select_dtypes(include = ['integer', 'float'])
# numerical_cols_corr = numeric_data.corr()
# plt.figure(figsize = (12,10))
# sns.heatmap(numerical_cols_corr,annot=True, cmap='viridis')
# # plt.show()

# получается странная ситуация, самая высокая корреляция 0,097.Количественные Признаки ни друг с другом,
# ни с исследуемым признаком не коррелируют. Тем не менее есть наверное вероятность как то обучаться и предсказывать

profession_dummies = pd.get_dummies(df['Profession'], prefix='', drop_first=True)
df = pd.concat([df, profession_dummies], axis=1)
df = df.drop('Profession', axis=1)

'''категориальную колонку Professsion преобразовали в фиктивные колонки 0/1 
в итоге получилось 14 колонок, 6 основных и 8 фиктивных из колонки профессия'''

# сколько строк с возрастом 0 лет?
# print((df['Age']==0).sum())
# 23 строчки
# подставим вместо нулей средний возраст по всему фрейму - 49 лет
mean_age = df['Age'].mean().round()

df['Age'].replace(0, mean_age)
df['Age'].replace(0, mean_age, inplace=True)

# сколько строк с Годовым доходом 0 долларов?
# print((df['Annual Income']==0).sum()) 2 строки
# удалим их
df = df.loc[df['Annual Income'] != 0]
print(df.head())
