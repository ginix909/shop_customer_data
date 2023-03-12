import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker


def missing_values_processing(df):
    '''2.Обработка пропущенных значений.
    функция удалит лишние колонки, строки с пропущенными значениями'''
    df_copy = df.copy()  # создаем копию направленного в функцию массива
    # удалим 'CustomerID' так как id каждой записи у нас уже есть
    df_copy = df_copy.drop(['CustomerID'], axis=1)

    # исследуем категориальные колонки на пропущенные значения
    categorical_data = df_copy.select_dtypes(include='object')  # колонки ['Gender', 'Profession']
    # удалим строки с пропущенными значениями колонки Профессия
    df_copy = df_copy[df_copy['Profession'].notna()]

    # исследуем численные колонки на нулевые значения
    numeric_cols = df_copy.select_dtypes(include=['int', 'float'])
    # колонки ['Age', 'Annual income','Spending score','work experience','family size' ]
    # некоторые колонки не должны иметь нули
    some_numeric_cols = df[['Age', 'Family Size']]
    # print((df['Age']==0).sum())
    # print((df['Family Size']==0).sum())
    # заменим 0 в колонке Age на средний возраст
    mean_age = df_copy['Age'].mean().round()
    df_copy['Age'].replace(0, mean_age, inplace=True)

    # удалим строки с нулевыми значениями Annual Income, всего их две
    df_copy = df_copy.loc[df['Annual Income'] != 0]
    return df_copy


def feature_engineering(df_copy):
    '''Normalization, Standartization, Feature-encoding, Log-transformation, Principal component analisys, Data imputation
    Feature-encoding: сделали из двух колонок - колонки с фиктивными переменными'''
    df_copy2 = df_copy.copy()
    # разобьем колонку Gender на 2 колонки с 0 и 1
    df_copy2['Gender'] = pd.get_dummies(df_copy2['Gender'], prefix='', drop_first=True).rename(
        columns={'_Male': 'Gender'})

    # разобьем колонку Profession на 8 колонок с профессией
    profession_dummies = pd.get_dummies(df_copy2['Profession'], prefix='', drop_first=True)
    df_copy2 = pd.concat([df_copy2, profession_dummies], axis=1)
    df_copy2 = df_copy2.drop('Profession', axis=1)

    return df_copy2


def visualization(df_copy2):
    '''Функуия принимает обработанный датасет для визуализации распределения свободных признаков, коррелляции'''
    # print(df_copy2.describe())
    # разделим область figure на 4 части
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))

    # График распреления признака Age
    ax[0, 0].hist(x=df_copy2['Age'])
    ax[0, 0].set(title='Age',
                 ylabel='Количество значений')

    # график распределения признака Annual Income
    ax[1, 0].hist(x=df_copy2['Annual Income'])
    ax[1, 0].set(title='Annual Income',
                 ylabel='Количество значений')
    ax[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(50000))

    # график распределения признака Spending Score
    ax[0, 1].hist(x=df_copy2['Spending Score'])
    ax[0, 1].set(title='Spending Score',
                 ylabel='Количество значений')

    # график распределения признака Work Experience
    ax[1, 1].hist(x=df_copy2['Work Experience'])
    ax[1, 1].set(title='Work Experience',
                 ylabel='Количество значений')

    # график распределение признака Gender
    ax[2, 1] = sns.countplot(x=df_copy2['Gender'])
    ax[2, 1].set(title='Gender',
                 ylabel='Количество значений')

    # график распределения признака
    ax[2, 0].hist(x=df_copy2['Family Size'])
    ax[2, 0].set(title='Family Size',
                 ylabel='Количество значений')

    # отобразим корреляцию между всеми переменными
    numerical_data = df_copy2.select_dtypes(include=['integer', 'float'])
    numerical_cols_corr = numerical_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(numerical_cols_corr, annot=True, cmap='viridis')

    plt.show()


def conclusions(df_copy2):
    print(f'Датасэт Customers.csv содержит {df_copy2.shape[0]} строк,{df_copy2.shape[1]} свободных признаков')
    print('Целевой признак Spending Score является количественной дискретной величиной.')
    print('Исходя из этого можно применить следующие модели машинного обучения:')
    print('Деревья решений, случайный лес, метод к-средних, понижение размерности')
    print(f'Признаки коррелируют с целевой переменной след образом:')
    correlation_series=(df_copy2.corr())["Spending Score"]
    correlation_series.drop(index='Spending Score', inplace=True)
    print(f'{correlation_series}')


customers_data = pd.read_csv('Customers.csv')
processed_data = feature_engineering(missing_values_processing(df=customers_data))
visualization(processed_data)
conclusions(processed_data)