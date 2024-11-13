import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.impute import SimpleImputer
#
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from sklearn.model_selection import train_test_split
#
# from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge, ElasticNet
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, AdaBoostRegressor
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor
#
# from sklearn.metrics import mean_squared_error


# Гипотеза о корреляции между реакциями и настроением:
# Можно предположить, что положительные реакции (“rocket”, “buy-up”)
# связаны с положительными комментариями, а отрицательные
# (“dislike”, “not-convinced”) — с негативными.
# Проведите анализ тональности текста и посмотрите, коррелирует ли он с типом реакций.

logger = logging.getLogger(__name__)

# Установим уровень логирования на WARNING, чтобы исключить отладочные сообщения
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def analys_start():
    """
    Анализируем данные
    """
    logger.info("Начинаем анализ данных ...")
    file_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "laptop_price_dataset.csv"
    )
    # Читаем CSV файл
    df = pd.read_csv(file_path)

    logger.info(f'Дублирующиеся значения: {df.duplicated().sum().item()}')
    df_description_all = df.describe(include='all')

    # ------------------------------------ Анализ по разным атрибутам ноутбуков ------------------------------------
    # Проанализируем кол-во ноутбуков по каждому бренду, т.е. сколько ноутбуков у каждого бренда
    company = df['Company'].value_counts()

    sns.set_style('darkgrid', {"grid.color": "0.6", "grid.linestyle": ":"})
    company = company.sort_values(ascending=True)
    company.plot(kind='barh', edgecolor='#1F77B4', title='Количество ноутбуков по каждому бренду')

    # Отобразим график количества компаний-производителей ноутбуков
    plt.xlabel("Количество ноутбуков")
    plt.ylabel("Бренды")
    plt.xticks(range(0, company.max() + 1, 10))
    plt.show()

    company_count_df: pd.DataFrame = pd.DataFrame(company.sort_values(ascending=False))

    # Проанализируем кол-во ноутбуков по типу ноутбука, т.е. сколько ноутбуков каждого типа
    # Типы ноутбуков:
    # Notebook - обычный ноутбук (стандартный),
    # Gaming - игровой ноутбук (для компьютерных игр),
    # Ultrabook - ультрабук (компактный, тонкий по толщине, производительный для такого формата),
    # 2 in 1 Convertible - ноутбук 2 в 1 (ноутбук-трансформер),
    # Workstation - ноутбук-рабочая станция (большая диагональ, хороший аккумулятор - отлично подходит для долгой работы),
    # Netbook -
    type_name = df['TypeName'].value_counts()

    sns.set_style('darkgrid', {"grid.color": "0.6", "grid.linestyle": ":"})
    type_name.plot(kind='bar', color='#BD3A3C', edgecolor='#BD3A3C', title='Количество ноутбуков по типам')
    plt.xlabel("Типы ноутбуков")
    plt.ylabel("Кол-во ноутбуков")
    plt.xticks(rotation=30)
    plt.show()

    type_laptops_df: pd.DataFrame = pd.DataFrame(type_name)

    # Проанализируем разрешения экранов
    resolution = df['ScreenResolution'].value_counts()

    plt.figure(figsize=(8, 10))
    sns.set_style('darkgrid', {"grid.color": "0.6", "grid.linestyle": ":"})
    resolution = resolution.sort_values(ascending=True)
    resolution.plot(kind='barh', color='#C4B324', edgecolor='#C4B324', title='Количество ноутбуков разных разрешений экранов')
    plt.xlabel("Количество ноутбуков")
    plt.ylabel("Разрешения экранов")
    plt.show()

    screen_resolutions_df: pd.DataFrame = pd.DataFrame(resolution.sort_values(ascending=False))

    # Проанализируем по производителям процессоров
    cpu_company = df['CPU_Company'].value_counts()

    cpu_company.plot(kind='bar', color='gray', edgecolor='black', width=0.3, title='Количество ноутбуков по производителям процессора (CPU)')
    plt.show()

    cpu_company_df: pd.DataFrame = pd.DataFrame(cpu_company)

    # Проанализируем по частоте процессора ноутбука
    freq = df['CPU_Frequency (GHz)'].value_counts()

    plt.figure(figsize=(7, 7))
    sns.set_style('darkgrid', {"grid.color": "0.6", "grid.linestyle": ":"})
    freq = freq.sort_values(ascending=True)
    freq.plot(kind='barh', color='gray', edgecolor='black', title='Количество ноутбуков по частоте процессора')
    plt.xlabel("Количество ноутбуков с частотой процессора")
    plt.ylabel("Частота процессора")
    plt.show()

    freq_df: pd.DataFrame = pd.DataFrame(freq.sort_values(ascending=False))

    # Проанализируем количество оперативной памяти
    ram = df['RAM (GB)'].value_counts()

    sns.set_style('darkgrid', {"grid.color": "0.6", "grid.linestyle": ":"})
    ram.plot(kind='bar', color='gray', edgecolor='black', title="Количество по оперативной памяти")
    plt.xlabel("Оперативная память")
    plt.ylabel("Количество ноутбуков")
    plt.xticks(rotation=0)
    plt.show()

    ram_df: pd.DataFrame = pd.DataFrame(ram)

    # Количество памяти
    memory = df['Memory'].value_counts()

    plt.figure(figsize=(8, 10))
    sns.set_style('darkgrid', {"grid.color": "0.6", "grid.linestyle": ":"})
    memory = memory.sort_values(ascending=True)
    memory.plot(kind='barh', color='#2B9479', edgecolor='#2B9479', title="Количество по памяти")
    plt.xlabel("Количество ноутбуков")
    plt.ylabel("Память")
    plt.show()

    memory_df: pd.DataFrame = pd.DataFrame(memory.sort_values(ascending=False))

    # Проанализируем производителей графических процессоров
    gpu_company = df['GPU_Company'].value_counts()

    gpu_company.plot(kind='bar', color='#BA9329', edgecolor='#BA9329', width=0.3, title='Количество по графическим процессорам')
    plt.xlabel("Бренды")
    plt.ylabel("Количество графич. процессоров")
    plt.xticks(rotation=0)
    plt.show()

    gpu_company_df: pd.DataFrame = pd.DataFrame(gpu_company)

    # Проанализируем по операционным системам
    op_sys = df['OpSys'].value_counts()

    plt.figure(figsize=(10, 4))
    op_sys.plot(kind='bar', color='#BA9329', edgecolor='#BA9329', title='Количество по операционным системам')
    plt.xticks(rotation=0)
    plt.show()

    op_sys_df: pd.DataFrame = pd.DataFrame(op_sys)

    # Проанализируем веса ноутбуков
    weight = df['Weight (kg)']

    # plt.figure(figsize=(10,4))
    sns.histplot(weight, kde=True)
    plt.xticks(rotation=0)
    plt.show()

    weight_df: pd.DataFrame = pd.DataFrame(weight.describe())

    # Проанализируем по цене
    price_euro = df['Price (Euro)']

    # plt.figure(figsize=(10,4))
    sns.histplot(price_euro, kde=True)
    plt.xlabel("Тысячи евро")
    plt.ylabel("Количество ноутбуков")
    plt.xticks(rotation=0)
    plt.show()

    pd.DataFrame(price_euro.describe())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stdout,
        format="%(asctime)s | %(name)s | %(levelname)s | %(lineno)d | %(message)s",
        datefmt="%I:%M:%S"
    )
    analys_start()
