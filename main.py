import logging
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

# Установим уровень логирования на WARNING, чтобы исключить отладочные сообщения
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def turn_memory_into_GB(value):
    if "GB" in value:
        return float(value[: value.find("GB")]) * 1000
    elif "TB" in value:
        return float(value[: value.find("TB")]) * 10**6

# Гипотеза: Цена ноутбука зависит от множества характеристик,
# включая бренд, тип ноутбука, объем оперативной памяти, тип хранилища и разрешение экрана.

def analys_start():
    """
    Анализируем данные
    """
    logger.info("Начинаем анализ данных ...")
    file_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "laptop_price_dataset.csv"
    )
    # Читаем CSV файл
    df: pd.DataFrame = pd.read_csv(file_path)

    # -------------------------------- Провожу очистку и подготовку данных --------------------------------
    # убираю столбец Product
    df = df.drop("Product", axis=1)

    df = df.join(pd.get_dummies(df.Company))
    df = df.drop("Company", axis=1)

    df = df.join(pd.get_dummies(df.TypeName))
    df = df.drop("TypeName", axis=1)

    # Разбиваем столбец ScreenResolution (разрешение экрана) на ширину и высоту
    df["ScreenResolution"] = df.ScreenResolution.str.split().apply(lambda x: x[-1])
    df["Screen Width"] = df.ScreenResolution.str.split("x").apply(lambda x: x[0])
    df["Screen Height"] = df.ScreenResolution.str.split("x").apply(lambda x: x[1])
    # приводим Screen Width и Height к типу int
    df["Screen Width"] = df["Screen Width"].astype("int")
    df["Screen Height"] = df["Screen Height"].astype("int")
    # убираю столбец ScreenResolution
    df = df.drop("ScreenResolution", axis=1)

    # переименуем столбцы
    df["CPU Brand"] = df["CPU_Company"]
    df["CPU Frequency"] = df["CPU_Frequency (GHz)"]

    # удаляем старые столбцы CPU_Company и CPU_Frequency (GHz)
    df = df.drop(columns=["CPU_Company", "CPU_Frequency (GHz)"])

    # переименуем столбец
    df["RAM"] = df["RAM (GB)"]
    # удалим старый столбец
    df = df.drop("RAM (GB)", axis=1)

    df["Memory Amount"] = df.Memory.str.split(" ").apply(lambda x: x[0])
    df["Memory Type"] = df.Memory.str.split(" ").apply(lambda x: x[1])
    df["Memory Amount"] = df["Memory Amount"].apply(turn_memory_into_GB)
    df = df.drop("Memory Type", axis=1)
    df = df.drop("Memory", axis=1)

    df["Weight"] = df["Weight (kg)"]
    df = df.drop("Weight (kg)", axis=1)

    df["GPU Brand"] = df["GPU_Company"]
    df = df.drop("GPU_Company", axis=1)

    df = df.join(pd.get_dummies(df.OpSys))
    df = df.drop("OpSys", axis=1)

    cpu_categories = pd.get_dummies(df["CPU Brand"])
    cpu_categories.columns = [col + "_CPU" for col in cpu_categories.columns]

    df = df.join(cpu_categories)
    df = df.drop("CPU Brand", axis=1)

    gpu_categories = pd.get_dummies(df["GPU Brand"])
    gpu_categories.columns = [col + "_GPU" for col in gpu_categories.columns]

    df = df.join(gpu_categories)
    df = df.drop("GPU Brand", axis=1)

    df = df.drop(columns=["CPU_Type", "GPU_Type"])

    target_correlations = df.corr()["Price (Euro)"].apply(abs).sort_values()

    # Определяем 20 наиболее значимых признаков (с самой высокой корреляцией с Price (Euro))
    selected_features = target_correlations[-20 :].index
    selected_features = list(selected_features)

    limited_df = df[selected_features]

    # Строю корреляционную матрицу ограниченной выборки из 20 параметров
    plt.figure(figsize=(18, 15))
    sns.heatmap(limited_df.corr(), annot=True, cmap="YlGnBu")
    plt.show()

    # Разделяем признаки на факторные (X) и результативный (y)
    X, y = limited_df.drop("Price (Euro)", axis=1), limited_df["Price (Euro)"]

    # Разделяю данные на обучающую и тестовые выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    scaler = StandardScaler()

    # fit_transform вычисляет параметры масштабирования
    # (среднее, стандартное отклонение) на обучающей выборке
    # и масштабирует её
    X_trained_scaled = scaler.fit_transform(X_train)

    # transform масштабирует тестовую выборку,
    # используя параметры на обучающей
    X_test_scaled = scaler.transform(X_test)

    # Обучение модели RandomForestRegressor на масштабированных данных
    forest = RandomForestRegressor()
    forest.fit(X_trained_scaled, y_train)

    print(f"Оценка качества модели на тестовой выборке: {forest.score(X_test_scaled, y_test)}")

    # предсказание модели на тестовой выборке
    y_pred = forest.predict(X_test_scaled)

    # Визуализирую предсказанные и реальные значения
    plt.figure(figsize=(12, 8))
    plt.scatter(y_pred, y_test)
    plt.plot(range(0, 6000), range(0, 6000), c="red")
    plt.show()

    # Тестирую модель на одном экземпляре из тестовой выборки
    X_new_scaled = scaler.transform([X_test.iloc[0]])
    print(f"Предсказанная цена для одного экземпляра: {forest.predict(X_new_scaled)}")
    print(f"Значение целевой переменной для одного экземпляра: {y_test.iloc[0]}")

    # Итог
    # После построения корреляционной матрицы я:
    # 	1.	Анализирую корреляции, чтобы выбрать значимые признаки.
    # 	2.	Сужаю набор данных до наиболее значимых признаков.
    # 	3.	Обучаю модель на подготовленных данных.
    # 	4.	Оцениваю её качество и интерпретирую предсказания.


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stdout,
        format="%(asctime)s | %(name)s | %(levelname)s | %(lineno)d | %(message)s",
        datefmt="%I:%M:%S"
    )
    analys_start()
