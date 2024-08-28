from operator import itemgetter
import numpy as np
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

# генерируем исходные данные: 750 строк-наблюдений и 14 столбцов-признаков
np.random.seed(0)
size = 750
X = np.random.uniform(0, 1, (size, 14))
# Задаем функцию-выход: регрессионную проблему Фридмана
Y = (10 * np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2] - .5)**2 + 10*X[:, 3] + 5*X[:, 4]**5 + np.random.normal(0, 1))
# Добавляем зависимость признаков
X[:, 10:] = X[:, :4] + np.random.normal(0, .025, (size, 4))
names = ["x%s" % i for i in range(1, 15)]  # - список признаков вида ['x1', 'x2', 'x3', ..., 'x14']
ranks = dict()


def rank_to_dict(ranks, names):
     # получение абсолютных значений оценок(модуля)
     ranks = np.abs(ranks)
     minmax = MinMaxScaler()
     # преобразование данных
     ranks = minmax.fit_transform(np.array(ranks).reshape(14, 1)).ravel()
     # округление элементов массива
     ranks = map(lambda x: round(x, 2), ranks)
     # преобразование данных
     return dict(zip(names, ranks))


# Модель: случайное Лассо (RandomizedLasso) - устаревшее, поэтому используем Ridge-регрессия (Ridge Regression)
def ridge_regressions():
    # Создание экземпляра модели Ridge
    ridge_model = Ridge()
    ridge_model.fit(X, Y)
    ranks['Ridge'] = rank_to_dict(ridge_model.coef_, names)


# Модель: рекурсивное сокращение признаков (Recursive Feature Elimination – RFE)
def recursive_feature_elimination():
    # создание модели LinearRegression
    estimator = LinearRegression()
    # создание модели RFE
    rfe_model = RFE(estimator)
    rfe_model.fit(X, Y)
    ranks['Recursive Feature Elimination'] = rank_to_dict_rfe(rfe_model.ranking_, names)


def rank_to_dict_rfe(ranking, names):
    # нахождение обратных значений рангов
    n_ranks = [float(1 / i) for i in ranking]
    # округление элементов массива
    n_ranks = map(lambda x: round(x, 2), n_ranks)
    # преобразование данных
    return dict(zip(names, n_ranks))


# Модель: линейная корреляция (f_regression)
def linear_correlation():
    # вычисление линейной корреляции между X и y
    correlation, p_values = f_regression(X, Y)
    ranks['linear correlation'] = rank_to_dict(correlation, names)


if __name__ == '__main__':
    ridge_regressions()
    recursive_feature_elimination()
    linear_correlation()

    for key, value in ranks.items():  # Вывод нормализованных оценок важности признаков каждой модели
        ranks[key] = sorted(value.items(), key=itemgetter(1), reverse=True)
    for key, value in ranks.items():
        print(key)
        print(value)

    mean = {}  # - нахождение средних значений оценок важности по 3м моделям
    for key, value in ranks.items():
        for item in value:
            if item[0] not in mean:
                mean[item[0]] = 0
            mean[item[0]] += item[1]
    for key, value in mean.items():
        res = value / len(ranks)
        mean[key] = round(res, 2)
    mean = sorted(mean.items(), key=itemgetter(1), reverse=True)
    print("Mean")
    print(mean)