import os.path
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

picfld = os.path.join('static', 'charts')

X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_informative=2, random_state=None,
                           n_clusters_per_class=1)
# sklearn.datasets.samples_generator.make_classification - используется для создания случайных задач классификации N.
# n_samples - Количество случайных чисел
# n_features - количество признаков (измерений) для каждого числа.
# n_informative - Количество информативных характеристик
# n_redundant -количество избыточных признаков, которые не вносят дополнительной информации.
# random_state - опциональный параметр для установки начального состояния генератора случайных чисел.
# n_clusters_per_class - Количество кластера в каждой категории
# Функция возвращает два значения:
# X: массив размера [n_samples, n_features], содержащий сгенерированные признаки.
# y: массив размера [n_samples], содержащий сгенерированные целевые переменные (классы).

rng = np.random.RandomState(2)
# добавление шума к данным
X += 2 * rng.uniform(size=X.shape)
linearly_dataset = (X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

# Модель: линейная регрессия
def linear_regression():
    # Модель линейной регрессии
    model = LinearRegression()
    # Обучение на тренировочных данных
    model.fit(X_train, y_train)
    # Выполнение прогноза
    y_pred = model.predict(X_test)
    # Вычисление коэффициента детерминации
    r_sq = model.score(X_test, y_test)
    # Создание графика
    plt.plot(y_test, c="#bd0000", label="\"y\" исходная")
    plt.plot(y_pred, c="#00BFFF", label="\"y\" предсказанная \n" "Кд = " + str(r_sq))
    plt.title("Линейная регрессия")
    plt.legend(loc='lower left')
    plt.savefig('static/charts/LinearRegressionChart.png')
    plt.close()


# Модель: полиномиальная регрессия (со степенью 4)
def polynomial_regression():
    # Генерирование объекта полинома,
    # где degree - степень полинома,
    # include_bias - установка вектора смещения в полиномиальные признаки
    pf = PolynomialFeatures(degree=4, include_bias=False)
    # Преобразование исходного набора данных X_train в полиномиальные признаки
    X_poly_train = pf.fit_transform(X_train)
    # Преобразование исходного набора данных X_test в полиномиальные признаки
    X_poly_test = pf.fit_transform(X_test)
    # Модель линейной регрессии
    model = LinearRegression()
    # Обучение модели линейной регрессии на преобразованных полиномиальных признаках
    model.fit(X_poly_train, y_train)
    # Выполнение прогноза
    y_pred = model.predict(X_poly_test)
    # Вычисление коэффициента детерминации
    r_sq = model.score(X_poly_test, y_test)
    # Создание графика
    plt.plot(y_test, c="#bd0000", label="\"y\" исходная")
    plt.plot(y_pred, c="#00BFFF",
             label="\"y\" предсказанная \n" "Кд = " + str(r_sq))
    plt.legend(loc='lower left')
    plt.title("Полиномиальная регрессия")
    plt.savefig('static/charts/PolynomialRegressionChart.png')
    plt.close()


# Модель: персептрон
def perceptron():
    # Модель персептрона
    model = Perceptron()
    # Обучение на тренировочных данных
    model.fit(X_train, y_train)
    # Выполнение прогноза
    y_pred = model.predict(X_test)
    # Вычисление точности работы персептрона
    accuracy = accuracy_score(y_test, y_pred)
    # Создание графика
    plt.plot(y_test, c="#bd0000", label="\"y\" исходная")
    plt.plot(y_pred, c="#00BFFF",
             label="\"y\" предсказанная \n" "Точность = " + str(accuracy))
    plt.legend(loc='lower left')
    plt.title("Персептрон")
    plt.savefig('static/charts/PerceptronChart.png')
    plt.close()


if __name__ == '__main__':
    linear_regression()
    polynomial_regression()
    perceptron()
