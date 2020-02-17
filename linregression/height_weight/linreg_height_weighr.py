import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.optimize as scopt
'''
Шаг 1: Первичный анализ данных c Pandas
'''
data = pd.read_csv('weights_heights.csv', index_col='Index')
print(data.head(5))

data.plot(y='Height', kind='hist', color='red',  title='Height (inch.) distribution')
plt.show()
data.plot(y='Weight', kind='hist', color='green',  title='Weight (pounds) distribution')
plt.show()

#Добавим еще один столбик к таблице "Индекс массы тела"
def make_bmi(height_inch, weight_pound):
    METER_TO_INCH, KILO_TO_POUND = 39.37, 2.20462
    return (weight_pound / KILO_TO_POUND) / (height_inch / METER_TO_INCH) ** 2

data['BMI'] = data.apply(lambda row: make_bmi(row['Height'],row['Weight']), axis=1)

print(data.head(5))

# Постройте картинку, на которой будут отображены попарные зависимости признаков , 'Height', 'Weight' и 'BMI' друг от друга.
sns.set(style="ticks", color_codes=True)
sns.pairplot(data)
plt.show()

#Создайте в DataFrame *data* новый признак *weight_category*, который будет иметь 3 значения: 1 – если вес меньше 120 фунтов. (~ 54 кг.), 3 - если вес  больше или равен 150 фунтов (~68 кг.), 2 – в остальных случаях. Постройте «ящик с усами» (boxplot), демонстрирующий зависимость роста от весовой категории. Используйте метод *boxplot* библиотеки Seaborn и метод *apply* Pandas DataFrame. Подпишите ось *y* меткой «Рост», ось *x* – меткой «Весовая категория»."
def make_weight_category(weight_pound):
    if weight_pound < 120:
        return 1
    elif weight_pound >= 150:
        return 3
    else:
        return 2

data['WeightCategory'] = data.apply(lambda row: make_weight_category(row['Weight']), axis=1)
#Постройте «ящик с усами» (boxplot), демонстрирующий зависимость роста от весовой категории. Используйте метод *boxplot* библиотеки Seaborn. Подпишите ось *y* меткой «Рост», ось *x* – меткой «Весовая категория»."
print(data.head(5))
sns.boxplot(y="Height", x="WeightCategory", data=data)
plt.show()

data.plot(y="Height", x="Weight",kind='scatter', title='Зависимость роста от веса')
plt.show()

'''
Шаг 2: Минимизация квадратичной ошибки
'''
# весовая функция
def y(weight, w0, w1):
    return w0 + w1*weight
# ошибка в вычислениях весовой функции
def error(w1, w0=50):
    errorScore = data.apply(lambda row: (row["Height"] - y(row["Weight"], w0, w1))**2, axis=1)
    return errorScore.sum()

firsPlotHeight = data.apply(lambda row: y(row['Weight'], 60, 0.05), axis=1)
secondPlotHeight = data.apply(lambda row: y(row['Weight'], 50, 0.16), axis=1)

plt.plot(data.Weight, data.Height, "o", data.Weight, firsPlotHeight, 'y', data.Weight, secondPlotHeight, 'g')
plt.show()
#Подбираем веса так чтобы минимизировать ошиьббку
w_1_set = np.linspace(-5, 5, 10)
errorSores = [error(w1) for w1 in w_1_set]

#Посмотрим на график зависимоти ошибки от веса w1
plt.plot(w_1_set, errorSores, 'b')
plt.show()

#Найдем w1 оптимальное
res = scopt.minimize_scalar(error, bounds=(-5, 5))
w1_opt = res.x
optHeightParams = data.apply(lambda row: y(row['Weight'], 50, w1_opt), axis=1)
plt.plot(data.Weight, data.Height, "o", data.Weight, optHeightParams, 'r')
plt.show()

'''
Урааа!!)) Теперь немного 3D картинок и поиск минимума по обоим параметрам
'''
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d') # get current axis

# # Создаем массивы NumPy с координатами точек по осям X и У. \n",
# # Используем метод meshgrid, при котором по векторам координат \n",
# # создается матрица координат. Задаем нужную функцию Z(x, y).\n",
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# Z = np.sin(np.sqrt(X**2 + Y**2))
# print('sin: ', Z)
#
# # Наконец, используем метод *plot_surface* объекта типа Axes3DSubplot. Также подписываем оси.\n",
# surf = ax.plot_surface(X, Y, Z)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

# Построим график зависимости функции ошибки от w1 и  w0

w_0_set = np.linspace(-100, 100, 10)
W0, W1 = np.meshgrid(w_0_set, w_1_set)
errorScores2 = error(W1,W0)
surf = ax.plot_surface(W0, W1, errorScores2)
ax.set_xlabel('W0')
ax.set_ylabel('W1')
ax.set_zlabel('Error')
plt.show()

def error2(x):
    w0, w1 = x
    errorScore = data.apply(lambda row: (row["Height"] - y(row["Weight"], w0, w1))**2, axis=1)
    return errorScore.sum()

#Подбираем веса так чтобы минимизировать ошибку
bounds = ((-100, 100), (-5, 5))
res2 = scopt.minimize(error2, np.array([0, 0]), method="L-BFGS-B", bounds=bounds)
print(res2.x)
w0_opt2, w1_opt2 = res2.x
optHeightParams2 = data.apply(lambda row: y(row['Weight'], w0_opt2, w1_opt2), axis=1)

plt.plot(data.Weight, data.Height, "o", data.Weight, optHeightParams2, 'y')
plt.show()