# ============================================================
# Часть 1. Решение системы уравнений
# ============================================================
# Ищем коэффициенты квадратурной формулы с тремя узлами a, (a+b)/2, b:
#     integral_a^b f(x) dx  ~  A*f(a) + B*f((a+b)/2) + C*f(b),
# которая точна для многочленов степени 0, 1 и 2.
# Если сдвинуть начало в точку a, то для f = 1, x, x^2 получаем три уравнения.

import sympy as sp

# Объявляем переменные (a_sym, b_sym - символьные, чтобы не путать с числами ниже)
a_sym, b_sym = sp.symbols('a b')
A, B, C = sp.symbols('A B C')
h_sym = sp.symbols('h', positive=True)

# Записываем уравнения системы
eq1 = sp.Eq(b_sym - a_sym, A + B + C)

eq2 = sp.Eq(
    (b_sym - a_sym)**2 / 2,
    B * (b_sym - a_sym) / 2 + C * (b_sym - a_sym)
)

eq3 = sp.Eq(
    (b_sym - a_sym)**3 / 3,
    B * ((b_sym - a_sym) / 2)**2 + C * (b_sym - a_sym)**2
)

# Решаем систему
solution = sp.solve((eq1, eq2, eq3), (A, B, C))

# Выводим найденные коэффициенты через h = b - a
print("h = b - a, h != 0")
for variable in (A, B, C):
    value = sp.simplify(solution[variable].subs(b_sym, a_sym + h_sym))
    print(f"{variable} = {value}")

# Проверяем решение подстановкой
residuals = [
    sp.simplify((eq.lhs - eq.rhs).subs(solution))
    for eq in (eq1, eq2, eq3)
]
print("Проверка: разности левой и правой частей =", residuals)

# Сравниваем с формулой Симпсона: веса h/6, 4h/6, h/6
simpson_weights = (h_sym / 6, 2 * h_sym / 3, h_sym / 6)
is_simpson = all(
    sp.simplify(solution[v].subs(b_sym, a_sym + h_sym) - w) == 0
    for v, w in zip((A, B, C), simpson_weights)
)
print("Совпадает с формулой Симпсона (h/6, 2h/3, h/6):", is_simpson)


# ============================================================
# Часть 2. Вычисление интеграла sin(x)/x на [1, 2]
# ============================================================

# Подключаем библиотеки для математических вычислений
import numpy as np
from scipy.integrate import quad


# Задаём функцию, которую будем интегрировать
def f(x):
    return np.sin(x) / x


# Задаём границы интегрирования
a = 1
b = 2

# Количество подотрезков
n = 6

# Вычисляем шаг разбиения
h = (b - a) / n


# Формулы с 1, 2 и 3 узлами на каждом подотрезке
def panel_rule(nodes):
    result = 0

    # Проходим по всем подотрезкам
    for i in range(n):
        x = a + i * h

        # Один узел: средняя точка (прямоугольник)
        if nodes == 1:
            result += h * f(x + h / 2)

        # Два узла: трапеция
        elif nodes == 2:
            result += h / 2 * (f(x) + f(x + h))

        # Три узла: Симпсон (веса из системы выше: h/6, 4h/6, h/6)
        elif nodes == 3:
            result += h / 6 * (f(x) + 4 * f(x + h / 2) + f(x + h))

    # Возвращаем полученное значение интеграла
    return result


# Составная формула трапеций
def trapezoid():

    # Учитываем значения функции на границах
    result = (f(a) + f(b)) / 2

    # Складываем значения функции во внутренних точках
    for i in range(1, n):
        result += f(a + i * h)

    # Умножаем сумму на шаг разбиения
    return result * h


# Составная формула Симпсона (n должно быть чётным)
def simpson():

    # Начальная сумма значений на границах
    result = f(a) + f(b)

    # Перебираем внутренние точки
    for i in range(1, n):

        # Чётные точки умножаем на 2
        if i % 2 == 0:
            result += 2 * f(a + i * h)

        # Нечётные точки умножаем на 4
        else:
            result += 4 * f(a + i * h)

    # Возвращаем результат по формуле Симпсона
    return result * h / 3


# Составная формула 3/8 (n должно делиться на 3)
def three_eighths():

    # Учитываем значения на границах
    result = f(a) + f(b)

    # Перебираем внутренние точки
    for i in range(1, n):

        # Если номер точки кратен 3, коэффициент равен 2
        if i % 3 == 0:
            result += 2 * f(a + i * h)

        # Для остальных точек коэффициент равен 3
        else:
            result += 3 * f(a + i * h)

    # Возвращаем результат по формуле 3/8
    return result * 3 * h / 8


# Вычисляем эталонное значение интеграла
# для сравнения с приближёнными результатами
exact, _ = quad(f, a, b)

# Сохраняем результаты в словарь: название -> (значение, число различных узлов)
methods = {
    "Средняя точка (1 узел на отрезок)": (panel_rule(1), n),
    "Трапеции (2 узла на отрезок)": (panel_rule(2), n + 1),
    "Симпсон (3 узла на отрезок, шаг h/2)": (panel_rule(3), 2 * n + 1),
    "Трапеции (составная)": (trapezoid(), n + 1),
    "Симпсон (составная)": (simpson(), n + 1),
    "Правило 3/8": (three_eighths(), n + 1),
}

# Выводим эталонное значение интеграла
print()
print("Интеграл sin(x)/x на [1, 2], n =", n)
print("Точное значение:", round(exact, 10))
print()

# Перебираем результаты каждого метода
for name, (value, nodes) in methods.items():

    # Абсолютная погрешность
    error = abs(exact - value)
    print(f"{name}: {value:.10f}, погрешность = {error:.10f}, узлов f: {nodes}")

# Самый точный метод вообще
best = min(methods, key=lambda name: abs(exact - methods[name][0]))
print("\nСамый точный метод (без учёта числа узлов):", best)

# Честное сравнение: только методы, использующие одинаковое число узлов (n + 1)
same_nodes = {k: v for k, v in methods.items() if v[1] == n + 1}
best_same = min(same_nodes, key=lambda name: abs(exact - same_nodes[name][0]))
print(f"Самый точный среди методов с {n + 1} узлами:", best_same)
