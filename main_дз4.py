import math


def f(x, r):
    """Логистическое отображение."""
    return r * x * (1.0 - x)


def g_and_derivative(r, n_iter, x0=0.5):
    """
    Считаем g(r) = f^(n_iter)(x0) - x0 и её производную по r.
    Производная считается по правилу цепочки одновременно с итерациями:
        x_{k+1} = r*x_k*(1-x_k)
        dx_{k+1}/dr = x_k*(1-x_k) + r*(1-2*x_k)*dx_k/dr
    """
    x = x0
    dx = 0.0
    for _ in range(n_iter):
        dx = x * (1.0 - x) + r * (1.0 - 2.0 * x) * dx
        x = f(x, r)
    return x - x0, dx


def find_superstable(r_guess, n_iter, tol=1e-15, max_iter=50):
    """
    Ищем параметр r, при котором x = 0.5 (критическая точка) является
    периодической точкой периода n_iter, т.е. решаем f^(n_iter)(0.5) = 0.5.
    Это "сверхустойчивый цикл" - он лежит внутри каждого окна периода 2^n.
    Метод - Ньютона, начальное приближение берём из экстраполяции.
    """
    r = r_guess
    for _ in range(max_iter):
        g, dg = g_and_derivative(r, n_iter)
        step = g / dg
        r -= step
        if abs(step) < tol:
            break
    return r


# Первые два значения известны точно:
#   R0 = 2          (период 1: f(0.5) = 0.5)
#   R1 = 1 + sqrt5  (период 2)
r_values = [2.0, 1.0 + math.sqrt(5.0)]

print("Поиск параметров сверхустойчивых циклов периода 2^n")
print("-" * 52)

delta = 4.4  # грубое начальное приближение для δ (потом уточняется)
for n in range(2, 9):
    # Экстраполяция: каждый следующий шаг в δ раз меньше предыдущего
    r_guess = r_values[-1] + (r_values[-1] - r_values[-2]) / delta

    # Число итераций растёт как 2^n (период цикла)
    n_iter = 2 ** n

    r_new = find_superstable(r_guess, n_iter)
    r_values.append(r_new)

    # Оценка константы Фейгенбаума по последним трём точкам
    delta = (r_values[-2] - r_values[-3]) / (r_values[-1] - r_values[-2])
    print(f"n = {n:2d} | r = {r_values[-1]:.10f} | δ ≈ {delta:.8f}")

print()
print(f"Оценка константы Фейгенбаума: δ ≈ {delta:.8f}")
print(f"Теоретическое значение:        δ ≈ 4.66920161")
print(f"Относительная погрешность:     {abs(delta - 4.66920161) / 4.66920161:.2e}")
