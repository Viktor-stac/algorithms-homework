import numpy as np


# ============================================================
# 1) МЕТОД ПРОГОНКИ (для трёхдиагональных систем)
# ============================================================

def tridiagonal_solve(a, b, c, d):
    """
    Решает трёхдиагональную систему методом прогонки (алгоритм Томаса).

    Система имеет вид (для i = 0..n-1):
        a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i]

    Параметры:
        a - поддиагональ, a[0] не используется (обычно 0)
        b - главная диагональ
        c - наддиагональ, c[n-1] не используется (обычно 0)
        d - правая часть

    Возвращает:
        x - решение системы
    """
    n = len(d)
    a = a.copy().astype(float)
    b = b.copy().astype(float)
    c = c.copy().astype(float)
    d = d.copy().astype(float)

    # --- Прямой ход: приводим к виду x[i] = P[i]*x[i+1] + Q[i] ---
    P = np.zeros(n)
    Q = np.zeros(n)

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * P[i - 1]
        if i < n - 1:
            P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom

    # --- Обратный ход: находим x[i] от последнего к первому ---
    x = np.zeros(n)
    x[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


def make_full_matrix_from_diagonals(a, b, c):
    """Вспомогательная функция: собирает полную матрицу A из трёх диагоналей (для проверки)"""
    n = len(b)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = b[i]
        if i > 0:
            A[i, i - 1] = a[i]
        if i < n - 1:
            A[i, i + 1] = c[i]
    return A


# ============================================================
# 2) МЕТОД ПРОСТОЙ ИТЕРАЦИИ
# ============================================================

def simple_iteration_solve(A, b, tol=1e-10, max_iter=1000):
    """
    Решает систему A @ x = b методом простой итерации.

    Система приводится к виду x = B @ x + g, где:
        B = E - D^{-1} @ A   (D - диагональ A)
        g = D^{-1} @ b

    Метод сходится, если норма B меньше 1 (например, при диагональном
    преобладании матрицы A).

    Возвращает:
        x - приближённое решение
        iterations - число выполненных итераций
    """
    n = len(b)
    D = np.diag(A)

    if np.any(D == 0):
        raise ValueError("На диагонали есть нули - метод простой итерации в таком виде неприменим")

    D_inv = 1.0 / D
    B = np.eye(n) - D_inv[:, None] * A
    g = D_inv * b

    # Проверка достаточного условия сходимости (не обязательное, но полезное предупреждение)
    norm_B = np.linalg.norm(B, ord=np.inf)
    if norm_B >= 1:
        print(f"Внимание: норма матрицы B = {norm_B:.3f} >= 1, "
              f"сходимость не гарантирована (нет диагонального преобладания)")

    x = np.zeros(n)
    for iteration in range(1, max_iter + 1):
        x_new = B @ x + g
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, iteration
        x = x_new

    print(f"Внимание: метод не сошёлся за {max_iter} итераций")
    return x, max_iter


# ============================================================
# Примеры использования и проверка
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("МЕТОД ПРОГОНКИ")
    print("=" * 60)

    # Трёхдиагональная система 5x5
    n = 5
    a = np.array([0, 1, 1, 1, 1], dtype=float)   # поддиагональ (a[0] не используется)
    b = np.array([4, 4, 4, 4, 4], dtype=float)   # главная диагональ
    c = np.array([1, 1, 1, 1, 0], dtype=float)   # наддиагональ (c[-1] не используется)
    d = np.array([5, 6, 6, 6, 5], dtype=float)   # правая часть

    x_progonka = tridiagonal_solve(a, b, c, d)
    print("Решение методом прогонки: ", x_progonka)

    A_full = make_full_matrix_from_diagonals(a, b, c)
    x_check = np.linalg.solve(A_full, d)
    print("Решение через numpy:      ", x_check)
    print("Максимальная разница:     ", np.max(np.abs(x_progonka - x_check)))

    print()
    print("=" * 60)
    print("МЕТОД ПРОСТОЙ ИТЕРАЦИИ")
    print("=" * 60)

    # Система с диагональным преобладанием (условие сходимости)
    A_iter = np.array([
        [10.0, 1.0, 1.0],
        [2.0, 12.0, 1.0],
        [1.0, 1.0, 15.0]
    ])
    b_iter = np.array([12.0, 15.0, 17.0])

    x_iter, iterations = simple_iteration_solve(A_iter, b_iter)
    print(f"Решение методом простой итерации ({iterations} итераций): {x_iter}")

    x_iter_check = np.linalg.solve(A_iter, b_iter)
    print(f"Решение через numpy:                        {x_iter_check}")
    print("Максимальная разница:                       ", np.max(np.abs(x_iter - x_iter_check)))
