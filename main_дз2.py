import numpy as np
import time


def lu_decomposition(A):
    """
    LU-разложение с частичным выбором ведущего элемента: P @ A = L @ U

    Возвращает:
        L - нижнетреугольная матрица с единицами на диагонали
        U - верхнетреугольная матрица
        P - матрица перестановок
    """
    n = A.shape[0]
    A = A.copy().astype(float)
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        # 1) Выбираем строку с максимальным по модулю элементом в столбце k
        # (частичный выбор ведущего элемента для устойчивости)
        pivot_row = np.argmax(np.abs(A[k:, k])) + k
        if pivot_row != k:
            A[[k, pivot_row], :] = A[[pivot_row, k], :]
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            # Переставляем уже вычисленную часть L тоже
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        # 2) Проверка на вырожденность
        if A[k, k] == 0:
            raise ValueError("Матрица вырождена, LU-разложение невозможно")

        # 3) Обнуляем элементы ниже диагонали в столбце k
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            L[i, k] = factor
            A[i, k:] -= factor * A[k, k:]

    U = A
    return L, U, P


def forward_substitution(L, b):
    """Решает L @ y = b для нижнетреугольной L (прямой ход)"""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y


def backward_substitution(U, y):
    """Решает U @ x = y для верхнетреугольной U (обратный ход)"""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def solve_with_lu(L, U, P, b):
    """Решает A @ x = b, используя готовое LU-разложение (PA = LU)"""
    b_permuted = P @ b
    y = forward_substitution(L, b_permuted)
    x = backward_substitution(U, y)
    return x


def solve_naive_gaussian(A, b):
    """
    Решение системы Ax = b 'с нуля' полным исключением Гаусса
    (без переиспользования какого-либо разложения). Используется как
    честная база для сравнения: тот же 'ручной' стиль реализации,
    что и в lu_decomposition, только работа проделывается заново
    для каждого b.
    """
    n = len(b)
    Aug = np.hstack([A.copy().astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n - 1):
        pivot_row = np.argmax(np.abs(Aug[k:, k])) + k
        if pivot_row != k:
            Aug[[k, pivot_row], :] = Aug[[pivot_row, k], :]
        for i in range(k + 1, n):
            factor = Aug[i, k] / Aug[k, k]
            Aug[i, k:] -= factor * Aug[k, k:]

    return backward_substitution(Aug[:, :n], Aug[:, n])


# ========================================
# Пример использования и замер выигрыша по времени
# ========================================
if __name__ == "__main__":
    np.random.seed(42)

    n = 150           # размер матрицы
    num_systems = 150  # сколько разных b решаем с одной и той же A

    # Фиксированная матрица A (делаем её хорошо обусловленной)
    A = np.random.rand(n, n) + n * np.eye(n)

    # Набор правых частей b_1, ..., b_num_systems
    B = [np.random.rand(n) for _ in range(num_systems)]

    # --- Способ 1: LU-разложение один раз, потом быстро решаем все системы ---
    start = time.perf_counter()
    L, U, P = lu_decomposition(A)
    for b in B:
        x_lu = solve_with_lu(L, U, P, b)
    time_lu = time.perf_counter() - start

    # --- Способ 2: наивный подход, каждый раз решаем систему заново (полный Гаусс) ---
    start = time.perf_counter()
    for b in B:
        x_naive = solve_naive_gaussian(A, b)
    time_naive = time.perf_counter() - start

    # --- Проверка корректности (решения должны совпадать) ---
    x_lu_check = solve_with_lu(L, U, P, B[0])
    x_naive_check = solve_naive_gaussian(A, B[0])
    max_diff = np.max(np.abs(x_lu_check - x_naive_check))

    print(f"Размер матрицы: {n}x{n}, количество систем: {num_systems}\n")
    print(f"Время с LU-разложением (1 раз) : {time_lu:.4f} сек")
    print(f"Время наивным способом (каждый раз): {time_naive:.4f} сек")
    print(f"Выигрыш по времени: {time_naive / time_lu:.2f}x быстрее\n")
    print(f"Максимальная разница между решениями LU и naive: {max_diff:.2e}")
    print("(должна быть близка к нулю - значит, решения совпадают)")
