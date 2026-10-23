# Исследуем итерации функции phi(x) = mu * sqrt(x) * (1 - nu * sqrt(x)).
# Для разных (mu, nu):
#   1) считаем итерации и смотрим, сходится процесс или нет;
#   2) строим график phi(x) вместе с y = x;
#   3) строим график x_k по номеру итерации;
#   4) сравниваем с теорией (неподвижная точка и множитель |phi'(x*)|).
#
# Теория. Неподвижная точка: x = phi(x). Пусть s = sqrt(x), тогда
#   s = mu * (1 - nu * s)  =>  s* = mu / (1 + mu*nu),  x* = s*^2.
# Производная: phi'(x) = mu / (2*sqrt(x)) - mu*nu, откуда в точке x*
#   phi'(x*) = (1 - mu*nu) / 2.
# Итерации сходятся к x*, если |phi'(x*)| < 1, то есть при 0 < mu*nu < 3.

import numpy as np
import matplotlib.pyplot as plt


def phi(x, mu, nu):
    s = np.sqrt(x)
    return mu * s * (1 - nu * s)


def fixed_point(mu, nu):
    """Неподвижная точка x* = (mu / (1 + mu*nu))^2."""
    return (mu / (1 + mu * nu)) ** 2


def multiplier(mu, nu):
    """Значение производной phi'(x*) = (1 - mu*nu) / 2."""
    return (1 - mu * nu) / 2


def iterate(mu, nu, x0=0.1, eps=1e-6, max_iter=100):
    # если x0 < 0, то sqrt сразу не посчитать
    if x0 < 0:
        return None, 0, "x0 < 0", []

    x = x0
    values = [x]

    for k in range(1, max_iter + 1):
        x_new = phi(x, mu, nu)
        values.append(x_new)

        # если ушли в минус, дальше sqrt нельзя
        if x_new < 0:
            return x_new, k, "вышли из области sqrt", values

        # проверяем, не сошлось ли
        if abs(x_new - x) < eps:
            return x_new, k, "сошлось", values

        x = x_new

    # не сошлось: проверим, не чередуются ли два значения (цикл периода 2)
    if abs(values[-1] - values[-3]) < eps:
        return x, max_iter, "не сошлось: цикл периода 2", values

    return x, max_iter, "не сошлось", values


def plot_case(mu, nu, values):
    """Одна фигура на тест: слева phi(x) и y = x, справа x_k от k."""
    # границы по x: показываем область, где phi(x) >= 0, и все точки итераций
    xmax = min(2.0, max(1.3 / nu ** 2, 1.2 * max(values)))
    xs = np.linspace(0, xmax, 400)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"mu = {mu}, nu = {nu}")

    ax1.plot(xs, phi(xs, mu, nu), label="phi(x)")
    ax1.plot(xs, xs, "--", label="y = x")
    x_star = fixed_point(mu, nu)
    ax1.plot(x_star, x_star, "ro", label=f"x* = {x_star:.4f}")
    ax1.set_title("График phi(x)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(range(len(values)), values, marker="o", markersize=3)
    ax2.axhline(x_star, color="r", linestyle=":", label="x* (теория)")
    ax2.set_title("Итерации x_k")
    ax2.set_xlabel("k")
    ax2.set_ylabel("x_k")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()


# наборы параметров: (mu, nu, x0)
tests = [
    (1, 1, 0.1),
    (1, 2, 0.1),
    (2, 0.5, 0.1),
    (1, 3.5, 0.01),
    (1, 4, 0.1)
]

for mu, nu, x0 in tests:
    x, iters, status, values = iterate(mu, nu, x0)
    x_star = fixed_point(mu, nu)
    m = multiplier(mu, nu)

    print("-----------------------------")
    print(f"mu = {mu}, nu = {nu}, x0 = {x0}")
    if status == "сошлось":
        print(f"x* (численно) = {x}")
    else:
        print(f"последнее значение = {x}")
    print(f"x* (теория)   = {x_star}")
    if abs(m) < 1:
        verdict = "|phi'| < 1, неподвижная точка устойчива"
    else:
        verdict = "|phi'| > 1, неподвижная точка неустойчива"
    print(f"phi'(x*) = {m:.3f}  ->  {verdict}")
    print(f"итераций = {iters}")
    print(f"статус = {status}")

    plot_case(mu, nu, values)

# все окна с графиками показываем один раз в конце
plt.show()
