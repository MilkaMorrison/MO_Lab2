import math
from typing import List

GREEN = "\033[32m"
RESET = "\033[0m"

def f(a: float, b: float, c: float, x1: float, x2: float) -> float:
    return a * x1 ** 2 + b * x1 * x2 + c * x2 ** 2

def gradient(a: float, b: float, c: float, x1: float, x2: float) -> List[float]:
    df_dx1 = 2 * a * x1 + b * x2
    df_dx2 = b * x1 + 2 * c * x2
    return [df_dx1, df_dx2]

def func_t(a: float, b: float, c: float, t: float, x1: float, x2: float) -> float:
    return f(a, b, c, x1 - t * gradient(a, b, c, x1, x2)[0], x2 - t * gradient(a, b, c, x1, x2)[1])

def golden_section_method(a: float, b: float, c: float, x1 : float, x2: float) -> float:
    a_t = -1
    b_t = 1
    epsilon = 0.001
    print(f"epsilon = {epsilon}")
    k = 0

    y = a_t + (3 - math.sqrt(5)) / 2 * (b_t - a_t)
    print(f"y = {y}")
    z = a_t + b_t - y
    print(f"z = {z}")
    L = abs(b_t - a_t)

    while L > epsilon:
        if func_t(a, b, c, y, x1, x2) <= func_t(a, b, c, z, x1, x2):
            print(f"f(y) = {func_t(a, b, c, y, x1, x2)}, f(z) = {func_t(a, b, c, z, x1, x2)}")
            b_t = z
            z = y
            y = a_t + b_t - y
            print(f"a = {a_t}, b = {b_t}, y = {y}, z = {z}")
        else:
            print(f"f(y) = {func_t(a, b, c, y, x1, x2)}, f(z) = {func_t(a, b, c, z, x1, x2)}")
            a_t = y
            y = z
            z = a_t + b_t - z
            print(f"a = {a_t}, b = {b_t}, y = {y}, z = {z}")
        L = abs(b_t - a_t)
        print(f"L = {L}")
        k += 1
    return (a_t + b_t) / 2

def steepest_gradient_descent() -> None:
    a, b, c = map(float, input('Введите коэффициенты a, b, и c (через пробел): ').split())
    x = list(map(float, input('Введите начальный вектор x (через пробел): ').split()))
    e1 = float(input('Введите e1: '))
    e2 = float(input('Введите e2: '))
    M = int(input('Введите максимальное число итераций M: '))

    prev_x = x.copy()

    print(f"Начальная точка: ({x[0]}, {x[1]})")
    print(f"Начальное значение функции: {f(a, b, c, x[0], x[1]):.8f}\n")

    for k in range(M):
        print(f"{GREEN}--- Итерация {k} ---{RESET}")

        grad = gradient(a, b, c, x[0], x[1])
        print(f"Градиент: ({grad[0]}, {grad[1]})")

        grad_norm = math.sqrt(grad[0] ** 2 + grad[1] ** 2)
        print(f"Норма градиента: {grad_norm}")

        if grad_norm < e1:
            print(f"{GREEN}\nУсловие остановки по норме градиента выполнено{RESET}")
            print(f"Найденная точка: ({x[0]:.8f}, {x[1]:.8f})")
            print(f"Значение функции: {f(a, b, c, x[0], x[1]):.8f}")
            k = k + 1
            print(f"Количество итераций: {k}")
            return

        t = golden_section_method(a, b, c, x[0], x[1])
        new_x = [x[0] - t * grad[0], x[1] - t * grad[1]]
        print(f"  t = {t}, new_x = ({new_x[0]:.8f}, {new_x[1]:.8f}), "
                  f"f(new_x) = {f(a, b, c, new_x[0], new_x[1]):.8f}, f(x) = {f(a, b, c, x[0], x[1]):.8f}")

        delta_x = math.sqrt((new_x[0] - x[0]) ** 2 + (new_x[1] - x[1]) ** 2)
        delta_f = abs(f(a, b, c, new_x[0], new_x[1]) - f(a, b, c, x[0], x[1]))

        if k > 0 and delta_x < e2 and delta_f < e2:
            delta_x_prev = math.sqrt((x[0] - prev_x[0]) ** 2 + (x[1] - prev_x[1]) ** 2)
            delta_f_prev = abs(f(a, b, c, x[0], x[1]) - f(a, b, c, prev_x[0], prev_x[1]))

            if delta_x_prev < e2 and delta_f_prev < e2:
                print(f"{GREEN}\nУсловие остановки по малым изменениям выполнено{RESET}")
                print(f"Найденная точка: ({new_x[0]:.8f}, {new_x[1]:.8f})")
                print(f"Значение функции: {f(a, b, c, new_x[0], new_x[1]):.8f}")
                k = k + 1
                print(f"Количество итераций: {k}")
                return

        prev_x = x.copy()
        x = new_x.copy()

        print(f"Новая точка: ({x[0]:.8f}, {x[1]:.8f})")
        print(f"Значение функции: {f(a, b, c, x[0], x[1]):.8f}\n")

    print(f"{GREEN}\nДостигнуто максимальное количество итераций{RESET}")
    print(f"Финальная точка: ({x[0]:.6f}, {x[1]:.8f})")
    print(f"Значение функции: {f(a, b, c, x[0], x[1]):.8f}")
    print(f"Количество итераций: {M}")

if __name__ == "__main__":
    steepest_gradient_descent()