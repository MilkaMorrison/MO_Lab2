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

def gradient_descent() -> None:
    a, b, c = map(float, input('Введите коэффициенты a, b, и c (через пробел): ').split())
    x = list(map(float, input('Введите начальный вектор x (через пробел): ').split()))
    e1 = float(input('Введите e1: '))
    e2 = float(input('Введите e2: '))
    M = int(input('Введите максимальное число итераций M: '))
    t = float(input('Введите шаг t: '))

    prev_x = x.copy()

    print(f"\nНачальная точка: x = ({x[0]}, {x[1]})")
    print(f"Начальное значение функции: f(x) = {f(a, b, c, x[0], x[1]):.8f}\n")

    for k in range(M):
        print(f"{GREEN}--- Итерация {k} ---{RESET}")

        grad = gradient(a, b, c, x[0], x[1])
        print(f"Градиент: ({grad[0]}, {grad[1]})")

        grad_norm = math.sqrt(grad[0] ** 2 + grad[1] ** 2)
        print(f"Норма градиента: {grad_norm}")

        if grad_norm < e1:
            print(f"{GREEN}\nУсловие остановки по норме градиента выполнено.{RESET}")
            print(f"Найденная точка: ({x[0]:.8f}, {x[1]:.8f})")
            print(f"Значение функции: {f(a, b, c, x[0], x[1]):.8f}")
            k += 1
            print(f"Количество итераций: {k}")
            return

        max_t_reductions = 20
        #while True:
        for _ in range(max_t_reductions):
            new_x = [x[0] - t * grad[0], x[1] - t * grad[1]]
            print(f"  t = {t}, new_x = ({new_x[0]:.8f}, {new_x[1]:.8f}), "
                  f"f(new_x) = {f(a, b, c, new_x[0], new_x[1]):.8f}, f(x) = {f(a, b, c, x[0], x[1]):.8f}")
            if f(a, b, c, new_x[0], new_x[1]) < f(a, b, c, x[0], x[1]):
                break
            t /= 2.0
        else:
            print(f"{GREEN}Шаг t стал слишком маленьким. Остановка.{RESET}")
            return

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
    print(f"Финальная точка: ({x[0]:.8f}, {x[1]:.8f})")
    print(f"Значение функции: {f(a, b, c, x[0], x[1]):.8f}")
    print(f"Количество итераций: {M}")

if __name__ == "__main__":
    gradient_descent()