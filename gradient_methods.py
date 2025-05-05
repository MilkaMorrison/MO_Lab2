import math
from typing import List, Tuple, Dict

def f(a: float, b: float, c: float, x1: float, x2: float) -> float:
    return a * x1 ** 2 + b * x1 * x2 + c * x2 ** 2

def gradient(a: float, b: float, c: float, x1: float, x2: float) -> List[float]:
    df_dx1 = 2 * a * x1 + b * x2
    df_dx2 = b * x1 + 2 * c * x2
    return [df_dx1, df_dx2]

def func_t(a: float, b: float, c: float, t: float, x1: float, x2: float) -> float:
    return f(a, b, c, x1 - t * gradient(a, b, c, x1, x2)[0], x2 - t * gradient(a, b, c, x1, x2)[1])

def golden_section_method(a: float, b: float, c: float, x1: float, x2: float) -> float:
    a_t = -1
    b_t = 1
    epsilon = 0.001
    k = 0

    y = a_t + (3 - math.sqrt(5)) / 2 * (b_t - a_t)
    z = a_t + b_t - y
    L = abs(b_t - a_t)

    while L > epsilon:
        if func_t(a, b, c, y, x1, x2) <= func_t(a, b, c, z, x1, x2):
            b_t = z
            z = y
            y = a_t + b_t - y
        else:
            a_t = y
            y = z
            z = a_t + b_t - z
        L = abs(b_t - a_t)
        k += 1
    return (a_t + b_t) / 2

def gradient_descent(a: float, b: float, c: float, x: List[float],
                     e1: float, e2: float, M: int, t: float) -> Tuple[Dict, List[List[float]]]:
    points = [x.copy()]
    prev_x = x.copy()

    for k in range(M):
        grad = gradient(a, b, c, x[0], x[1])
        grad_norm = math.sqrt(grad[0] ** 2 + grad[1] ** 2)

        if grad_norm < e1:
            return {
                'point': x,
                'value': f(a, b, c, x[0], x[1]),
                'iterations': k + 1
            }, points

        max_t_reductions = 20
        for _ in range(max_t_reductions):
            new_x = [x[0] - t * grad[0], x[1] - t * grad[1]]
            if f(a, b, c, new_x[0], new_x[1]) < f(a, b, c, x[0], x[1]):
                break
            t /= 2.0
        else:
            return {
                'point': x,
                'value': f(a, b, c, x[0], x[1]),
                'iterations': k + 1,
                'message': 'Шаг t стал слишком маленьким'
            }, points

        delta_x = math.sqrt((new_x[0] - x[0]) ** 2 + (new_x[1] - x[1]) ** 2)
        delta_f = abs(f(a, b, c, new_x[0], new_x[1]) - f(a, b, c, x[0], x[1]))

        if k > 0 and delta_x < e2 and delta_f < e2:
            delta_x_prev = math.sqrt((x[0] - prev_x[0]) ** 2 + (x[1] - prev_x[1]) ** 2)
            delta_f_prev = abs(f(a, b, c, x[0], x[1]) - f(a, b, c, prev_x[0], prev_x[1]))

            if delta_x_prev < e2 and delta_f_prev < e2:
                points.append(new_x.copy())
                return {
                    'point': new_x,
                    'value': f(a, b, c, new_x[0], new_x[1]),
                    'iterations': k + 1
                }, points

        prev_x = x.copy()
        x = new_x.copy()
        points.append(x.copy())

    return {
        'point': x,
        'value': f(a, b, c, x[0], x[1]),
        'iterations': M,
        'message': 'Достигнуто максимальное количество итераций'
    }, points

def steepest_gradient_descent(a: float, b: float, c: float, x: List[float],
                              e1: float, e2: float, M: int) -> Tuple[Dict, List[List[float]]]:
    points = [x.copy()]
    prev_x = x.copy()

    for k in range(M):
        grad = gradient(a, b, c, x[0], x[1])
        grad_norm = math.sqrt(grad[0] ** 2 + grad[1] ** 2)

        if grad_norm < e1:
            return {
                'point': x,
                'value': f(a, b, c, x[0], x[1]),
                'iterations': k + 1
            }, points

        t = golden_section_method(a, b, c, x[0], x[1])
        new_x = [x[0] - t * grad[0], x[1] - t * grad[1]]

        delta_x = math.sqrt((new_x[0] - x[0]) ** 2 + (new_x[1] - x[1]) ** 2)
        delta_f = abs(f(a, b, c, new_x[0], new_x[1]) - f(a, b, c, x[0], x[1]))

        if k > 0 and delta_x < e2 and delta_f < e2:
            delta_x_prev = math.sqrt((x[0] - prev_x[0]) ** 2 + (x[1] - prev_x[1]) ** 2)
            delta_f_prev = abs(f(a, b, c, x[0], x[1]) - f(a, b, c, prev_x[0], prev_x[1]))

            if delta_x_prev < e2 and delta_f_prev < e2:
                points.append(new_x.copy())
                return {
                    'point': new_x,
                    'value': f(a, b, c, new_x[0], new_x[1]),
                    'iterations': k + 1
                }, points

        prev_x = x.copy()
        x = new_x.copy()
        points.append(x.copy())

    return {
        'point': x,
        'value': f(a, b, c, x[0], x[1]),
        'iterations': M,
        'message': 'Достигнуто максимальное количество итераций'
    }, points