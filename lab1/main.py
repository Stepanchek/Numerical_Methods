import numpy as np


CONST_LEFT = -5
CONST_RIGHT = 0
CONST_EPS = 1e-3
CONST_POINTS = 1000

def calc_func(x):
    return x**2 - 9  #x^2-9


def derivative_function(x):
    return x*2      #2x


def dichotomy():
        step = 0
        is_last_operation = False

        left = CONST_LEFT
        right = CONST_RIGHT

        if calc_func(left)*calc_func(right) > 0:
            print("Dichotomy: F(a)*F(b) > 0, no root")
            return

        while not is_last_operation:
            step += 1
            mid = (left+right) / 2
            if calc_func(mid)*calc_func(right) < 0:
                left = mid
            else:
                right = mid

            delta = abs(left-right)

            if delta < CONST_EPS:
                is_last_operation = True
        print(f' x: {(right+left)/2}')


def relaxation():
        points = np.linspace(CONST_LEFT, CONST_RIGHT, CONST_POINTS)
        dx = 1e-5
        min_derivative = np.min(np.abs(calc_func(points + dx) - calc_func(points)) / dx)
        max_derivative = np.max(np.abs(calc_func(points + dx) - calc_func(points)) / dx)
        t = 2 / (max_derivative + min_derivative)
        step = 0
        x = (CONST_LEFT + CONST_RIGHT) / 2
        x_next = x
        is_last_operation = False
        while not is_last_operation:
            step += 1
            f_val = calc_func(x)
            if (calc_func(x + dx) - calc_func(x)) / dx > 0:
                x_next = x - t * f_val
            else:
                x_next = x + t * f_val
            delta = abs(x - x_next)
            x = x_next

            if delta < CONST_EPS:
                is_last_operation = True
        print(f' x: {x_next}')


def newton():
        step = 0
        x = (CONST_LEFT + CONST_RIGHT) / 2
        x_next = x
        is_last_operation = False
        while not is_last_operation:
            step += 1
            x_next = x - calc_func(x) / derivative_function(x) #rename name of func
            delta = abs(x - x_next)
            x = x_next
            if delta < CONST_EPS:
                is_last_operation = True
        print(f' x: {x_next}')


print("Dichotomy method: ")
dichotomy()


print("Relaxation method")
relaxation()

print("Newton method")
newton()
