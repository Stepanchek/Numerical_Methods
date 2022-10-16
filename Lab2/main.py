import random
import numpy as np
from scipy.linalg import hilbert

EPS = 1e-3
MIN_VAL = -50
MAX_VAL = 50


def log_solution(matrix, b, solution):
    print("Matrix: ")
    print(matrix)
    print("B vector: ")
    print(b)
    print("Solution: ")
    print(solution)
    print("Matrix * solution: ")
    print(np.matmul(matrix, solution))
    print(f"Matriix * solution - b : {get_vector_norm(np.matmul(matrix, solution) - b)}")


def gen_matrix(dimension, is_hilbert=False):
    if is_hilbert:
        return np.array(hilbert(dimension))

    result = np.empty([dimension, dimension])

    for i in range(dimension):
        for j in range(dimension):
            result[i, j] = random.randint(MIN_VAL, MAX_VAL)
        result[i, i] = np.sum(np.abs(result[i])) + 1

    return result


def gen_vector(dimension):
    result = np.zeros(dimension)

    for i in range(dimension):
        result[i] = random.randint(MIN_VAL, MAX_VAL)

    return result


# Gauss method
# create E matrix(n*n)
# {1, 0}
# {0, 1}
def get_e_matrix(dimension, k, l):
    result = np.identity(dimension)
    result[[k, l]] = result[[l, k]]
    return result


def get_m_matrix(matrix, k):
    dimension = matrix.shape[0]
    result = np.identity(dimension)
    result[k, k] = 1 / matrix[k, k]
    for i in range(k + 1, dimension):
        result[i, k] = -matrix[i, k] / matrix[k, k]

    return result


def get_lead_elem(matrix, column, start):
    dimension = matrix.shape[0]
    result = start

    for i in range(start + 1, dimension):
        if abs(matrix[i, column]) > abs(matrix[result, column]):
            result = i

    return result


def gauss_method(matrix, b):
    dimension = matrix.shape[0]
    cur_matrix = np.copy(matrix)
    cur_b = np.copy(b)
    result = np.zeros(dimension)

    for i in range(dimension):
        leading_idx = get_lead_elem(cur_matrix, i, i)
        p = get_e_matrix(dimension, i, leading_idx)
        cur_matrix = np.matmul(p, cur_matrix)
        m = get_m_matrix(cur_matrix, i)
        cur_matrix = np.matmul(m, cur_matrix)
        cur_b =  np.matmul(m , np.matmul( p, cur_b))

    print(cur_matrix)
    print(cur_b)

    for i in range(dimension - 1, -1, -1):
        result[i] = cur_b[i]
        for j in range(0, i):
            cur_b[j] -= cur_matrix[j, i] * result[i]

    print(result)
    print(cur_b)
    return result


# Jacobi
def get_vector_norm(x):
    return np.max(np.abs(x))


def jacobi_method(matrix, b, eps=EPS):
    dimension = b.shape[0]
    result = np.ones(dimension)

    while True:
        prev_result = np.array(result)

        for i in range(dimension):
            delta = 0
            for j in range(dimension):
                if i != j:
                    delta += matrix[i, j] * prev_result[j]

            result[i] = (b[i] - delta) / matrix[i, i]

        if get_vector_norm(result - prev_result) <= eps:
            break

    return result


#Seidel
def seidel_method(matrix, b, eps=EPS):
    dimension = b.shape[0]
    result = np.ones(dimension)

    while True:
        prev_result = np.array(result)

        for i in range(dimension):
            delta = 0

            for j in range(dimension):
                if j < i:
                    delta += matrix[i, j] * result[j]
                elif j > i:
                    delta += matrix[i, j] * prev_result[j]

            result[i] = (b[i] - delta) / matrix[i, i]

        if get_vector_norm(result - prev_result) <= eps:
            break

    return result

test_matrix = gen_matrix(3)
test_matrix_hilbert = gen_matrix(3, True)
test_b = gen_vector(3)
print("=============================Gauss=============================")
log_solution(test_matrix, test_b, gauss_method(test_matrix, test_b))
log_solution(test_matrix_hilbert, test_b, gauss_method(test_matrix_hilbert, test_b))
print("=============================Jacobi=============================")
log_solution(test_matrix, test_b, jacobi_method(test_matrix, test_b))
print("=============================Seidel=============================")
log_solution(test_matrix, test_b, seidel_method(test_matrix, test_b))
log_solution(test_matrix_hilbert, test_b, seidel_method(test_matrix_hilbert, test_b))