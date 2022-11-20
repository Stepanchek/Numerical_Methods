import random
import numpy as np

D = 0.15
EPS = 0.01


def l_norm(x):
    return np.linalg.norm(x)


def make_random_matrix(n):
    start_matrix = np.zeros([n, n])

    for i in range(n):
        for j in range(n):
            start_matrix[i, j] = random.randint(0, 1)

    print("START MATRIX:")
    print(start_matrix)
    return start_matrix


def g_matrix(matrix):
    n = matrix.shape[0]

    # for sink nodes & disconnected components
    count_line = 0
    for curr_line in matrix:
        curr_sum = 0.0
        for curr_item in curr_line:
            curr_sum += curr_item
        if curr_sum == 0.0:
            count_col = 0
            for _ in curr_line:
                matrix[count_line, count_col] = 1
                count_col += 1
        count_line += 1

    aMatr = matrix.T / matrix.sum(axis=1)
    print("A MATRIX:")
    print(aMatr)

    # for disconnected components
    googleMatr = (1-D)*aMatr + D*(np.ones((n, n))/n)
    print("G MATRIX:")
    print(googleMatr)
    return googleMatr


def power_method(matrix, max_iter=10):
    n = matrix.shape[0]
    res = np.ones(n)

    for _ in range(max_iter):
        prev_res = res
        res = np.matmul(matrix, res)

        if l_norm(res - prev_res)/n <= EPS:
            return res

    return res



def pagerank_main(matrix):
    google_matr = g_matrix(matrix)
    res = power_method(google_matr)
    print("RES:")
    print(res)
    return res


matrix = make_random_matrix(5)
res = pagerank_main(matrix)


