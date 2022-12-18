import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

form = [
    [1, 1, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [0, 1, 1, 1, 1],
    [0, 0, 1, 1, 0]
]
form = [
    [0, 0, 1, 1, 0],
    [0, 1, 1, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 0, 0, 1]
]
form_size1 = 4
form_size2 = 5

# form = np.ones((form_size1, form_size2))
# form = [
#     [1, 1, 1],
#     [0, 0, 0]
# ]
# form_size1 = 2
# form_size2 = 3

#form = np.ones((form_size1, form_size2))


def u_1(y, x, a=2, b=5, c=7):
    return 2 * x + 5 * y + 7


def u_2(y, x, a=2, b=5, c=7):
    return a * x ** 2 + b * y + c


def u_3(y, x, a=2, b=5, c=7):
    return a * x ** 2 + b * y ** 2 + c


def u_4(y, x):
    return np.sin(x) - np.cos(y) #+ 20


def fi_1(y, x):
    return 0


def fi_2(y, x):
    return -4


def fi_3(y, x):
    return -14


def fi_4(y, x):
    return np.sin(x) - np.cos(y)


def g_1(z, mode):
    global y0, y1, x0, x1
    if mode == 0:
        return u_1(z, x0)
    if mode == 1:
        return u_1(y0, z)
    if mode == 2:
        return u_1(z, x1)
    if mode == 3:
        return u_1(y1, z)


def g_2(z, mode):
    global y0, y1, x0, x1
    if mode == 0:
        return u_2(z, x0)
    if mode == 1:
        return u_2(y0, z)
    if mode == 2:
        return u_2(z, x1)
    if mode == 3:
        return u_2(y1, z)


def g_3(z, mode):
    global y0, y1, x0, x1
    if mode == 0:
        return u_3(z, x0)
    if mode == 1:
        return u_3(y0, z)
    if mode == 2:
        return u_3(z, x1)
    if mode == 3:
        return u_3(y1, z)


def g_4(z, mode):
    global y0, y1, x0, x1
    if mode == 0:
        return u_4(z, x0)
    if mode == 1:
        return u_4(y0, z)
    if mode == 2:
        return u_4(z, x1)
    if mode == 3:
        return u_4(y1, z)


def scalar(obj1, obj2):
    sum = 0
    if len(obj1.shape) == 2:
        N1 = obj1.shape[0] - 1
        N2 = obj1.shape[1] - 1
        for i in range(1, N1):
            for j in range(1, N2):
                sum += obj1[i, j] * obj2[i, j]
    else:
        N = obj1.shape[0] - 1
        for i in range(1, N):
            sum += obj1[i] * obj2[i]
    return sum


def multiply_matr_as_vec2(oper2, N1, N2, hy, hx):
    answer = np.zeros(oper2.shape)
    for i in range(1, N1 - 1):
        for j in range(1, N2 - 1):
            answer[i, j] = right_part2(oper2, hy, hx, i, j)
    return answer


def _bicg_stab(x0, x1, y0, y1, hx, hy, eps, f, g, u_fun):
    Ny = int((y1 - y0) / hy) + 1
    Nx = int((x1 - x0) / hx) + 1
    u = np.zeros((Ny, Nx))
    b = np.zeros((Ny, Nx))
    for i in range(Ny):
        u[i, 0] = g(y0 + i * hy, 0)
        u[i, Nx - 1] = g(y0 + i * hy, 2)
    for i in range(Nx):
        u[0, i] = g(x0 + i * hx, 1)
        u[Ny - 1, i] = g(x0 + i * hx, 3)
    for i in range(1, Ny - 1):  # задаём узлы
        for j in range(1, Nx - 1):
            u[i][j] = u[i][0]
            b[i][j] = f(y0 + i * hy, x0 + j * hx)
    x = u.copy()
    # Сформировали начальное приближение решения - матрицу u, c которой будем работать как с вектором, т.е. операция матричного умножения A на u будет действовать на матрциу u как на вектор
    # Сформировали вектор правой части - f(y,x)
    draw_graph(u, hy, hx)
    Au = multiply_matr_as_vec2(x, Ny, Nx, hy, hx)
    r = b - Au
    r_ = r.copy()
    r_[0][0] += np.sign(r_[0][0]) * 10 + 1
    p = r
    NN = int((Nx) * (Ny) / 1)
    break_iter = NN

    for j in range(NN):
        Ap = multiply_matr_as_vec2(p, Ny, Nx, hy, hx)
        al = scalar(r, r_) / scalar(Ap, r_)
        s = r - al * Ap
        As = multiply_matr_as_vec2(s, Ny, Nx, hy, hx)
        w = scalar(As, s) / scalar(As, As)
        x = x + al * p + w * s
        r_prev = r.copy()
        r = s - w * As
        # if abs(w) < eps:
        # if np.linalg.norm(r)/np.linalg.norm(b) < eps:
        # r_delta = r_prev - r
        if scalar(r, r) ** 0.5 < eps:
            break_iter = j
            break
        bt = scalar(r, r_) * al / (scalar(r_prev, r_) * w)
        p = r + bt * (p - w * Ap)
    mistake_second_norm = 0
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            u[i, j] = x[i, j]
            mistake_second_norm += (u[i, j] - u_fun(hy * i, hx * j)) ** 2
    return u, mistake_second_norm ** 0.5, scalar(r, r) ** 0.5, break_iter


def my_print(m):
    len1 = len(m)
    len2 = len(m[0])
    for i in range(len1):
        for j in range(len2):
            print('%2.3e' % m[i, j], end=' ')
        print()
        print()
    print()
    print()


def fill_G(f, i, j, hy, hx, N_, _N, matr):
    global x0, y0, x1, y1
    if i == -1:
        for i in range(N_, _N + 1):
            matr[i][j] = f(y0 + i * hy, x0 + j * hx)
    else:
        for j in range(N_, _N + 1):
            matr[i][j] = f(y0 + i * hy, x0 + j * hx)

def right_part2(oper2, hy, hx, i, j):
    opi1 = oper2[i - 1, j]
    opi2 = oper2[i + 1, j]
    opi4 = oper2[i, j - 1]
    opi5 = oper2[i, j + 1]
    return -((opi1 - 2 * oper2[i, j] + opi2) / hy ** 2 + (opi4 - 2 * oper2[i, j] + opi5) / hx ** 2)

def scalar_spec(obj1, obj2):
    global form_size1, form_size2, form
    sum = 0
    N1 = int((obj1.shape[0] - 1)/form_size1)
    N2 = int((obj1.shape[1] - 1)/form_size2)
    for ii in range(form_size1):
        for jj in range(form_size2):
            if form[ii][jj] == 0:
                continue
            for i in range(ii * N1 + 1, ii * N1 + N1):
                for j in range(jj * N2 + 1, jj * N2 + N2):
                    sum += obj1[i, j] * obj2[i, j]
            if (ii < form_size1 - 1) and (form[ii + 1][jj] == 1):
                # если текущая подобласть не является последней по вертикали, но является последней по горизонтали и следующая подобласть по вертикали является частью формы
                for j in range(jj * N2 + 1, jj * N2 + N2):
                    i = ii * N1 + N1
                    sum += obj1[i, j] * obj2[i, j]

            if (jj < form_size2 - 1) and (form[ii][jj + 1] == 1):
                # если текущая подобласть не является последней по горизонтали, но является последней по вертикали и следующая подобласть по горизонтали является частью формы
                for i in range(ii * N1 + 1, ii * N1 + N1):
                    j = jj * N2 + N2
                    sum += obj1[i, j] * obj2[i, j]

            if (jj < form_size2 - 1) and (form[ii][jj + 1] == 1) and (ii < form_size1 - 1) and (form[ii + 1][jj] == 1):
                sum += obj1[(ii + 1) * N1, (jj + 1) * N2] * obj2[(ii + 1) * N1, (jj + 1) * N2]
    return sum



def multiply_matr_as_vec2_spec(oper2, N1, N2, hy, hx):
    global form_size1, form_size2, form
    answer = np.zeros(oper2.shape)
    for ii in range(form_size1):
        for jj in range(form_size2):
            if form[ii][jj] == 0:
                continue
            for i in range(ii * N1 + 1, ii * N1 + N1):
                for j in range(jj * N2 + 1, jj * N2 + N2):
                    answer[i, j] = right_part2(oper2, hy, hx, i, j)
            if (ii < form_size1 - 1) and (form[ii + 1][jj] == 1):
                # если текущая подобласть не является последней по вертикали, но является последней по горизонтали и следующая подобласть по вертикали является частью формы
                for j in range(jj * N2 + 1, jj * N2 + N2):
                    i = ii * N1 + N1
                    answer[i, j] = right_part2(oper2, hy, hx, i, j)

            if (jj < form_size2 - 1) and (form[ii][jj + 1] == 1):
                # если текущая подобласть не является последней по горизонтали, но является последней по вертикали и следующая подобласть по горизонтали является частью формы
                for i in range(ii * N1 + 1, ii * N1 + N1):
                    j = jj * N2 + N2
                    answer[i, j] = right_part2(oper2, hy, hx, i, j)

            if (jj < form_size2 - 1) and (form[ii][jj + 1] == 1) and (ii < form_size1 - 1) and (form[ii + 1][jj] == 1):
                answer[(ii + 1) * N1, (jj + 1) * N2] = right_part2(oper2, hy, hx, (ii + 1) * N1, (jj + 1) * N2)
    return answer


def _bicg_stab_spec(x0, x1, y0, y1, hx, hy, eps, f, g, u_fun):
    global form, form_size1, form_size2
    Ny = int((y1 - y0) / hy)  # Размеры одной подобласти
    Nx = int((x1 - x0) / hx)
    Nfull_y = Ny * form_size1 + 1
    Nfull_x = Nx * form_size2 + 1
    u_matrix = np.zeros((Nfull_y, Nfull_x))
    b = form_b(f)

    # Заполнение граничных точек по всей карте области
    for ii in range(form_size1):
        for jj in range(form_size2):
            if form[ii][jj] == 0:
                continue
            if ii == 0:  # Верх области
                fill_G(u_fun, ii * Ny, -1, hy, hx, jj * Nx, (jj + 1) * Nx, u_matrix)
            if ii == form_size1 - 1:  # Низ области
                fill_G(u_fun, (ii + 1) * Ny, -1, hy, hx, jj * Nx, (jj + 1) * Nx, u_matrix)
            if jj == 0:     # Левая граница области
                fill_G(u_fun, -1, jj * Nx, hy, hx, ii * Ny, (ii + 1) * Ny, u_matrix)
            if jj == form_size2 - 1:    # Правая граница области
                fill_G(u_fun, -1, (jj + 1) * Nx, hy, hx, ii * Ny, (ii + 1) * Ny, u_matrix)

            if (ii + 1 < form_size1) and (form[ii + 1][jj] == 0):  # Если следующая клетка области не является частью области задания, заполняем границу с ней
                fill_G(u_fun, (ii + 1) * Ny, -1, hy, hx, jj * Nx, (jj + 1) * Nx, u_matrix)
            if (ii > 0) and (form[ii - 1][jj] == 0):  # Аналогично с предыдущей областью
                fill_G(u_fun, ii * Ny, -1, hy, hx, jj * Nx, (jj + 1) * Nx, u_matrix)

            if (jj + 1 < form_size2) and (form[ii][jj + 1] == 0):   # То же самое по горизонтали
                fill_G(u_fun, -1, (jj + 1) * Nx, hy, hx, ii * Ny, (ii + 1) * Ny, u_matrix)
            if (jj > 0) and (form[ii][jj - 1] == 0):
                fill_G(u_fun, -1, jj * Nx, hy, hx, ii * Ny, (ii + 1) * Ny, u_matrix)

    for ii in range(form_size1):  # задаём узлы
        for jj in range(form_size2):
            if form[ii][jj] == 0:
                continue
            for i in range(ii * Ny + 1, ii * Ny + Ny):
                for j in range(jj * Nx + 1, jj * Nx + Nx):
                    u_matrix[i][j] = u_matrix[i][jj * Nx]
            if (jj + 1 < form_size2) and (form[ii][jj + 1] == 1):
                for i in range(ii * Ny + 1, (ii + 1) * Ny):
                    u_matrix[i][(jj + 1) * Nx] = u_matrix[ii * Ny][(jj + 1) * Nx]
            if (ii + 1 < form_size1) and (form[ii + 1][jj] == 1):
                for j in range(jj * Nx + 1, (jj + 1) * Nx):
                    u_matrix[(ii + 1) * Ny][j] = u_matrix[(ii + 1) * Ny][jj * Nx]
            if (ii + 1 < form_size1) and (form[ii + 1][jj] == 1) and (jj + 1 < form_size2) and (form[ii][jj + 1] == 1):
                u_matrix[(ii + 1) * Ny, (jj + 1) * Nx] = u_matrix[(ii + 1) * Ny][jj * Nx]

    #draw_graph(u_matrix, hy, hx, mode=-1)
    # Задали начальное приближение
    x = u_matrix.copy()
    Au = multiply_matr_as_vec2_spec(u_matrix, Ny, Nx, hy, hx)
    r = b - Au
    r_ = r.copy()
    p = r
    NN = Nx * Ny
    #NN = Nfull_y * Nfull_x
    break_iter = NN

    for j in range(NN):
        Ap = multiply_matr_as_vec2_spec(p, Ny, Nx, hy, hx)
        al = scalar_spec(r, r_) / scalar_spec(Ap, r_)
        s = r - al * Ap
        # print('al')
        # print(al)
        # print('Ap')
        # my_print(Ap)
        # print('\n\n\n')
        As = multiply_matr_as_vec2_spec(s, Ny, Nx, hy, hx)
        w = scalar_spec(As, s) / scalar_spec(As, As)
        x = x + al * p + w * s
        r_prev = r.copy()
        r = s - w * As
        # if abs(w) < eps:
        # if np.linalg.norm(r)/np.linalg.norm(b) < eps:
        # r_delta = r_prev - r
        if scalar_spec(r, r) ** 0.5 < eps:
            break_iter = j
            break
        bt = scalar_spec(r, r_) * al / (scalar_spec(r_prev, r_) * w)
        p = r + bt * (p - w * Ap)
    for ii in range(form_size1):  # задаём узлы
        for jj in range(form_size2):
            if form[ii][jj] == 0:
                continue
            for i in range(ii * Ny + 1, ii * Ny + Ny):
                for j in range(jj * Nx + 1, jj * Nx + Nx):
                    u_matrix[i][j] = x[i, j]
            if (jj + 1 < form_size2) and (form[ii][jj + 1] == 1):
                for i in range(ii * Ny + 1, (ii + 1) * Ny):
                    u_matrix[i][(jj + 1) * Nx] = x[i, (jj + 1) * Nx]
            if (ii + 1 < form_size1) and (form[ii + 1][jj] == 1):
                for j in range(jj * Nx + 1, (jj + 1) * Nx):
                    u_matrix[(ii + 1) * Ny][j] = x[(ii + 1) * Ny, j]
            if (ii + 1 < form_size1) and (form[ii + 1][jj] == 1) and (jj + 1 < form_size2) and (form[ii][jj + 1] == 1):
                    u_matrix[(ii + 1) * Ny][(jj + 1) * Nx] = x[(ii + 1) * Ny][(jj + 1) * Nx]

    return u_matrix, break_iter


def jac_specified(f, u):
    global x0, x1, y0, y1, hy, hx, eps, form, form_size1, form_size2  # x0, x1, y0, y1 - размер одной подобласти
    Ny = N1 = int((y1 - y0) / hy)  # Размеры одной подобласти
    Nx = N2 = int((x1 - x0) / hx)
    u_matrix = np.zeros((form_size1 * N1 + 1, form_size2 * N2 + 1))

    masses_u = []
    it = 0
    # Заполнение граничных точек по всей карте области
    for ii in range(form_size1):
        for jj in range(form_size2):
            if form[ii][jj] == 0:
                continue
            masses_u[0] = np.zeros((Ny + 1, Nx + 1))
            if ii == 0 or ii == form_size1 - 1:  # Верх или низ области
                fill_G(u, (ii + int(ii == form_size1 - 1)) * N1, -1, hy, hx, jj * N2, (jj + 1) * N2, u_matrix)
            if jj == 0 or jj == form_size2 - 1:
                fill_G(u, -1, (jj + int(jj == form_size2 - 1)) * N2, hy, hx, ii * N1, (ii + 1) * N1, u_matrix)

            if (ii + 1 < form_size1) and (form[ii + 1][
                                              jj] == 0):  # Если следующая клетка области не является частью области задания, заполняем границу с ней
                fill_G(u, (ii + 1) * N1, -1, hy, hx, jj * N2, (jj + 1) * N2, u_matrix)
            if (ii > 0) and (form[ii - 1][jj] == 0):  # Аналогично с предыдущей областью
                fill_G(u, ii * N1, -1, hy, hx, jj * N2, (jj + 1) * N2, u_matrix)

            if (jj + 1 < form_size2) and (form[ii][jj + 1] == 0):
                fill_G(u, -1, (jj + 1) * N2, hy, hx, ii * N1, (ii + 1) * N1, u_matrix)
            if (jj > 0) and (form[ii][jj - 1] == 0):
                fill_G(u, -1, jj * N2, hy, hx, ii * N1, (ii + 1) * N1, u_matrix)

    #draw_graph(u_matrix, hy, hx, -1)
    for ii in range(form_size1):  # задаём узлы
        for jj in range(form_size2):
            if form[ii][jj] == 0:
                continue
            for i in range(ii * Ny + 1, ii * Ny + Ny):
                for j in range(jj * Nx + 1, jj * Nx + Nx):
                    u_matrix[i][j] = u_matrix[ii * Ny][j]
            if (jj + 1 < form_size2) and (form[ii][jj + 1] == 1):
                for i in range(ii * Ny + 1, (ii + 1) * Ny):
                    u_matrix[i][(jj + 1) * Nx] = u_matrix[ii * Ny][(jj + 1) * Nx]
            if (ii + 1 < form_size1) and (form[ii + 1][jj] == 1):
                for j in range(jj * Nx + 1, (jj + 1) * Nx):
                    u_matrix[(ii + 1) * Ny][j] = u_matrix[(ii + 1) * Ny][jj * Nx]
            if (ii + 1 < form_size1) and (form[ii + 1][jj] == 1) and (jj + 1 < form_size2) and (form[ii][jj + 1] == 1):
                u_matrix[(ii + 1) * Ny, (jj + 1) * Nx] = u_matrix[(ii + 1) * Ny][jj * Nx]

    #draw_graph(u_matrix, hy, hx, -1)
    # Завершили построение области
    flag = False
    q = 0
    while not flag:
        q += 1
        old_u_matrix = u_matrix.copy()
        max_delta = 0
        sum_delta = 0
        for ii in range(form_size1):
            for jj in range(form_size2):
                if form[ii][jj] == 0:
                    continue
                for i in range(ii * N1 + 1, ii * N1 + N1):
                    for j in range(jj * N2 + 1, jj * N2 + N2):
                        u_matrix[i][j] = (1 / 4) * (
                                old_u_matrix[i + 1][j] + old_u_matrix[i - 1][j] + old_u_matrix[i][j - 1] +
                                old_u_matrix[i][
                                    j + 1]) + (h ** 2 / 4) * f(i * h, j * h)
                        sum_delta += (u_matrix[i][j] - old_u_matrix[i][j]) ** 2
                        if abs(u_matrix[i][j] - old_u_matrix[i][j]) > max_delta:
                            max_delta = abs(u_matrix[i][j] - old_u_matrix[i][j])

                if (ii < form_size1 - 1) and (form[ii + 1][jj] == 1):
# если текущая подобласть не является последней по вертикали и следующая подобласть по вертикали является частью формы
                    for j in range(jj * N2 + 1, jj * N2 + N2):
                        i = ii * N1 + N1
                        u_matrix[i][j] = (1 / 4) * (
                                old_u_matrix[i + 1][j] + old_u_matrix[i - 1][j] + old_u_matrix[i][j - 1] +
                                old_u_matrix[i][
                                    j + 1]) + (h ** 2 / 4) * f(i * h, j * h)
                        sum_delta += (u_matrix[i][j] - old_u_matrix[i][j]) ** 2
                        if abs(u_matrix[i][j] - old_u_matrix[i][j]) > max_delta:
                            max_delta = abs(u_matrix[i][j] - old_u_matrix[i][j])

                if (jj < form_size2 - 1) and (form[ii][jj + 1] == 1):
# если текущая подобласть не является последней по горизонтали и следующая подобласть по горизонтали является частью формы
                    for i in range(ii * N1 + 1, ii * N1 + N1):
                        j = jj * N2 + N2
                        u_matrix[i][j] = (1 / 4) * (
                                old_u_matrix[i + 1][j] + old_u_matrix[i - 1][j] + old_u_matrix[i][j - 1] +
                                old_u_matrix[i][
                                    j + 1]) + (h ** 2 / 4) * f(i * h, j * h)
                        sum_delta += (u_matrix[i][j] - old_u_matrix[i][j]) ** 2
                        if abs(u_matrix[i][j] - old_u_matrix[i][j]) > max_delta:
                            max_delta = abs(u_matrix[i][j] - old_u_matrix[i][j])

                if (jj < form_size2 - 1) and (form[ii][jj + 1] == 1) and (ii < form_size1 - 1) and (form[ii + 1][jj] == 1):
                    i = ii * N1 + N1
                    j = jj * N2 + N2
                    u_matrix[i][j] = (1 / 4) * (
                            old_u_matrix[i + 1][j] + old_u_matrix[i - 1][j] + old_u_matrix[i][j - 1] +
                            old_u_matrix[i][
                                j + 1]) + (h ** 2 / 4) * f(i * h, j * h)
                    sum_delta += (u_matrix[i][j] - old_u_matrix[i][j]) ** 2
                    if abs(u_matrix[i][j] - old_u_matrix[i][j]) > max_delta:
                        max_delta = abs(u_matrix[i][j] - old_u_matrix[i][j])

        if sum_delta ** 0.5 < eps: flag = True
    return u_matrix, q


def jac(f, g, u_fun):
    global x0, x1, y0, y1, hy, hx, eps  # x0, x1, y0, y1 - размер одной подобласти
    N1 = int((y1 - y0) / hy) + 1
    N2 = int((x1 - x0) / hx) + 1
    u_matrix = np.zeros((N1, N2))
    for i in range(N1):  # задаём границу
        u_matrix[i][0] = g(y0 + i * hy, 0)
        u_matrix[i][-1] = g(y0 + i * hy, 2)
    for i in range(N2):  # задаём границу
        u_matrix[0][i] = g(x0 + i * hx, 1)
        u_matrix[-1][i] = g(x0 + i * hx, 3)
    for i in range(1, N1 - 1):  # задаём узлы
        for j in range(1, N2 - 1):
            u_matrix[i][j] = u_matrix[i][0]
    flag = False
    q = 0
    while not flag:
        q += 1
        old_u_matrix = u_matrix.copy()
        max_delta = 0
        for i in range(1, N1 - 1):
            for j in range(1, N2 - 1):
                u_matrix[i][j] = (1 / 4) * (
                        old_u_matrix[i + 1][j] + old_u_matrix[i - 1][j] + old_u_matrix[i][j - 1] + old_u_matrix[i][
                    j + 1]) + (h ** 2 / 4) * f(i * h, j * h)
                if abs(u_matrix[i][j] - old_u_matrix[i][j]) > max_delta:
                    max_delta = abs(u_matrix[i][j] - old_u_matrix[i][j])
        if max_delta < eps: flag = True
    mistake_second_norm = 0
    for i in range(1, N1 - 1):
        for j in range(1, N2 - 1):
            mistake_second_norm += (u_matrix[i, j] - u_fun(hy * i, hx * j)) ** 2
    return u_matrix, q, mistake_second_norm ** 0.5


def solve_clear(u):
    global x0, x1, y0, y1, hx, hy
    N1 = int((y1 - y0) / hy + 1)
    N2 = int((x1 - x0) / hx + 1)
    clear = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            clear[i][j] = u(i * hy, j * hx)
    return clear


def solve_clear_spec(fun):
    global x0, x1, y0, y1, hy, hx, form_size1, form_size2, form
    N1 = int((y1 - y0) / hy)
    N2 = int((x1 - x0) / hx)
    clear = np.zeros((form_size1 * N1 + 1, form_size2 * N2 + 1))
    for ii in range(form_size1):
        for jj in range(form_size2):
            if form[ii][jj] == 0:
                continue
            for i in range(ii * N1, ii * N1 + N1 + 1):
                for j in range(jj * N2, jj * N2 + N2 + 1):
                    clear[i, j] = fun(y0 + i * hy, x0 + j * hx)
    return clear

def form_b(fun):
    global x0, x1, y0, y1, hy, hx, form_size1, form_size2, form
    N1 = int((y1 - y0) / hy)
    N2 = int((x1 - x0) / hx)
    b = np.zeros((form_size1 * N1 + 1, form_size2 * N2 + 1))
    for ii in range(form_size1):
        for jj in range(form_size2):
            if form[ii][jj] == 0:
                continue
            for i in range(ii * N1 + 1, ii * N1 + N1):
                for j in range(jj * N2 + 1, jj * N2 + N2):
                    b[i, j] = fun(y0 + i * hy, x0 + j * hx)
    return b

def draw_graph(digital, hy, hx, mode=0):
    global x0, x1, y0, y1, form_size1, form_size2  # x0, x1, y0, y1 - размер одной подобласти
    if mode == -1:
        Ny_big = form_size1 * int((y1 - y0) / hy) + 1
        Nx_big = form_size2 * int((x1 - x0) / hx) + 1
        Ny = int((y1 - y0) / hy)
        Nx = int((y1 - y0) / hx)
        Z = np.zeros((Ny_big, Nx_big))
        it = 0
        for ii in range(form_size1):
            for jj in range(form_size2):
                if form[ii][jj] == 0:
                    continue
                for i in range(Ny):
                    for j in range(Nx):
                        Z[ii * Ny + i, jj * Nx + j] = digital[it][i, j]
                it += 1
        yY, xX = [y0 + i * hy for i in range(Ny)], [x0 + j * hx for j in range(Nx)]
    else:
        Ny = int((y1 - y0) / hy) + 1
        Nx = int((x1 - x0) / hx) + 1
        yY, xX = [y0 + i * hy for i in range(Ny)], [x0 + j * hx for j in range(Nx)]
        Z = digital.copy()
    xgrid, ygrid = np.meshgrid(xX, yY)
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    axes.plot_surface(xgrid, ygrid, Z, cmap=cm.coolwarm)
    plt.show()


x0 = 0
x1 = 1
y0 = 0
y1 = 1
eps = 0.001
h = 0.1
hy = h
hx = h
mas_funs = [[u_1, fi_1, g_1], [u_2, fi_2, g_2], [u_3, fi_3, g_3], [u_4, fi_4, g_4]]
mas_ends = [[0, 1, 0, 1], [0, 1, 0, 2], [1, 3, 2, 6]]
mas_eps = [0.001, 0.000001]
mas_hy = [0.1, 0.05, 0.01]
mas_hx = [0.1, 0.05, 0.01]
[u, f, g] = mas_funs[3]


# print(clear[7])
# print(digital2[7])
# print('Norm', np.linalg.norm(digital2 - clear, ord='fro'))


#Решение задачи для специфической области стабилизированным методом бисопряжённых градиентов
clear = solve_clear_spec(u)
digital2 = _bicg_stab_spec(x0, x1, y0, y1, hx, hy, eps, f, g, u)
#my_print(digital2[0])
print(scalar_spec(digital2[0] - clear, digital2[0] - clear) ** 0.5)
draw_graph(digital2[0] - clear, hy, hx, -1)
draw_graph(digital2[0], hy, hx, -1)
draw_graph(clear, hy, hx, -1)

exit()

#Решение задачи для специфической области методом Якоби
clear = solve_clear_spec(u)
digital2 = jac_specified(f, u)
my_print(digital2[0])
print(scalar_spec(digital2[0] - clear, digital2[0] - clear) ** 0.5)
draw_graph(digital2[0] - clear, hy, hx, -1)
draw_graph(digital2[0], hy, hx, -1)
draw_graph(clear, hy, hx, -1)

exit()


clear = solve_clear(u)
digital1 = jac(f, g, u)
digital1_alt = _bicg_stab(x0, x1, y0, y1, hx, hy, eps, f, g, u)
draw_graph(clear, hy, hx)
draw_graph(digital1[0], hy, hx)
draw_graph(digital1_alt[0], hy, hx)
#print(digital1[2])
#print(digital1_alt[1])
# print('eps', eps, 'step', h)
# Метод варианта из ЛР3

exit()


print(r'\tiny')
print(r'\hfill\break')
print(r'\begin{tabular}{|'+'>{\small} c|'*(10)+'}')
print(r'\hline')
print(r'\Omega = (x, y) & eps & hy & hx & $||x_{appr} - x||_{sbcg} $ & $it_{sbcg} $ & $time_{sbcg} $ & $||x_{appr} - x||_{jac} $ & $it_{jac} $ & $time_{jac} $\\')
print(r'\hline')
for i1 in range(2):
    for i2 in range(2):
        for i3 in range(2):
            for i4 in range(2):
                [y0, y1, x0, x1] = mas_ends[i1]
                eps = mas_eps[i2]
                hy = mas_hy[i3]
                hx = mas_hx[i4]
                tm1 = time.time()
                U_appr, iters = _bicg_stab_spec(x0, x1, y0, y1, hx, hy, eps, f, g, u)
                tm2 = time.time()
                clear = solve_clear_spec(u)
                print('{} - {}, {} - {}'.format(x0, x1, y0 ,y1), end = ' & ')
                print(eps, end=' & ')
                print(hy, end=' & ')
                print(hx, end=' & ')
                print('%2.6e' % scalar_spec(U_appr - clear, U_appr - clear) ** 0.5, end=' & ')
                print(iters, end=' & ')
                print('%2.3e' % (tm2 - tm1), end=' & ')

                tm1 = time.time()
                U_appr, iters = jac_specified(f, u)
                tm2 = time.time()
                print('%2.6e' % scalar_spec(U_appr - clear, U_appr - clear) ** 0.5, end=' & ')
                print(iters, end=' & ')
                print('%2.3e' % (tm2 - tm1), end=r'\\')
                print()
                print(r'\hline')
print(r'\end{tabular}')
print(r'\hfill\break')
exit()