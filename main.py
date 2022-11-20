import numpy as np
import matplotlib.pyplot as plt

form = [
    [1, 1, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [0, 1, 1, 1, 1],
    [0, 0, 1, 1, 0]
]
form = [
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
]
form_size1 = 4
form_size2 = 5

def u_1(y, x, a=2, b=5, c=7):
    return a * x + b * y + c


def u_2(y, x, a=2, b=5, c=7):
    return a * x ** 2 + b * y + c


def u_3(y, x, a=2, b=5, c=7):
    return a * x ** 2 + b * y ** 2 + c


def u_4(y, x):
    return np.sin(x) - np.cos(y)


def fi_1(y, x):
    return 0


def fi_2(y, x):
    return -4


def fi_3(y, x):
    return -14


def fi_4(y, x):
    return -np.sin(x) + np.cos(y)


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
    for i in range(1, N1-1):
        for j in range(1, N2-1):
            answer[i, j] = right_part2(oper2, hy, hx, i, j)
    return answer


def right_part2(oper2, hy, hx, i, j):
    x = hx*j
    y = hy*i
    opi1 = oper2[i - 1, j]
    opi2 = oper2[i + 1, j]
    opi4 = oper2[i, j - 1]
    opi5 = oper2[i, j + 1]
    return -((opi1 - 2*oper2[i, j] + opi2)/hy**2 + (opi4 - 2*oper2[i, j] + opi5)/hx**2)


def _bicg_stab_spec(x0, x1, y0, y1, hx, hy, eps, f, g, u_fun):
    Ny = int((y1 - y0) / hy) + 1
    Nx = int((x1 - x0) / hx) + 1
    u = np.zeros((Ny, Nx))
    b = np.zeros((Ny, Nx))
    for i in range(Ny):
        u[i, 0] = g(y0 + i*hy, 0)
        u[i, Nx-1] = g(y0 + i*hy, 2)
    for i in range(Nx):
        u[0, i] = g(x0 + i*hx, 1)
        u[Ny-1, i] = g(x0 + i*hx, 3)
    for i in range(1, Ny - 1):  # задаём узлы
        for j in range(1, Nx - 1):
            u[i][j] = u[i][0]
            b[i][j] = f(y0 + i*hy, x0 + j*hx)
    x = u.copy()
    # Сформировали начальное приближение решения - матрицу u, c которой будем работать как с вектором, т.е. операция матричного умножения A на u будет действовать на матрциу u как на вектор
    # Сформировали вектор правой части - f(y,x)

    Au = multiply_matr_as_vec2(x, Ny, Nx, hy, hx)
    r = b - Au
    r_ = r.copy()
    r_[0][0] += np.sign(r_[0][0])*10 + 1
    p = r
    NN = int((Nx)*(Ny)/1)
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
        #if abs(w) < eps:
        #if np.linalg.norm(r)/np.linalg.norm(b) < eps:
        #r_delta = r_prev - r
        if scalar(r, r)**0.5 < eps:
            break_iter = j
            break
        bt = scalar(r, r_) * al / (scalar(r_prev, r_) * w)
        p = r + bt * (p - w * Ap)
    mistake_second_norm = 0
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            u[i, j] = x[i, j]
            mistake_second_norm += (u[i, j] - u_fun(hy*i, hx*j))**2
    return u, mistake_second_norm**0.5, scalar(r, r)**0.5, break_iter


def fill_G(f, i, j, hy, hx, N_, _N, matr):
    global x0, y0, x1, y1
    if i == -1:
        for i in range(N_, _N):
            matr[i][j] = f(y0 + i*hy, x0 + j*hx)
    else:
        for j in range(N_, _N):
            matr[i][j] = f(y0 + i*hy, x0 + j*hx)

def jac_specified(f, u):
    global x0, x1, y0, y1, hy, hx, eps, form, form_size1, form_size2   # x0, x1, y0, y1 - размер одной подобласти
    N1 = int((y1 - y0) / hy)    # Размеры одной подобласти
    N2 = int((x1 - x0) / hx)
    u_matrix = np.zeros((form_size1 * N1 + 1, form_size2 * N2 + 1))

    # Заполнение граничных точек по всей карте области
    for ii in range(form_size1):
        for jj in range(form_size2):
            if form[ii][jj] == 0:
                continue
            if ii == 0 or ii == form_size1 - 1: # Верх или низ области
                fill_G(u, (ii + int(ii == form_size1 - 1))*N1, -1, hy, hx, jj*N2, (jj+1)*N2, u_matrix)
            if jj == 0 or jj == form_size2 - 1:
                fill_G(u, -1, (jj + int(jj == form_size2 - 1))*N2, hy, hx, ii*N1, (ii+1)*N1, u_matrix)

            if (ii + 1 < form_size1) and (form[ii + 1][jj] == 0):
                fill_G(u, (ii+1) * N1, -1, hy, hx, jj * N2, (jj + 1) * N2, u_matrix)
            if (ii - 1 > 0) and (form[ii - 1][jj] == 0):
                fill_G(u, ii * N1, -1, hy, hx, jj * N2, (jj + 1) * N2, u_matrix)

            if (jj + 1 < form_size2) and (form[ii][jj + 1] == 0):
                fill_G(u, -1, (jj + 1)*N2 + 1, hy, hx, ii*N1, (ii+1)*N1, u_matrix)
            if (jj - 1 > 0) and (form[jj - 1][jj] == 0):
                fill_G(u, -1, jj*N2 + 1, hy, hx, ii*N1, (ii+1)*N1, u_matrix)


    for ii in range(form_size1):          # задаём узлы
        for jj in range(form_size2):
            if form[ii][jj] == 0:
                continue
            for i in range(ii*N1 + 1, ii*N1+N1 - 1):
                for j in range(jj*N2 + 1, jj*N2+N2 - 1):
                    u_matrix[i][j] = u_matrix[ii*N1][j]
            if (ii + 1 < form_size1) and (form[ii + 1][jj] == 1):
                fill_G(u, ii * N1 + 1, -1, hy, hx, jj * N2, (jj + 1) * N2, u_matrix)
            if (jj + 1 < form_size2) and (form[ii][jj + 1] == 1):
                fill_G(u, -1, jj*N2 + 1, hy, hx, ii*N1, (ii+1)*N1, u_matrix)
    draw_graph(u_matrix, hy, hx, -1)
    flag = False
    q = 0
    while not flag:
        q += 1
        old_u_matrix = u_matrix.copy()
        max_delta = 0
        for ii in range(form_size1):
            for jj in range(form_size2):
                if form[ii][jj] == 0:
                    continue
                for i in range(ii * N1 + 1, ii * N1 + N1 - 1):
                    for j in range(jj * N2 + 1, jj * N2 + N2 - 1):
                        u_matrix[i][j] = (1 / 4) * (
                                    old_u_matrix[i + 1][j] + old_u_matrix[i - 1][j] + old_u_matrix[i][j - 1] + old_u_matrix[i][
                                j + 1]) + (h ** 2 / 4) * f(i * h, j * h)
                        if abs(u_matrix[i][j] - old_u_matrix[i][j]) > max_delta:
                            max_delta = abs(u_matrix[i][j] - old_u_matrix[i][j])
                if (ii + 1 < form_size1) and (form[ii + 1][jj] == 1):
                    for j in range(jj * N2 + 1, jj * N2 + N2 - 1):
                        i = ii * N1 + N1
                        u_matrix[i][j] = (1 / 4) * (
                                old_u_matrix[i + 1][j] + old_u_matrix[i - 1][j] + old_u_matrix[i][j - 1] + old_u_matrix[i][
                            j + 1]) + (h ** 2 / 4) * f(i * h, j * h)
                        if abs(u_matrix[i][j] - old_u_matrix[i][j]) > max_delta:
                            max_delta = abs(u_matrix[i][j] - old_u_matrix[i][j])
                if (jj + 1 < form_size2) and (form[ii][jj + 1] == 1):
                    for i in range(ii * N1 + 1, ii * N1 + N1 - 1):
                        j = jj * N2 + N2
                        u_matrix[i][j] = (1 / 4) * (
                                old_u_matrix[i + 1][j] + old_u_matrix[i - 1][j] + old_u_matrix[i][j - 1] + old_u_matrix[i][
                            j + 1]) + (h ** 2 / 4) * f(i * h, j * h)
                        if abs(u_matrix[i][j] - old_u_matrix[i][j]) > max_delta:
                            max_delta = abs(u_matrix[i][j] - old_u_matrix[i][j])
        if max_delta < eps: flag = True
    print('iters needed', q)
    return u_matrix


def jac(f, g):
    global x0, x1, y0, y1, hy, hx, eps, form, form_size1, form_size2   # x0, x1, y0, y1 - размер одной подобласти
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
    print('iters needed', q)
    return u_matrix


def solve_clear(u):
    global x0, x1, y0, y1, h
    clear = np.zeros((int((y1 - y0) / h + 1), int((x1 - x0) / h + 1)))
    N1 = int((y1 - y0) / h + 1)
    N2 = int((x1 - x0) / h + 1)
    for i in range(N1):
        for j in range(N2):
            clear[i][j] = u(i * h, j * h)
    return clear


def draw_graph(digital, hy, hx, mode = 0):
    global x0, x1, y0, y1, form_size1, form_size2   # x0, x1, y0, y1 - размер одной подобласти
    if mode == -1:
        Ny = form_size1 * int((y1 - y0) / hy) + 1
        Nx = form_size2 * int((x1 - x0) / hx) + 1
        yY, xX = [y0 + i * hy for i in range(Ny)], [x0 + j * hx for j in range(Nx)]
    else:
        Ny = int((y1 - y0) / hy) + 1
        Nx = int((x1 - x0) / hx) + 1
        yY, xX = [y0 + i * hy for i in range(Ny)], [x0 + j * hx for j in range(Nx)]
    xgrid, ygrid = np.meshgrid(xX, yY)
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    axes.plot_surface(xgrid, ygrid, digital)
    plt.show()


x0 = 0
x1 = 1
y0 = 0
y1 = 2
eps = 0.001
h = 0.1
hy = h
hx = h
mas_funs = [[u_1, fi_1, g_1], [u_2, fi_2, g_2], [u_3, fi_3, g_3], [u_4, fi_4, g_4]]
[u, f, g] = mas_funs[0]

clear = solve_clear(u)
digital2 = jac_specified(f, g)
draw_graph(digital2, hy, hx, -1)


# print(clear[7])
# print(digital2[7])
# print('Norm', np.linalg.norm(digital2 - clear, ord='fro'))
exit()
digital1 = jac(f, g)
digital1_alt = _bicg_stab_spec(x0, x1, y0, y1, hx, hy, eps, f, g, u)
draw_graph(clear, hy, hx)
draw_graph(digital1, hy, hx)
draw_graph(digital1_alt[0], hy, hx)
# print('eps', eps, 'step', h)
# Метод варианта из ЛР3