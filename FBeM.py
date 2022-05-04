import numpy as np


class iGranule:

    def __init__(self):
        """
        Initialize object
        """
        self.l = 0
        self.lambd = 0
        self.Lambd = 0
        self.L = 0

    def fits(self, xj, rho):
        """
        Check if xj fits in this granule
        :param xj:
        :param rho:
        :return:
        """
        a = (xj >= (self.midpoint() - rho / 2))
        b = (xj <= (self.midpoint() + rho / 2))

        return a and b

    def midpoint(self):
        """
        Returns the midpoint of the granule
        :return:
        """
        return (self.lambd + self.Lambd) / 2

    def com_similarity(self, x):
        """
        Check similarity with complete observation x
        :param x: complete observation
        :param n:
        :return:
        """

        aux = abs(self.l - x) + abs(self.lambd - x) + abs(self.Lambd - x) + abs(self.L - x)

        return aux


class oGranule:

    def __init__(self):
        self.u = 0
        self.ups = 0
        self.Ups = 0
        self.U = 0
        self.coef = []

    def p(self, x):
        """
        Functional consequent with complete data
        :param x:
        :return:
        """

        # Compute dot multiplication. The one is added in order to compute
        # a0
        x = np.insert(x, 0, 1)
        out = sum(self.coef * x)

        return out

    def p_mp(self, x):
        """
        Functional consequent with complete data, taking into account midpoint
        :param x:
        :return:
        """

        # Compute dot multiplication. The one is added in order to compute
        # a0
        x = np.insert(x, 0, 1)
        out = sum(self.coef * x)

        return out

    def midpoint(self):
        """
        Returns the midpoint of the granule
        :return:
        """
        return (self.ups + self.Ups) / 2

    def fits_(self, y, rho):
        """
        Check if xj fits in this granule
        :param xj:
        :param rho:
        :return:
        """
        a = (y >= self.midpoint() - rho / 2)
        b = (y <= self.midpoint() + rho / 2)

        return a and b

    def fits(self, y):
        """
        Check if xj fits in this granule
        :param y:
        :return:
        """

        return self.coef == y


def min_u(granules, indices, m):
    """
    Used for convex hull
    :param granules:
    :param indices:
    :param m:
    :return:
    """
    min = np.inf
    index_min = None

    for i in indices:
        for j in range(0, m):
            if granules[i].oGranules[j].u < min:
                min = granules[i].oGranules[j].u
                index_min = i

    return min


def max_U(granules, indices, m):
    """
    Used for convex hull
    :param granules:
    :param indices:
    :param m:
    :return:
    """
    max = -np.inf
    index_max = None

    for i in indices:
        for j in range(0, m):
            if granules[i].oGranules[j].U > max:
                max = granules[i].oGranules[j].U
                index_max = i

    return max


def power(list, power=1):
    """
    Do the power operation over the list's items
    :param list:
    :param power:
    :return:
    """
    return [item ** power for item in list]


def div(list, operand=1):
    """
    Do the division operation over the list' items
    :param list:
    :param operand:
    :return:
    """
    return [item / operand for item in list]


def sqroot(list):
    """
    Do the sqrt operation over the list's items
    :param list:
    :return:
    """
    return [np.sqrt(item) for item in list]


def sub(list1, list2):
    """
    Subtract a list from another item by item
    :param list1:
    :param list2:
    :return:
    """
    if len(list1) != len(list2):
        raise Exception("Listas de tamanho diferente")

    return [list1[i] - list2[i] for i in range(0, len(list1))]


def sum_dot(s, i, p):
    """
    Sum the items of a list multiplied by a factor from p.
    Used for calculation of affine functions
    :param s:
    :param i:
    :param p:
    :return:
    """
    sumd = 0

    for l in range(0, len(i)):
        sumd += s[i[l]] * float(p[l])

    return sumd


def sum_(s):
    """
    Sum items of a list
    :param s:
    :return:
    """
    sumd = 0

    for l in s:
        sumd += l

    return sumd


def sum_specific(s, i):
    """
    Sum the i items of a list
    :param s:
    :param i:
    :return:
    """
    sumd = 0

    for l in i:
        sumd += s[l]

    return sumd


def normalize(array, min, max):
    """
    Normalize data
    :param array:
    :param min:
    :param max:
    :return:
    """
    return [(x - min) / (max - min) for x in array]


def least_sqr(A, B):
    """
    Using least sqaures numpy
    :param A:
    :param B:
    :return:
    """
    return np.linalg.lstsq(np.array(A), np.array(B), rcond=None)[0].tolist()


def lstsqr_pinv(A, B):
    b = copy.deepcopy(B)
    B_crude = copy.deepcopy(b)
    B = []
    for i in b:
        B.append([i])

    list_ret = np.dot(np.linalg.pinv(A), B).tolist()
    list_ = []
    for i in range(0, len(list_ret)):
        for j in range(0, len(list_ret[i])):
            list_.append(float(list_ret[i][j]))

    return list_


def mldivide(A, B):
    A_crude = copy.deepcopy(A)
    A = np.array(A)
    b = copy.deepcopy(B)
    B_crude = copy.deepcopy(b)
    B = []
    for i in b:
        B.append([i])

    B = np.array(B)
    num_vars = A.shape[1]
    rank = np.linalg.matrix_rank(A)
    solutions = []

    if rank == num_vars:
        sol = least_sqr(A_crude, B_crude)
        return sol
    else:
        for nz in combinations(range(num_vars), rank):
            try:
                sol = np.zeros((num_vars, 1))
                sol[nz, :] = np.asarray(np.linalg.solve(A[:, nz], B))
                solutions.append(sol)
            except np.linalg.LinAlgError:
                pass

    dist_to_zero = np.zeros((len(solutions), len(solutions[0])))
    min_dist = np.inf
    closest = []
    solutions = np.array(solutions)

    for i in range(0, len(solutions)):
        aux = 0
        for j in range(0, len(solutions[i].tolist())):
            for k in range(0, len(solutions[i][j].tolist())):
                aux += abs(0 - solutions[i][j][k])
        dist_to_zero[i][j] = aux
        if aux < min_dist:
            min_dist = aux
            closest = [i, j]

    ret_array = []
    array = solutions[closest[0]].tolist()
    for i in array:
        ret_array.append(i[0])

    return ret_array


def mldivide_matlab(A, B):
    """
    Not working
    :param A:
    :param B:
    :return:
    """
    b = copy.deepcopy(B)
    B_crude = copy.deepcopy(b)
    B = []
    for i in b:
        B.append([i])

    c_matrix = matrix_divide.mldivide(A, B).tolist()

    ret_array = []
    for i in c_matrix:
        ret_array.append(i[0].real)

    return ret_array


def solve_minnonzero(A, b):
    A = np.array(A)
    b = copy.deepcopy(b)
    B_crude = copy.deepcopy(b)
    B = []
    for i in b:
        B.append([i])

    x1, res, rnk, s = np.linalg.lstsq(A, B, rcond=-1)
    if rnk == A.shape[1]:
        ret_array = []
        for i in range(0, len(x1)):
            for j in range(0, len(x1[i])):
                ret_array.append(x1[i][j])

        return ret_array  # nothing more to do if A is full-rank

    Q, R, P = qr(A.T, mode='full', pivoting=True)
    Z = Q[:, rnk:].conj()
    C = np.linalg.solve(Z[rnk:], -x1[rnk:])
    array = (x1 + Z.dot(C)).tolist()

    ret_array = []
    for i in range(0, len(array)):
        for j in range(0, len(array[i])):
            ret_array.append(array[i][j])

    return ret_array


def ls_nnls(A, b):
    test_A = np.array(A)

    try:
        test_b = np.array(b)
        output = list(nnls(test_A, test_b)[0])
        return output
    except Exception as e:
        print(str(e))


import numpy as np
import copy


class Granule:
    def __init__(self):
        self.iGranules = []
        self.oGranules = []
        self.act = 0
        self.points = []
        self.xs = []
        self.ys = []

    def granule_similarity(self, granule):
        """

        :param granule:
        :return:
        """
        term = 0
        for i in range(0, len(self.iGranules)):
            term = term + abs(self.iGranules[i].l - granule.iGranules[i].l) + \
                   abs(self.iGranules[i].lambd - granule.iGranules[i].lambd) + \
                   abs(self.iGranules[i].Lambd - granule.iGranules[i].Lambd) + \
                   abs(self.iGranules[i].L - granule.iGranules[i].L)

        return term

    def calculate_rls(self):
        """
        Calculate least squares
        :return:
        """
        XX = []
        colOut = copy.deepcopy(self.ys)
        col = copy.deepcopy(self.xs)

        for i in range(0, len(col)):
            XX.insert(i, copy.deepcopy(col[i]))
            XX[i].insert(0, 1)

        # try:
        for k in range(0, len(self.oGranules)):
            # result = least_sqr(XX, colOut)
            # result = lstsqr_pinv(XX, colOut)
            result = mldivide_matlab(XX, colOut)
            # result = mldivide(XX, colOut)
            # result = ls_nnls(XX, colOut)
            # result = solve_minnonzero(XX, colOut)
            self.oGranules[k].coef = result
        # except Exception as e:
        #    print(str(e))
        #    pass

    def get_granule_for_3d_plotting(self, only_core=0):
        if len(self.iGranules) > 2: raise Exception("Not possible to plot")
        if len(self.oGranules) > 1: raise Exception("Not possible to plot")

        x = self.iGranules
        y = self.oGranules

        if only_core:
            return np.array([
                [x[0].lambd, x[1].lambd, y[0].ups],
                [x[0].Lambd, x[1].lambd, y[0].ups],
                [x[0].lambd, x[1].Lambd, y[0].ups],
                [x[0].Lambd, x[1].Lambd, y[0].ups],
                [x[0].lambd, x[1].lambd, y[0].Ups],
                [x[0].Lambd, x[1].lambd, y[0].Ups],
                [x[0].lambd, x[1].Lambd, y[0].Ups],
                [x[0].Lambd, x[1].Lambd, y[0].Ups]
            ])

        return np.array([
            [x[0].l, x[1].l, y[0].u],
            [x[0].L, x[1].l, y[0].u],
            [x[0].l, x[1].L, y[0].u],
            [x[0].L, x[1].L, y[0].u],
            [x[0].l, x[1].l, y[0].U],
            [x[0].L, x[1].l, y[0].U],
            [x[0].l, x[1].L, y[0].U],
            [x[0].L, x[1].L, y[0].U]
        ])

    def get_faces_for_3d_plotting(self):
        if len(self.iGranules) > 2: raise Exception("Not possible to plot")
        if len(self.oGranules) > 1: raise Exception("Not possible to plot")

        granule = self.get_granule_for_3d_plotting()

        return [
            [granule[0, :], granule[1, :], granule[3, :], granule[2, :]],
            [granule[1, :], granule[5, :], granule[7, :], granule[3, :]],
            [granule[2, :], granule[3, :], granule[7, :], granule[6, :]],
            [granule[0, :], granule[4, :], granule[6, :], granule[2, :]],
            [granule[6, :], granule[7, :], granule[5, :], granule[4, :]],
            [granule[0, :], granule[4, :], granule[5, :], granule[1, :]]
        ]


import numpy as np
import operator as op
import copy


# FBeM Class
class FBeM:
    def __init__(self, n: int = 2):
        """
        Initialization of the object
        """
        self.acc = []
        self.right = []
        self.wrong = []
        self.output = []
        self.data = []
        self.c = 0
        self.h = 0
        self.rho = 0.7
        self.n = n
        self.m = 1
        self.hr = 48
        self.alpha = 0
        self.eta = 0.5
        self.counter = 1

        # ROC Curve
        self.tp = []
        self.tn = []
        self.fp = []
        self.fn = []

        self.granules = []
        self.ys = []

        self.P = []  # C
        self.store_num_rules = []
        self.vec_rho = []
        self.debug = False

        # debugging variables
        self.merged_granules = 0
        self.created_granules = 0
        self.deleted_granules = 0
        self.file = open("log.txt", "w")

    def create_new_granule(self, index, x, y):
        """
        Create new granule
        :param index:
        :param x:
        :param y:
        :return:
        """
        g = Granule()

        """ Input granules """
        for i in range(0, self.n):
            newGranule = iGranule()
            newGranule.l = x[i]
            newGranule.lambd = x[i]
            newGranule.Lambd = x[i]
            newGranule.L = x[i]

            g.iGranules.insert(i, newGranule)

        """ Output granules """
        for k in range(0, self.m):
            newOGranule = oGranule()
            newOGranule.u = y[k]
            newOGranule.ups = y[k]
            newOGranule.Ups = y[k]
            newOGranule.U = y[k]

            """ Coefficients alpha """
            newOGranule.coef = y[0]

            g.points.append(self.h)  #
            g.act = 0
            g.oGranules.insert(k, newOGranule)

        g.xs.append(x)
        g.ys.append(y[0])

        self.granules.insert(index, g)
        self.c += 1
        self.created_granules += 1

        """ Debugging """
        """
        if self.debug:
            ax = create_3d_space()
            ax = plot_granule_3d_space(granule=g, ax=ax, i=3)

            plot_show()
        """
        """ /Debugging """

    def create_check_imaginary_granule(self, granule1, granule2):
        """
        Create new imaginary granule as a junction of two granules
        And check the possibility of this granule to become real
        :param granule1:
        :param granule2:
        :return:
        """
        g = Granule()
        J = 0
        K = 0

        """ Input granules """
        for i in range(0, self.n):
            new_granule = iGranule()
            gran_1 = granule1.iGranules[i]
            gran_2 = granule2.iGranules[i]

            new_granule.l = min([gran_1.l, gran_2.l])
            new_granule.lambd = min([gran_1.lambd, gran_2.lambd])
            new_granule.Lambd = max([gran_1.Lambd, gran_2.Lambd])
            new_granule.L = max([gran_1.L, gran_2.L])

            if new_granule.midpoint() - self.rho / 2 <= new_granule.l and \
                    new_granule.midpoint() + self.rho / 2 >= new_granule.L:
                J = J + 1

            g.iGranules.insert(i, new_granule)

        """ Output granules """
        for k in range(0, self.m):
            newOGranule = oGranule()
            gran_1 = granule1.oGranules[k]
            gran_2 = granule2.oGranules[k]

            if gran_1.coef == gran_2.coef:
                K = K + 1

            """ Coefficients alpha """
            g.points = granule1.points + granule2.points
            g.xs = granule1.xs + granule2.xs
            g.ys = granule1.ys + granule2.ys
            newOGranule.coef = gran_2.coef
            g.act = 0

            newOGranule.u = min([gran_1.u, gran_2.u])
            newOGranule.ups = min([gran_1.ups, gran_2.ups])
            newOGranule.Ups = max([gran_1.Ups, gran_2.Ups])
            newOGranule.U = max([gran_1.U, gran_2.U])

            g.oGranules.insert(k, newOGranule)

        """ Check if new imaginary granule can become real """
        become_real = False
        if J + K == self.n + self.m:
            self.c = self.c + 1

            self.granules.insert(self.c, g)
            self.merged_granules += 1
            become_real = True

        """ Debugging """
        """
        if self.debug:
            ax = create_3d_space()
            ax = plot_granule_3d_space(granule=granule1,ax=ax)
            ax = plot_granule_3d_space(granule=granule2,ax=ax,i=2)
            ax = plot_granule_3d_space(granule=g, ax=ax, i=3)

            plot_show()
        """
        """ /Debugging """

        return become_real

    def learn(self, x, y):
        """
        Learn as x and y enters
        :param x: observations
        :param y: expected output
        :return:
        """
        # starting from scratch
        if self.h == 0:

            self.ys.append(y[0])

            """ create new granule anyways """
            self.create_new_granule(self.c, x, y)

            self.P.append(round(np.random.rand()))
            self.store_num_rules.insert(self.h, self.c)
            self.vec_rho.insert(self.h, self.rho)

            if y[0] == self.P[0]:
                self.right.insert(self.h, 1)
                self.wrong.insert(self.h, 0)

                if y[0] == 1:
                    self.tp.insert(self.h, 1)
                    self.fp.insert(self.h, 0)
                    self.tn.insert(self.h, 0)
                    self.fn.insert(self.h, 0)
                else:
                    self.tp.insert(self.h, 0)
                    self.fp.insert(self.h, 0)
                    self.tn.insert(self.h, 1)
                    self.fn.insert(self.h, 0)
            else:
                self.right.insert(self.h, 0)
                self.wrong.insert(self.h, 1)

                if y[0] == 0:
                    self.tp.insert(self.h, 0)
                    self.fp.insert(self.h, 1)
                    self.tn.insert(self.h, 0)
                    self.fn.insert(self.h, 0)
                else:
                    self.tp.insert(self.h, 0)
                    self.fp.insert(self.h, 0)
                    self.tn.insert(self.h, 0)
                    self.fn.insert(self.h, 1)

            # self.rmse.append(np.sqrt(sum_(power(sub(self.ys, self.P), 2)) / (self.h)))
            # self.rmse.append(np.sqrt(sum_(power(sub(self.ys, self.P), 2))))
            # self.ndei.append(np.inf)

            self.file.write(str(self.h) + "\t" + str(self.P[self.h]) + "\t1\n")

            """
            # Avoiding division by 0
            if np.std(self.ys):
                self.ndei.append(self.rmse[self.h] / np.std(self.ys))
            else:
                self.ndei.append(np.inf)
            """
        else:

            """
            Incremental learning
            Test - y is not available yet
            Check rules that can accommodate x
            """
            I = []
            for i in range(0, self.c):
                J = 0
                for j in range(0, self.n):
                    if self.granules[i].iGranules[j].fits(xj=x[j], rho=self.rho):
                        J += 1

                if J == self.n:
                    """ granule i fits x! """
                    I.append(i)

            """ Check degree of similarity """
            S = np.array([])
            # for i in range(0, self.c):
            #    aux = 0
            #    for j in range(0, self.n):
            #        aux += self.granules[i].iGranules[j].com_similarity(x=x[j])

            #    S.insert(i, 1 - ((1 / (4 * self.n)) * aux))

            # I = [S.index(max(S))]

            for i in range(self.c):
                aux = sum([self.granules[i].iGranules[j].com_similarity(x=x[j]) for j in range(self.n)])

                S = np.insert(S, i, 1 - ((1 / (4 * self.n)) * aux))

            if len(I) == 0: I = [np.argmax(S)]

            # Vector for plot ROC
            # self.oClass1 = 0
            # self.oClass2 = 0

            # Otimizar e entender depois
            # for i in range(0, self.c):
            #    if self.granules[i].oGranules[0].coef == 0:
            #        self.oClass1 = max(self.oClass1, S[i])
            #    elif self.granules[i].oGranules[0].coef == 1:
            #        self.oClass2 = max(self.oClass2, S[i])
            #
            # output = []
            # output.append(self.oClass1 / (self.oClass1 + self.oClass2))
            # output.append(self.oClass2 / (self.oClass1 + self.oClass2))

            # self.output.insert(self.h, output)

            """ Prediction """
            # print("->", self.granules[I[0]].oGranules[0].coef)
            if self.granules[I[0]].oGranules[0].coef == []:
                print("chegou")
            self.P.insert(self.h, self.granules[I[0]].oGranules[0].coef)

            """ y becomes available // cumulative """
            if y[0] == self.P[len(self.P) - 1]:
                self.right.insert(self.h, 1)
                self.wrong.insert(self.h, 0)

                if y[0] == 1:
                    self.tp.insert(self.h, 1)
                    self.fp.insert(self.h, 0)
                    self.tn.insert(self.h, 0)
                    self.fn.insert(self.h, 0)
                else:
                    self.tp.insert(self.h, 0)
                    self.fp.insert(self.h, 0)
                    self.tn.insert(self.h, 1)
                    self.fn.insert(self.h, 0)
            else:
                self.right.insert(self.h, 0)
                self.wrong.insert(self.h, 1)

                if y[0] == 0:
                    self.tp.insert(self.h, 0)
                    self.fp.insert(self.h, 1)
                    self.tn.insert(self.h, 0)
                    self.fn.insert(self.h, 0)
                else:
                    self.tp.insert(self.h, 0)
                    self.fp.insert(self.h, 0)
                    self.tn.insert(self.h, 0)
                    self.fn.insert(self.h, 1)

            """
            Train
            Calculate granular consequent
            Check rules that accommodate x
            """
            """
            stop = 0
            if self.h == 24:
                stop = 1
            """

            I = []
            for i in range(0, self.c):
                J = 0
                K = 0

                #
                for j in range(0, self.n):
                    """ xj inside j-th expanded region? """
                    if self.granules[i].iGranules[j].fits(xj=x[j], rho=self.rho):
                        J += 1
                    # print(x, j, self.granules[i].iGranules[j].fits(xj=x[j], rho=self.rho), x[j], self.rho)
                # print('--- y')

                for k in range(0, self.m):
                    """ yk inside k-th expanded region? """
                    if self.granules[i].oGranules[k].fits(y=y[k]):
                        K += 1
                    # print(y, k, self.granules[i].oGranules[k].fits(y=y[k]), y[k])
                # print('---')
                if J + K == self.n + self.m:
                    I.append(i)

            """ Case 0: no granule fits x """
            if len(I) == 0:
                self.create_new_granule(x=x, y=y, index=self.c)
            else:
                """
                Adaptation of the most qualified granule
                If more than one granule fits the observation
                """
                if len(I) >= 2:
                    S = []
                    for i in range(0, len(I)):
                        aux = 0
                        for j in range(0, self.n):
                            aux += self.granules[I[i]].iGranules[j].com_similarity(x=x[j])

                        S.insert(I[i], 1 - ((1 / (4 * self.n)) * aux))

                    # print(S)
                    I = I[S.index(max(S))]
                else:
                    I = I[0]
                print(f"{I}")

                """ Adapting antecedent of granule I """
                """ Debugging """
                """
                if self.debug and stop:
                    ax = create_3d_space()
                    ax = plot_granule_3d_space(granule=self.granules[I], ax=ax, i=3)
                    ax.scatter(x[0], x[1], self.granules[I].oGranules[0].p(x=x))
                    ax.set_title("Teste")
                    plot_show()
                """
                """ /Debugging """
                for j in range(0, self.n):
                    ig = self.granules[I].iGranules[j]
                    mp = ig.midpoint()
                    if mp - self.rho / 2 < x[j] < ig.l:  # 1 ok
                        ig.l = x[j]  # Support expansion
                    if mp - self.rho / 2 < x[j] < ig.lambd:  # 2 ok
                        ig.lambd = x[j]  # core expansion
                    if ig.lambd < x[j] < mp:  # 3 ok
                        ig.lambd = x[j]  # core contraction
                    if mp < x[j] < mp + self.rho / 2:  # 4 ok
                        ig.lambd = mp  # core contraction
                    if mp - self.rho / 2 < x[j] < mp:
                        ig.Lambd = mp  # Core contraction
                    if mp < x[j] < ig.Lambd:
                        ig.Lambd = x[j]  # core contraction
                    if ig.Lambd < x[j] < mp + self.rho / 2:
                        ig.Lambd = x[j]  # core expansion
                    if ig.L < x[j] < mp + self.rho / 2:
                        ig.L = x[j]  # support expansion
                    """
                    if ig.l < x[j] < ig.lambd:
                        ig.lambd = x[j]  # Core expansion
                    if mp < x[j] < ig.Lambd:
                        ig.Lambd = x[j]  # Core contraction
                    if ig.Lambd < x[j] < ig.L:
                        ig.Lambd = x[j]  # Core expansion
                    if ig.Lambd < x[j] < mp + self.rho / 2:
                        ig.Lambd = x[j]  # Support expansion
                    if ig.L < x[j] < mp + self.rho / 2:
                        ig.L = x[j]  # Support expansion
                    """

                """ Check if support contraction is needed """
                for j in range(0, self.n):
                    ig = self.granules[I].iGranules[j]
                    mp = self.granules[I].iGranules[j].midpoint()

                    """ Inferior support """
                    if mp - self.rho / 2 > ig.l:
                        ig.l = mp - self.rho / 2
                        if mp - self.rho / 2 > ig.lambd:
                            ig.lambd = mp - self.rho / 2

                    """ Superior Support """
                    if mp + self.rho / 2 < ig.L:
                        ig.L = mp + self.rho / 2
                        if mp + self.rho / 2 < ig.Lambd:
                            ig.Lambd = mp + self.rho / 2

                """ Storing sample """
                self.granules[I].points.append(self.h)
                self.granules[I].ys.append(y[0])
                self.granules[I].xs.append(x)
                self.granules[I].act = 0

                """ Debugging """
                """
                if self.debug:
                    ax = create_3d_space()
                    ax = plot_granule_3d_space(granule=self.granules[I], ax=ax, i=3)
                    ax.scatter(x[0], x[1], self.granules[I].oGranules[0].p(x=x), c="green")
                    ax.scatter(x[0], x[1], y[0], c="yellow")
                    ax.set_title("Teste2")
                    plot_show()
                """
                """ /Debugging """

        """ Deleting granules if needed """
        granules_to_delete = []
        for K in range(0, self.c):
            self.granules[K].act += 1
            if self.granules[K].act >= self.hr:
                del self.granules[K]
                self.c -= 1
                break

        """ Coarsening granular structure """
        self.alpha += 1

        if self.alpha == self.hr:
            if self.c >= 3:
                """
                Calculating the similarity between granules
                While choosing two closest granules acording to S
                """
                S = np.zeros((self.c, self.c))
                gra1 = []
                ind1 = -1
                gra2 = []
                ind2 = -1
                aux = -np.inf

                for i1 in range(0, self.c):
                    for i2 in range(i1 + 1, self.c):
                        S[i1][i2] = self.granules[i1].granule_similarity(self.granules[i2])
                        S[i1][i2] = 1 - ((1 / (4 * self.n)) * S[i1][i2])
                        if aux < S[i1][i2]:
                            aux = S[i1][i2]
                            gra1 = self.granules[i1]
                            gra2 = self.granules[i2]
                            ind1 = i1
                            ind2 = i2

                res = self.create_check_imaginary_granule(granule1=gra1, granule2=gra2)
                if res:
                    """ deleting granules is possible """
                    del self.granules[ind1]
                    del self.granules[ind2]
                    self.c -= 2

            self.alpha = 0

        """ Adapt granules size """
        if self.counter == 1:
            self.b = self.c

        self.counter += 1

        if self.counter == self.hr:
            self.chi = self.c
            diff = self.chi - self.b

            if diff >= self.eta:
                self.rho *= (1 + diff / self.counter)  # increase rho
            else:
                self.rho *= (1 - (diff + self.eta) / self.counter)  # decrease rho

            self.counter = 1

        self.vec_rho.insert(self.h, self.rho)  # granules size along
        self.store_num_rules.insert(self.h, self.c)
        self.acc.insert(self.h, sum(self.right) / (sum(self.right) + sum(self.wrong)) * 100)
        self.h += 1

        """ Check if support contraction is needed """
        for i in range(0, self.c):
            for j in range(0, self.n):

                ig = self.granules[i].iGranules[j]
                mp = self.granules[i].iGranules[j].midpoint()

                """ Inferior support """
                if mp - self.rho / 2 > ig.l:
                    ig.l = mp - self.rho / 2
                    if mp - self.rho / 2 > ig.lambd:
                        ig.lambd = mp - self.rho / 2

                """ Superior Support """
                if mp + self.rho / 2 < ig.L:
                    ig.L = mp + self.rho / 2
                    if mp + self.rho / 2 < ig.Lambd:
                        ig.Lambd = mp + self.rho / 2


import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_color(x):
    if x == 0:
        return 'red'
    elif x == 1:
        return 'green'
    else:
        return 'blue'


def get_color_sigla(x):
    if x == 0:
        return 'r'
    elif x == 1:
        return 'g'
    else:
        return 'b'


def plot_granules(fbem):
    fig, ax = plt.subplots()

    for i in fbem.granules:
        for j in i.xs:
            if len(i.xs) == 1:
                plt.plot(j[0], j[1], get_color_sigla(i.oGranules[0].coef) + '*')
            else:
                plt.plot(j[0], j[1], 'k*')
        width = i.iGranules[0].L - i.iGranules[0].l if i.iGranules[0].L - i.iGranules[0].l > 0 else i.iGranules[0].L - \
                                                                                                    i.iGranules[
                                                                                                        0].l  # + .1
        height = i.iGranules[1].L - i.iGranules[1].l if i.iGranules[1].L - i.iGranules[1].l > 0 else i.iGranules[1].L - \
                                                                                                     i.iGranules[
                                                                                                         1].l  # + .1
        ax.add_patch(patches.Rectangle((i.iGranules[0].l, i.iGranules[1].l), width, height,
                                       color=get_color(i.oGranules[0].coef)))
    plt.show();

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_singular_output(fbem_instance, expected_output):
    """
    Plot singular output of predicted values from FBeM
    to be compared to the expected output
    :param fbem_instance: FBeM instance
    :param expected_output:
    :return:
    """
    # Plot singular output
    plt.figure()
    plt.plot(expected_output, 'k-', label="Expected output")
    plt.plot(fbem_instance.P, 'b-', label="Predicted output")
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.legend(loc=2)


def plot_granular_output(fbem_instance, expected_output):
    """
    Plot granular output predicted from FBeM
    to be compared to the expected output
    :param fbem_instance:
    :param expected_output:
    :return:
    """

    # Plot granular output
    plt.figure()
    plt.plot(expected_output, 'b-', label="Expected output")
    plt.plot(fbem_instance.PLB, 'r-', label="Lower bound")
    plt.plot(fbem_instance.PUB, 'g-', label="Upper bound")
    plt.legend(loc=2)


def plot_rmse_ndei(fbem_instance):
    """
    Plot RMSE and NDEI graphs from FBeM
    :param fbem_instance:
    :return:
    """

    plt.figure()
    plt.plot(fbem_instance.rmse, 'r-', label="RMSE")
    plt.plot(fbem_instance.ndei, 'g-', label="NDEI")
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.legend(loc=2)


def plot_rules_number(fbem_instance):
    """
    Plot the variation of number of FBeM rules
    :param fbem_instance:
    :return:
    """

    # Plot rule number
    plt.figure()
    plt.plot(fbem_instance.store_num_rules, 'r-', label="Number of rules")
    axes = plt.gca()
    axes.set_ylim([0, 30])
    plt.legend(loc=2)

def plot_acc(fbem_instance):
    """
    Plot accuraccy
    :param fbem_instance:
    :return:
    """
    plt.figure()
    plt.plot(fbem_instance.acc, 'r-', label="Accuracy")
    axes = plt.gca()
    axes.set_ylim([0, 110])
    plt.legend(loc=2)

def plot_rho_values(fbem_instance):
    """
    Plot the variation of number of rho values
    :param fbem_instance:
    :return:
    """

    plt.figure()
    plt.plot(fbem_instance.vec_rho, 'r-', label="Rho variation")
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.legend(loc=2)


def plot_granules_3d_space(fbem_instance, min=0, max=1, indices=[]):
    """
    Plot granules in 3D space

    :param fbem_instance:
    :param min:
    :param max:
    :param indices:
    :return:
    """

    if indices == []:
        indices = range(0, fbem_instance.c)

    colors = ["red", "blue", "black", "gray", "green", "cyan", "yellow", "pink", "fuchsia", "darkgray"]
    colors += colors
    colors += colors

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in indices:
        granule = fbem_instance.granules[i]
        gran = granule.get_granule_for_3d_plotting()
        faces = granule.get_faces_for_3d_plotting()

        # plot vertices
        ax.scatter3D(gran[:, 0], gran[:, 1], gran[:, 2], c=colors[i])

        # plot sides
        face1 = Poly3DCollection(verts=faces, facecolors=colors[i], linewidths=1, edgecolors=colors[i], alpha=.1)
        face1.set_alpha(0.25)
        face1.set_facecolor(colors=colors[i])
        ax.add_collection3d(face1)

        ax.text(gran[0, 0], gran[0, 1], gran[0, 1], s="gamma " + str(i) + " - Y=" +
                                                      str(fbem_instance.granules[i].oGranules[0].coef), color="black")

        #for x in granule.xs:
        #    ax.scatter(x[0], x[1], granule.oGranules[0].p(x), c=colors[i])

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Output')
    ax.set_xlim(min, max)
    ax.set_ylim(min, max)
    ax.set_zlim(min, max)

    return ax


def plot_granule_3d_space(granule, ax, i=1):
    """
    Plot granule in 3D space

    :param fbem_instance:
    :param min:
    :param max:
    :param indices:
    :return:
    """

    colors = ["red", "blue", "black", "gray", "green", "cyan", "yellow", "pink", "fuchsia", "darkgray"]
    colors += colors

    gran = granule.get_granule_for_3d_plotting()
    faces = granule.get_faces_for_3d_plotting()

    # plot vertices
    ax.scatter3D(gran[:, 0], gran[:, 1], gran[:, 2], c=colors[i])

    # plot sides
    face1 = Poly3DCollection(verts=faces, facecolors=colors[i], linewidths=1, edgecolors=colors[i], alpha=.1)
    face1.set_alpha(0.25)
    face1.set_facecolor(colors=colors[i])
    ax.add_collection3d(face1)

    ax.text(gran[0, 0], gran[0, 1], gran[0, 1], s="gamma " + str(i), color="black")

    for x in granule.xs:
        ax.scatter(x[0], x[1], granule.oGranules[0].p(x), c=colors[i])

    return ax

def plot_show():
    """
    Plot all the graphs previously prepared
    :return:
    """
    plt.show()

def create_3d_space():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Output')

    return ax


import pandas as pd
from sklearn.model_selection import train_test_split
from time import time, sleep
#from IPython.display import clear_output
from timeit import default_timer as timer
from river.metrics import Accuracy

n = 4

rules = []
acc = []
tempo_gasto = []

for i in range(0, n):

    fbi = FBeM()
    acc_r = Accuracy()
    fbi.debug = True
    to_normalize = 0

    data = pd.read_csv('https://bit.ly/iris_ds', header=None)
    data = data.sample(frac=1)

    # data = data.iloc[:-140,:]

    X = data.drop(columns=[4], axis=1)
    y = data[4].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

    # x_treino, x_teste, y_treino, y_teste = train_test_split(X, y)
    x_treino = X
    y_treino = y

    # Normalize data
    if to_normalize:
        fbi.rho = 1.
        fbi.hr = 20
        fbi.eta = 0.5


    else:
        fbi.rho = 1
        fbi.hr = 20
        fbi.eta = 0.5

    fbi.n = len(x_treino.columns)

    axis_1 = len(y_treino)
    axis_2 = len(x_treino)

    from timeit import default_timer as timer

    agora = timer()
    for i in range(0, axis_1):
        fbi.learn(x=x_treino.iloc[i], y=[y_treino.iloc[i]])
        print(i, y_treino.iloc[i], fbi.P[-1])
        acc_r.update(y_treino.iloc[i], fbi.P[-1])
        print(i, fbi.P)
    tempo_gasto.append(timer() - agora)
    print(f"Tempo gasto {timer() - agora}")

    # sleep(.75)
    # clear_output(wait=True)
    # plt.figure()
    # plot_granules(fbi)
    # plt.show()

    # axis_1 = len(y_teste)
    # axis_2 = len(x_teste)

    # for i in range(0, axis_1):
    #    fbi.learn(x=x_teste.iloc[i], y=[y_teste.iloc[i]])

    fbi.file.close()

    print("Final accuracy: ", fbi.acc[len(fbi.acc) - 1])
    acc.append(fbi.acc[len(fbi.acc) - 1])
    rules.append(np.mean(fbi.store_num_rules))

print(fbi.wrong)
print(fbi.right)
print("Rules: " + str(rules))
print("Acc: " + str(acc))
print(f"Acc {acc_r.get()}")
print(f"Acc: {np.mean(acc)} +/- {np.std(acc)}")
print(f"Tempo gasto {np.mean(tempo_gasto)} +/- {np.std(tempo_gasto)}")
print(fbi.granules[0].oGranules[0].coef)

plot_granules(fbi)
len(fbi.granules)