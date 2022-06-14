import math

from efg.granules.granule import Granule
from efg.granules.input_granule import InputGranule
from efg.granules.output_granule import OutputGranule
from efg.utils.utils import sum_, sum_specific, min_u, max_U, power, sub, sum_dot
import numpy as np


# FBeM Class
class EFGClassifier:

    def __init__(self, n: int = 2, rho: float = .7, hr: int = 48, alpha: int = 0, eta: float = .5) -> object:
        """
        Initialization of the object
        :rtype: object
        :type rho: expansion size of granules.
        :type hr: period until evaluating granules. One hr is one iteration.
        """
        self.__acc = []
        self.__right = []
        self.__wrong = []
        self.__output = []

        self.data = []
        self.c = 0
        self.h = 0
        self.rho = rho
        self.n = n
        self.m = 1
        self.hr = hr
        self.alpha = alpha
        self.eta = eta
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
        # self.file = open("log.txt", "w")

        self.__classes = {}

    @property
    def numeric_class(self) -> dict:
        return self.__classes

    def __create_new_granule(self, index, x, y):
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
            new_granule = InputGranule()
            new_granule.l = x[i]
            new_granule.lambd = x[i]
            new_granule.Lambd = x[i]
            new_granule.L = x[i]

            g.input_granules.insert(i, new_granule)

        """ Output granules """
        for k in range(0, self.m):
            new_o_granule = OutputGranule()
            new_o_granule.u = y[k]
            new_o_granule.ups = y[k]
            new_o_granule.Ups = y[k]
            new_o_granule.U = y[k]

            """ Coefficients alpha """
            new_o_granule.coef = y[0]

            g.points.append(self.h)  #
            g.act = 0
            g.output_granules.insert(k, new_o_granule)

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

    def __create_check_imaginary_granule(self, granule1, granule2):
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
            new_granule = InputGranule()
            gran_1 = granule1.input_granules[i]
            gran_2 = granule2.input_granules[i]

            new_granule.l = min([gran_1.l, gran_2.l])
            new_granule.lambd = min([gran_1.lambd, gran_2.lambd])
            new_granule.Lambd = max([gran_1.Lambd, gran_2.Lambd])
            new_granule.L = max([gran_1.L, gran_2.L])

            if new_granule.midpoint() - self.rho / 2 <= new_granule.l and \
                    new_granule.midpoint() + self.rho / 2 >= new_granule.L:
                J = J + 1

            g.input_granules.insert(i, new_granule)

        """ Output granules """
        for k in range(0, self.m):
            new_o_granule = OutputGranule()
            gran_1 = granule1.output_granules[k]
            gran_2 = granule2.output_granules[k]

            if gran_1.coef == gran_2.coef:
                K = K + 1

            """ Coefficients alpha """
            g.points = granule1.points + granule2.points
            g.xs = granule1.xs + granule2.xs
            g.ys = granule1.ys + granule2.ys
            new_o_granule.coef = gran_2.coef
            g.act = 0

            new_o_granule.u = min([gran_1.u, gran_2.u])
            new_o_granule.ups = min([gran_1.ups, gran_2.ups])
            new_o_granule.Ups = max([gran_1.Ups, gran_2.Ups])
            new_o_granule.U = max([gran_1.U, gran_2.U])

            g.output_granules.insert(k, new_o_granule)

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

    def predict_one(self, x):
        # fix compatibility with RiverML
        x = list(x.values())

        if self.h == 0:
            return round(np.random.rand())
        else:

            I = []
            S = np.array([])
            for i in range(0, self.c):
                J = 0
                similarity = 0
                for j in range(0, self.n):
                    similarity += self.granules[i].input_granules[j].com_similarity(x=x[j])
                    if self.granules[i].input_granules[j].fits(xj=x[j], rho=self.rho):
                        J += 1

                S = np.insert(S, i, 1 - ((1 / (4 * self.n)) * similarity))
                if J == self.n:
                    """ granule i fits x! """
                    I.append(i)

            """ Check degree of similarity """
            if len(I) == 0: I = [np.argmax(S)]

            """ Prediction """
            return self.granules[I[0]].output_granules[0].coef

    def learn_one(self, x, y):
        """
        Learn as x and y enters
        :param x: observations
        :param y: expected output
        :return:
        """
        # fix compatibility with RiverML
        x = list(x.values())

        # converting text to int
        if type(y) is not int:
            if type(y) is bool:
                y = int(y)
            if y not in self.__classes:
                self.__classes[y] = len(self.__classes)
            y = self.__classes[y]

        # since efgp was built to also perform multiple output,
        # we need to transform y to an array
        y = [y]

        # starting from scratch
        if self.h == 0:

            self.ys.append(y[0])

            """ create new granule anyways """
            self.__create_new_granule(self.c, x, y)

            self.P.append(round(np.random.rand()))
            self.store_num_rules.insert(self.h, self.c)
            self.vec_rho.insert(self.h, self.rho)

            if y[0] == self.P[0]:
                self.__right.insert(self.h, 1)
                self.__wrong.insert(self.h, 0)

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
                self.__right.insert(self.h, 0)
                self.__wrong.insert(self.h, 1)

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

            # self.file.write(str(self.h) + "\t" + str(self.P[self.h]) + "\t1\n")

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
            Check rules that can accommodate x and respective similarities
            """
            I = []
            S = np.array([])
            for i in range(0, self.c):
                J = 0
                similarity = 0
                for j in range(0, self.n):
                    similarity += self.granules[i].input_granules[j].com_similarity(x=x[j])
                    if self.granules[i].input_granules[j].fits(xj=x[j], rho=self.rho):
                        J += 1

                S = np.insert(S, i, 1 - ((1 / (4 * self.n)) * similarity))
                if J == self.n:
                    """ granule i fits x! """
                    I.append(i)

            """ Check degree of similarity """
            # S = np.array([])
            # for i in range(0, self.c):
            #    aux = 0
            #    for j in range(0, self.n):
            #        aux += self.granules[i].input_granules[j].com_similarity(x=x[j])

            #    S.insert(i, 1 - ((1 / (4 * self.n)) * aux))

            # I = [S.index(max(S))]

            # mesclar esse for ao de cima.
            # for i in range(self.c):
            #    aux = sum([self.granules[i].input_granules[j].com_similarity(x=x[j]) for j in range(self.n)])

            #    S = np.insert(S, i, 1 - ((1 / (4 * self.n)) * aux))

            if len(I) == 0: I = [np.argmax(S)]

            # Vector for plot ROC
            # self.oClass1 = 0
            # self.oClass2 = 0

            # Otimizar e entender depois
            # for i in range(0, self.c):
            #    if self.granules[i].output_granules[0].coef == 0:
            #        self.oClass1 = max(self.oClass1, S[i])
            #    elif self.granules[i].output_granules[0].coef == 1:
            #        self.oClass2 = max(self.oClass2, S[i])
            #
            # output = []
            # output.append(self.oClass1 / (self.oClass1 + self.oClass2))
            # output.append(self.oClass2 / (self.oClass1 + self.oClass2))

            # self.output.insert(self.h, output)

            """ Prediction """
            self.P.insert(self.h, self.granules[I[0]].output_granules[0].coef)

            """ y becomes available // cumulative """
            if y[0] == self.P[len(self.P) - 1]:
                self.__right.insert(self.h, 1)
                self.__wrong.insert(self.h, 0)

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
                self.__right.insert(self.h, 0)
                self.__wrong.insert(self.h, 1)

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
            Check rules that accommodate x and similarities
            """

            I = []
            S = np.array([])
            for i in range(0, self.c):
                J = 0
                K = 0

                # optimization
                if self.m == 1 and not self.granules[i].output_granules[0].fits(y=y[0]):
                    continue

                similarity = 0
                for j in range(0, self.n):
                    similarity += self.granules[i].input_granules[j].com_similarity(x=x[j])
                    """ xj inside j-th expanded region? """
                    if self.granules[i].input_granules[j].fits(xj=x[j], rho=self.rho):
                        J += 1
                    # print(x, j, self.granules[i].input_granules[j].fits(xj=x[j], rho=self.rho), x[j], self.rho)
                # print('--- y')

                for k in range(0, self.m):
                    """ yk inside k-th expanded region? """
                    if self.granules[i].output_granules[k].fits(y=y[k]):
                        K += 1
                    # print(y, k, self.granules[i].output_granules[k].fits(y=y[k]), y[k])
                # print('---')
                if J + K == self.n + self.m:
                    I.append(i)
                    S = np.append(S, 1 - ((1 / (4 * self.n)) * similarity))

            """ Case 0: no granule fits x """
            if len(I) == 0:
                self.__create_new_granule(x=x, y=y, index=self.c)
                I.append(self.c - 1)
            else:
                """
                Adaptation of the most qualified granule
                If more than one granule fits the observation
                """
                if len(I) >= 2:
                    """
                    S = []
                    for i in range(0, len(I)):
                        aux = 0
                        for j in range(0, self.n):
                            aux += self.granules[I[i]].input_granules[j].com_similarity(x=x[j])

                        S.insert(I[i], 1 - ((1 / (4 * self.n)) * aux))
                    """
                    # print(S)
                    I = I[np.argmax(S)]
                else:
                    I = I[0]

                """ Adapting antecedent of granule I """
                """ Debugging """
                """
                if self.debug and stop:
                    ax = create_3d_space()
                    ax = plot_granule_3d_space(granule=self.granules[I], ax=ax, i=3)
                    ax.scatter(x[0], x[1], self.granules[I].output_granules[0].p(x=x))
                    ax.set_title("Teste")
                    plot_show()
                """
                """ /Debugging """
                for j in range(0, self.n):
                    ig = self.granules[I].input_granules[j]
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
                    ig = self.granules[I].input_granules[j]
                    mp = self.granules[I].input_granules[j].midpoint()

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
                    ax.scatter(x[0], x[1], self.granules[I].output_granules[0].p(x=x), c="green")
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
                While choosing two closest granules according to S
                """
                S = np.zeros((self.c, self.c))
                gra1 = []
                ind1 = -1
                gra2 = []
                ind2 = -1
                aux = -np.inf

                for i1 in range(0, self.c):
                    for i2 in range(i1 + 1, self.c):
                        # optimization
                        if self.granules[i1].output_granules[0].coef != \
                                self.granules[i1].output_granules[0].coef:
                            continue

                        S[i1][i2] = self.granules[i1].granule_similarity(self.granules[i2])
                        S[i1][i2] = 1 - ((1 / (4 * self.n)) * S[i1][i2])
                        if aux < S[i1][i2]:
                            aux = S[i1][i2]
                            gra1 = self.granules[i1]
                            gra2 = self.granules[i2]
                            ind1 = i1
                            ind2 = i2

                if gra1 != [] and gra2 != []:
                    res = self.__create_check_imaginary_granule(granule1=gra1, granule2=gra2)
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
        self.__acc.insert(self.h, sum(self.__right) / (sum(self.__right) + sum(self.__wrong)) * 100)
        self.h += 1

        """ Check if support contraction is needed """

        if self.c == 1:
            I = [0]

        if type(I) is int:
            I = [I]

        for i in I:
            if i >= self.c:
                continue

            for j in range(0, self.n):
                ig = self.granules[i].input_granules[j]
                mp = self.granules[i].input_granules[j].midpoint()

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


class EFGPredictor:
    def __init__(self, n: int = 2, rho: float = .7, hr: int = 48, alpha: int = 0, eta: float = .5):
        """
        Initialization of the object
        """
        self.data = []
        self.c = 0
        self.h = 0
        self.__rho = rho
        self.n = n
        self.m = 1
        self.hr = hr
        self.alpha = 0
        self.eta = eta
        self.counter = 1

        self.rmse = []
        self.ndei = []
        self.granules = []
        self.ys = []

        self.P = []
        self.PUB = []
        self.PLB = []
        self.store_num_rules = []
        self.vec_rho = []
        self.debug = False

        # debugging variables
        self.merged_granules = 0
        self.created_granules = 0
        self.deleted_granules = 0
        # self.file = open("log.txt", "w")

    def __create_new_granule(self, index, x, y):
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
            new_granule = InputGranule()
            new_granule.l = x[i]
            new_granule.lambd = x[i]
            new_granule.Lambd = x[i]
            new_granule.L = x[i]

            g.input_granules.insert(i, new_granule)

        """ Output granules """
        for k in range(0, self.m):
            new_o_granule = OutputGranule()
            new_o_granule.u = y[k]
            new_o_granule.ups = y[k]
            new_o_granule.Ups = y[k]
            new_o_granule.U = y[k]

            """ Coefficients alpha """
            new_o_granule.coef = []
            new_o_granule.coef.append(y[0])

            for i in range(0, self.n):
                new_o_granule.coef.append(0)

            g.points.append(self.h)  #
            g.act = 0
            g.output_granules.insert(k, new_o_granule)

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

    def __create_check_imaginary_granule(self, granule1, granule2):
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
            new_granule = InputGranule()
            gran_1 = granule1.input_granules[i]
            gran_2 = granule2.input_granules[i]

            new_granule.l = min([gran_1.l, gran_2.l])
            new_granule.lambd = min([gran_1.lambd, gran_2.lambd])
            new_granule.Lambd = max([gran_1.Lambd, gran_2.Lambd])
            new_granule.L = max([gran_1.L, gran_2.L])

            if new_granule.midpoint() - self.__rho / 2 <= new_granule.l and new_granule.midpoint() + self.__rho / 2 >= new_granule.L:
                J = J + 1

            g.input_granules.insert(i, new_granule)

        """ Output granules """
        for k in range(0, self.m):
            new_o_granule = OutputGranule()
            gran_1 = granule1.output_granules[k]
            gran_2 = granule2.output_granules[k]

            new_o_granule.u = min([gran_1.u, gran_2.u])
            new_o_granule.ups = min([gran_1.ups, gran_2.ups])
            new_o_granule.Ups = max([gran_1.Ups, gran_2.Ups])
            new_o_granule.U = max([gran_1.U, gran_2.U])

            if new_o_granule.midpoint() - self.__rho / 2 <= new_o_granule.u and new_o_granule.midpoint() + self.__rho / 2 >= new_o_granule.U:
                K = K + 1

            """ Coefficients alpha """
            g.points = granule1.points + granule2.points
            g.xs = granule1.xs + granule2.xs
            g.ys = granule1.ys + granule2.ys
            g.act = 0

            new_o_granule.coef = [(x + y) / 2 for x, y in
                                  zip(granule1.output_granules[k].coef, granule2.output_granules[k].coef)]

            g.output_granules.insert(k, new_o_granule)

        """ Check if new imaginary granule can become real """
        become_real = False
        if J + K == self.n + self.m:
            self.c = self.c + 1
            g.calculate_rls()

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

    def predict_one(self, x):
        x = list(x.values())
        if self.h == 0:
            return np.random.rand(), [0, 1]

        else:

            I = []
            for i in range(0, self.c):
                J = 0
                for j in range(0, self.n):
                    if self.granules[i].input_granules[j].fits(xj=x[j], rho=self.__rho):
                        J += 1

                if J == self.n:
                    """ granule i fits x! """
                    I.append(i)

            """ Check degree of similarity """
            S = []
            for i in range(0, self.c):
                aux = 0
                for j in range(0, self.n):
                    aux += self.granules[i].input_granules[j].com_similarity(x=x[j])

                S.insert(i, 1 - ((1 / (4 * self.n)) * aux))

            """ If no rule encloses x """
            if len(I) == 0:
                I = [S.index(max(S))]

            """ Calculate functional consequent """
            p = []
            for i in range(0, len(I)):
                for j in range(0, self.m):
                    p_calc = self.granules[I[i]].output_granules[j].p(x=x)
                    p.insert(i, p_calc)

            """ Prediction """
            part_1 = sum_dot(S, I, p)
            part_2 = sum_specific(S, I)
            part_3 = part_1 / part_2

            plb = min_u(granules=self.granules, indices=I, m=self.m)
            pub = max_U(granules=self.granules, indices=I, m=self.m)

            """ P must belong to [PLB PUB] """
            if part_3 < plb:
                part_3 = pub

            if part_3 > plb:
                part_3 = pub

            return part_3, [plb, pub]

    def learn_one(self, x, y):
        """
        Learn as x and y enters
        :param x: observations
        :param y: expected output
        :return:
        """
        x = list(x.values())

        # starting from scratch
        if self.h == 0:

            self.ys.append(y[0])

            """ create new granule anyways """
            self.__create_new_granule(self.c, x, y)

            self.P.append(np.random.rand())
            self.PUB.append(1)  # Upper bound
            self.PLB.append(0)  # Lower bound
            self.store_num_rules.insert(self.h, self.c)
            self.vec_rho.insert(self.h, self.__rho)

            # self.rmse.append(np.sqrt(sum_(power(sub(self.ys, self.P), 2)) / (self.h + 1)))
            self.rmse.append(np.sqrt(sum_(power(sub(self.ys, self.P), 2))))
            self.ndei.append(np.inf)
            # self.file.write(str(self.h) + "\t" + str(self.P[self.h]) + "\t1\n")

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
                    if self.granules[i].input_granules[j].fits(xj=x[j], rho=self.__rho):
                        J += 1

                if J == self.n:
                    """ granule i fits x! """
                    I.append(i)

            """ Check degree of similarity """
            S = []
            for i in range(0, self.c):
                aux = 0
                for j in range(0, self.n):
                    aux += self.granules[i].input_granules[j].com_similarity(x=x[j])

                S.insert(i, 1 - ((1 / (4 * self.n)) * aux))

            """ If no rule encloses x """
            if len(I) == 0:
                I = [S.index(max(S))]

            ## Store points in the granules
            # for i in I:
            #    self.granules[i].xs.append(x)
            #    self.granules[i].points.append(self.h)

            """ Calculate functional consequent """
            if self.h == 562:
                stop = 0

            p = []
            for i in range(0, len(I)):
                for j in range(0, self.m):
                    p_calc = self.granules[I[i]].output_granules[j].p(x=x)
                    p.insert(i, p_calc)
                    ## Store ypoints in the granules
                    # self.granules[I[i]].ys.append(p_calc)

            """ Prediction """
            part_1 = sum_dot(S, I, p)
            part_2 = sum_specific(S, I)
            part_3 = part_1 / part_2

            self.P.insert(self.h, part_3)
            self.PLB.insert(self.h, min_u(granules=self.granules, indices=I, m=self.m))
            self.PUB.insert(self.h, max_U(granules=self.granules, indices=I, m=self.m))

            # self.file.write(str(self.h) + '\t' + str(self.P[self.h]) + '\t' + str(I) + '\n')

            """ P must belong to [PLB PUB] """
            if self.P[self.h] < self.PLB[self.h]:
                self.P[self.h] = self.PLB[self.h]

            if self.P[self.h] > self.PUB[self.h]:
                self.P[self.h] = self.PUB[self.h]

            self.store_num_rules.insert(self.h, self.c)  # number of rules

            """ y becomes available // cumulative """
            self.ys.append(y[0])

            # RMSE
            part = sub(self.ys, self.P)
            part = power(part, 2)
            part = sum_(part)
            part = np.sqrt(part / (self.h + 1))

            self.rmse.append(part)
            self.ndei.append(self.rmse[self.h] / np.std(self.ys))

            # self.rmse.append(sqroot(sum_(power(sub(self.ys, self.P), 2)) / (self.h + 1)))
            # self.ndei.insert(self.h, self.rmse[self.h] / np.std(self.ys))

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
                    if self.granules[i].input_granules[j].fits(xj=x[j], rho=self.__rho):
                        J += 1

                for k in range(0, self.m):
                    """ yk inside k-th expanded region? """
                    if self.granules[i].output_granules[k].fits_(y=y[k], rho=self.__rho):
                        K += 1

                if J + K == self.n + self.m:
                    I.append(i)

            """ Case 0: no granule fits x """
            if len(I) == 0:
                self.__create_new_granule(x=x, y=y, index=self.c)
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
                            aux += self.granules[I[i]].input_granules[j].com_similarity(x=x[j])

                        S.insert(I[i], 1 - ((1 / (4 * self.n)) * aux))

                    I = I[S.index(max(S))]
                else:
                    I = I[0]

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
                    ig = self.granules[I].input_granules[j]
                    mp = ig.midpoint()
                    if mp - self.__rho / 2 < x[j] < ig.l:  # 1 ok
                        ig.l = x[j]  # Support expansion
                    if mp - self.__rho / 2 < x[j] < ig.lambd:  # 2 ok
                        ig.lambd = x[j]  # core expansion
                    if ig.lambd < x[j] < mp:  # 3 ok
                        ig.lambd = x[j]  # core contraction
                    if mp < x[j] < mp + self.__rho / 2:  # 4 ok
                        ig.lambd = mp  # core contraction
                    if mp - self.__rho / 2 < x[j] < mp:
                        ig.Lambd = mp  # Core contraction
                    if mp < x[j] < ig.Lambd:
                        ig.Lambd = x[j]  # core contraction
                    if ig.Lambd < x[j] < mp + self.__rho / 2:
                        ig.Lambd = x[j]  # core expansion
                    if ig.L < x[j] < mp + self.__rho / 2:
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
                    ig = self.granules[I].input_granules[j]
                    mp = self.granules[I].input_granules[j].midpoint()

                    """ Inferior support """
                    if mp - self.__rho / 2 > ig.l:
                        ig.l = mp - self.__rho / 2
                        if mp - self.__rho / 2 > ig.lambd:
                            ig.lambd = mp - self.__rho / 2

                    """ Superior Support """
                    if mp + self.__rho / 2 < ig.L:
                        ig.L = mp + self.__rho / 2
                        if mp + self.__rho / 2 < ig.Lambd:
                            ig.Lambd = mp + self.__rho / 2

                """ Adapting consequent granule I """
                for k in range(0, self.m):
                    og = self.granules[I].output_granules[k]
                    mp = self.granules[I].output_granules[k].midpoint()

                    if mp - self.__rho / 2 < y[k] < og.u:  # 1 ok
                        og.u = y[k]  # Support expansion
                    if mp - self.__rho / 2 < y[k] < og.ups:  # 2 ok
                        og.ups = y[k]  # core expansion
                    if og.ups < y[k] < mp:  # 3 ok
                        og.ups = y[k]  # core contraction
                    if mp < y[k] < mp + self.__rho / 2:  # 4 ok
                        og.ups = mp  # core contraction
                    if mp - self.__rho / 2 < y[k] < mp:
                        og.Ups = mp  # Core contraction
                    if mp < y[k] < og.Ups:
                        og.Ups = y[k]  # core contraction
                    if og.Ups < y[k] < mp + self.__rho / 2:
                        og.Ups = y[k]  # core expansion
                    if og.U < y[k] < mp + self.__rho / 2:
                        og.U = y[k]  # support expansion

                    """
                    if mp - self.rho / 2 < y[k] < og.u:
                        og.u = y[k]  # Support expansion
                    if og.u < y[k] < og.ups:
                        og.ups = y[k]  # Core expansion
                    if og.ups < y[k] < mp:
                        og.ups = y[k]  # Core contraction
                    if mp < y[k] < og.Ups:
                        og.Ups = y[k]  # Core contraction
                    if og.Ups < y[k] < og.U:
                        og.Ups = y[k]  # Core expansion
                    if og.U < y[k] < mp + self.rho / 2:
                        og.U = y[k]  # Support expansion
                    """
                """ Check if support contraction is needed """
                for k in range(0, self.m):
                    og = self.granules[I].output_granules[k]
                    mp = self.granules[I].output_granules[k].midpoint()

                    """ Inferior support """
                    if mp - self.__rho / 2 > og.u:
                        og.u = mp - self.__rho / 2
                        if mp - self.__rho / 2 > og.ups:
                            og.ups = mp - self.__rho / 2

                    """ Superior support """
                    if mp + self.__rho / 2 < og.U:
                        og.U = mp + self.__rho / 2
                        if mp + self.__rho / 2 < og.Ups:
                            og.Ups = mp + self.__rho / 2

                """ Storing sample """
                self.granules[I].points.append(self.h)
                self.granules[I].ys.append(y[0])
                self.granules[I].xs.append(x)
                self.granules[I].act = 0

                """ Least Squares """
                self.granules[I].calculate_rls()  # ate aqui

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

                res = self.__create_check_imaginary_granule(granule1=gra1, granule2=gra2)
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
                self.__rho *= (1 + diff / self.counter)  # increase rho
            else:
                self.__rho *= (1 - (diff + self.eta) / self.counter)  # decrease rho

            self.counter = 1

        self.vec_rho.insert(self.h, self.__rho)  # granules size along

        self.h += 1

        """ Check if support contraction is needed """
        for i in range(0, self.c):
            for j in range(0, self.n):

                ig = self.granules[i].input_granules[j]
                mp = self.granules[i].input_granules[j].midpoint()

                """ Inferior support """
                if mp - self.__rho / 2 > ig.l:
                    ig.l = mp - self.__rho / 2
                    if mp - self.__rho / 2 > ig.lambd:
                        ig.lambd = mp - self.__rho / 2

                """ Superior Support """
                if mp + self.__rho / 2 < ig.L:
                    ig.L = mp + self.__rho / 2
                    if mp + self.__rho / 2 < ig.Lambd:
                        ig.Lambd = mp + self.__rho / 2


class FBeM_MD:
    def __init__(self):
        """
        Initialization of the object
        """
        self.data = []
        self.c = 0
        self.h = 0
        self.rho = 0.7
        self.n = 2
        self.m = 1
        self.hr = 48
        self.alpha = 0
        self.eta = 0.5
        self.counter = 1

        self.mp_total = 0
        self.use_only_complete_obs_ls = 0
        self.use_single_imputation = 0
        self.update_all_output_granules = 0

        self.margin = 0
        self.interval = 0
        self.similarity = 0  # 0 for com, 1 for core
        self.use_convex_hull = 1
        self.threshold_gran = 0
        self.use_only_max_similarity_sing = 0

        self.rmse = []
        self.ndei = []
        self.granules = []
        self.ys = []

        self.P = []
        self.PUB = []
        self.PLB = []
        self.store_num_rules = []
        self.vec_rho = []
        self.debug = False

        # debugging variables
        self.merged_granules = 0
        self.created_granules = 0
        self.deleted_granules = 0
        # self.file = open("log.txt", "w")

    def is_there_missing_data(self, x):
        """
        Check if there's missing data
        :param x:
        :return:
        """
        return any(e is None for e in x)

    def amount_missing_values(self, x):
        """
        Return amount of missing values
        :param x:
        :return:
        """

        return sum(e is None for e in x)

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
            new_granule = InputGranule()
            new_granule.l = x[i]
            new_granule.lambd = x[i]
            new_granule.Lambd = x[i]
            new_granule.L = x[i]

            """ Test with interval """
            # new_granule.l = new_granule.midpoint() - self.rho / 2
            # new_granule.lambd = new_granule.midpoint() - self.rho / 2
            # new_granule.Lambd = new_granule.midpoint() + self.rho / 2
            # new_granule.L = new_granule.midpoint() + self.rho / 2
            """ / """

            g.input_granules.insert(i, new_granule)

        """ Output granules """
        for k in range(0, self.m):
            new_o_granule = OutputGranule()
            new_o_granule.u = y[k] - self.margin
            new_o_granule.ups = y[k]
            new_o_granule.Ups = y[k]
            new_o_granule.U = y[k] + self.margin

            """ Test with interval """
            # new_o_granule.u = new_o_granule.midpoint() - self.rho / 2
            # new_o_granule.ups = new_o_granule.midpoint() - self.rho / 2
            # new_o_granule.Ups = new_o_granule.midpoint() + self.rho / 2
            # new_o_granule.U = new_o_granule.midpoint() + self.rho / 2
            """ / """

            """ Coefficients alpha """
            new_o_granule.coef = []
            new_o_granule.coef.append(y[0])

            for i in range(0, self.n):
                new_o_granule.coef.append(0)

            """ Coefficients beta """
            new_o_granule.coef_inc = []

            for i in range(0, self.n):
                new_o_granule.coef_inc.insert(i, [])
                new_o_granule.coef_inc[i].append(y[0])
                for j in range(1, self.n):
                    new_o_granule.coef_inc[i].append(0)

            g.points.append(self.h)  #
            g.act = 0
            g.output_granules.insert(k, new_o_granule)

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
            new_granule = InputGranule()
            gran_1 = granule1.iGranules[i]
            gran_2 = granule2.iGranules[i]

            new_granule.l = min([gran_1.l, gran_2.l])
            new_granule.lambd = min([gran_1.lambd, gran_2.lambd])
            new_granule.Lambd = max([gran_1.Lambd, gran_2.Lambd])
            new_granule.L = max([gran_1.L, gran_2.L])

            if new_granule.midpoint(self.mp_total) - self.rho / 2 <= new_granule.l and new_granule.midpoint(
                    self.mp_total) + self.rho / 2 >= new_granule.L:
                J = J + 1

            g.input_granules.insert(i, new_granule)

        """ Output granules """
        for k in range(0, self.m):
            new_o_granule = OutputGranule()
            gran_1 = granule1.oGranules[k]
            gran_2 = granule2.oGranules[k]

            new_o_granule.u = min([gran_1.u, gran_2.u])
            new_o_granule.ups = min([gran_1.ups, gran_2.ups])
            new_o_granule.Ups = max([gran_1.Ups, gran_2.Ups])
            new_o_granule.U = max([gran_1.U, gran_2.U])

            if new_o_granule.midpoint(self.mp_total) - self.rho / 2 <= new_o_granule.u and new_o_granule.midpoint(
                    self.mp_total) + self.rho / 2 >= new_o_granule.U:
                K = K + 1

            """ Coefficients alpha """
            g.points = granule1.points + granule2.points
            g.xs = granule1.xs + granule2.xs
            g.ys = granule1.ys + granule2.ys
            g.act = 0

            new_o_granule.coef = [(x + y) / 2 for x, y in zip(granule1.oGranules[k].coef, granule2.oGranules[k].coef)]

            """ /Coefficients alpha """
            """ Coefficients beta """
            new_o_granule.coef_inc = []
            for i in range(0, self.n):
                new_o_granule.coef_inc.insert(i, [(x + y) / 2 for x, y in zip(granule1.oGranules[k].coef_inc[i],
                                                                              granule2.oGranules[k].coef_inc[i])])
            """ /Coefficients beta """

            g.output_granules.insert(k, new_o_granule)

        """ Check if new imaginary granule can become real """
        become_real = False
        if J + K == self.n + self.m:
            self.c = self.c + 1
            # g.calculate_rls()

            """ Least squares for functional consequent with one term less MD """
            # g.calculate_rls_q()
            """ /Least squares for functional consequent with one term less MD """

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

        """ Check for missing data """
        missing_data = self.is_there_missing_data(x=x)
        qtd_missing_data = self.amount_missing_values(x=x)
        """ /check for missing data """

        """ Check for starting from scratch """
        """ It HAS TO START with complete information """

        if (self.h == 0 or self.c == 0) and missing_data:
            self.ys.append(y[0])
            self.P.insert(self.h, np.random.rand())
            self.PLB.insert(self.h, 0)
            self.PUB.insert(self.h, 1)
            self.store_num_rules.insert(self.h, self.c)
            self.vec_rho.insert(self.h, self.rho)

            if self.h == 0:
                div = 1
            else:
                div = self.h

            self.rmse.append(np.sqrt(np.sum((np.array(self.ys) - np.array(self.P)) ** div)))
            self.ndei.append(np.inf)
            # self.file.write(str(self.h) + "\t" + str(self.P[self.h]) + "\t1\n")

            self.h += 1
            return

        """ Check if the whole information is missing """
        """ If everything is none, observation is discarded """
        """ Previous behaviour is replicated """

        if sum(e is None for e in x) == self.n:
            self.ys.append(y[0])
            self.P.insert(self.h, self.ys[self.h - 1])
            self.PLB.insert(self.h, self.PLB[self.h - 1])
            self.PUB.insert(self.h, self.PUB[self.h - 1])
            self.store_num_rules.insert(self.h, self.c)
            self.vec_rho.insert(self.h, self.rho)

            self.rmse.append(np.sqrt(np.sum((np.array(self.ys) - np.array(self.P)) ** self.h)))
            self.ndei.append(np.inf)
            # self.file.write(str(self.h) + "\t" + str(self.P[self.h]) + "\t1\n")
            self.h += 1
            return

        # starting from scratch
        if self.c == 0:

            self.ys.append(y[0])

            """ create new granule anyways """
            self.create_new_granule(self.c, x, y)

            self.P.append(np.random.rand())
            self.PUB.append(1)  # Upper bound
            self.PLB.append(0)  # Lower bound
            self.store_num_rules.insert(self.h, self.c)
            self.vec_rho.insert(self.h, self.rho)

            # self.rmse.append(np.sqrt(sum_(power(sub(self.ys, self.P), 2)) / (self.h + 1)))
            self.rmse.append(np.sqrt(np.sum((np.array(self.ys) - np.array(self.P)) ** 2)))
            self.ndei.append(np.inf)
            # self.file.write(str(self.h) + "\t" + str(self.P[self.h]) + "\t1\n")

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
                    """ """
                    """ Missing data """
                    """ Emulate partial distance """
                    """ """
                    if x[j] is not None:
                        """ """
                        """ /Missing data """
                        """ """
                        if self.granules[i].iGranules[j].fits(xj=x[j], rho=self.rho):
                            J += 1

                if J == self.n - qtd_missing_data:
                    """ granule i fits x! """
                    I.append(i)

            """ Check degree of similarity """
            S = []
            for i in range(0, self.c):
                aux = 0
                for j in range(0, self.n):
                    """ """
                    """ Missing data """
                    """ Emulate partial distance """
                    """ """
                    if x[j] is not None:
                        """ """
                        """ /Missing data """
                        """ """
                        if self.similarity == 0:
                            aux += self.granules[i].iGranules[j].com_similarity(x=x[j])
                        elif self.similarity == 1:
                            aux += self.granules[i].iGranules[j].core_similarity(x=x[j])

                """ Adaptation for missing data """
                """ """
                dividend = (4 - self.amount_missing_values(x))
                if not dividend: dividend = 4
                S.insert(i, 1 - ((1 / (dividend * self.n)) * aux))
                # S.insert(i, 1 - ((1 / (4 * self.n)) * aux))
                """ """
                """ /Adaptation for missing data """

            """ If no rule encloses x """
            if len(I) == 0:
                I = [S.index(max(S))]

            """ Missing data checking for the calculation of functional consequent """
            """ """
            """ """
            p_output = 0
            if not missing_data:
                """ Calculate functional consequent """

                p = []
                Im = copy.deepcopy(I)
                if self.use_only_max_similarity_sing: Im = [S.index(max(S))]

                for i in range(0, len(Im)):
                    for j in range(0, self.m):
                        p_calc = self.granules[Im[i]].oGranules[j].p(x=x)
                        p.insert(i, p_calc)

                part_1 = sum_dot(S, Im, p)
                part_2 = sum_specific(S, Im)
                p_output = part_1 / part_2

                if math.isnan(p_output):
                    part = np.inf

            elif self.use_single_imputation == 0 and missing_data and qtd_missing_data == 1:
                """ Calculate functional consequent with one term less """
                Im = copy.deepcopy(I)
                if self.use_only_max_similarity_sing: Im = [S.index(max(S))]

                q = []
                for i in range(0, len(Im)):
                    for j in range(0, self.m):
                        q_calc = self.granules[Im[i]].oGranules[j].q(x=x, missing_data_index=x.index(None))
                        q.insert(i, q_calc)

                part_1 = sum_dot(S, Im, q)
                part_2 = sum_specific(S, Im)
                if part_1 > part_2:
                    stop = 0
                p_output = part_1 / part_2

            elif self.use_single_imputation == 1 or (missing_data and qtd_missing_data > 1):
                """ Multiple missing data imputation """
                rule = [S.index(max(S))]  # Most activated rule
                for j in range(0, len(x)):
                    if x[j] is None:
                        x[j] = self.granules[rule[0]].iGranules[j].midpoint(self.mp_total)

                p = []
                for j in range(0, self.m):
                    p_calc = self.granules[rule[0]].oGranules[j].p(x=x)
                    p.insert(j, p_calc)

                part_1 = sum_dot(S, rule, p)
                part_2 = sum_specific(S, rule)
                if part_1 > part_2:
                    stop = 0
                p_output = part_1 / part_2
                missing_data = False

            """ """
            """ Prediction """

            if not self.use_convex_hull:
                if self.threshold_gran > 0 and len(S) > 1:
                    I = [k for k, e in enumerate(S) if e > self.threshold_gran][0:5]
                else:
                    I = [S.index(max(S))]

            self.P.insert(self.h, p_output)
            self.PLB.insert(self.h, min_u(granules=self.granules, indices=I, m=self.m))
            self.PUB.insert(self.h, max_U(granules=self.granules, indices=I, m=self.m))

            # self.file.write(str(self.h) + '\t' + str(self.P[self.h]) + '\t' + str(I) + '\n')

            """ P must belong to [PLB PUB] """
            if self.P[self.h] < self.PLB[self.h]:
                self.P[self.h] = self.PLB[self.h]
                # self.P[self.h] = self.P[self.h-1]

            if self.P[self.h] > self.PUB[self.h]:
                self.P[self.h] = self.PUB[self.h]
                # self.P[self.h] = self.P[self.h - 1]
                # self.P[self.h] = self.PUB[self.h] - abs(self.PUB[self.h] - self.P[self.h]) * 0.6

            # if self.P[self.h] - self.P[self.h - 1] > .7 or self.P[self.h - 1] - self.P[self.h] > .7:
            #    self.P[self.h] = self.P[self.h - 1]
            self.store_num_rules.insert(self.h, self.c)  # number of rules

            """ y becomes available // cumulative """
            self.ys.append(y[0])

            # RMSE
            part = sub(self.ys, self.P)
            part = power(part, 2)
            part = sum_(part)
            part = np.sqrt(part / (self.h + 1))

            self.rmse.append(part)
            if np.std(self.ys) != 0:
                self.ndei.append(self.rmse[self.h] / np.std(self.ys))
            else:
                self.ndei.append(np.inf)

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
                    """ Missing data """
                    if x[j] is None:
                        continue
                    """ /Missing data """
                    """ xj inside j-th expanded region? """
                    if self.granules[i].iGranules[j].fits(xj=x[j], rho=self.rho):
                        J += 1

                for k in range(0, self.m):
                    """ yk inside k-th expanded region? """
                    if self.granules[i].oGranules[k].fits(y=y[k], rho=self.rho):
                        K += 1

                if J + K == self.n + self.m - qtd_missing_data:
                    I.append(i)

            """ Case 0: no granule fits x """
            if len(I) == 0:
                """ Missing data """
                if not missing_data:
                    """ /Missing data """
                    self.create_new_granule(x=x, y=y, index=self.c)
            else:
                """
                Adaptation of the most qualified granule
                If more than one granule fits the observation
                """
                if not self.update_all_output_granules:
                    if len(I) >= 2:
                        S = []
                        for i in I:  # range(0, len(I)):
                            aux = 0
                            for j in range(0, self.n):
                                """ """
                                """ Missing data """
                                """ Emulate partial distance """
                                """ """
                                if x[j] is not None:
                                    """ """
                                    """ /Missing data """
                                    """ """
                                    if self.similarity == 0:
                                        aux += self.granules[i].iGranules[j].com_similarity(x=x[j])
                                    elif self.similarity == 1:
                                        aux += self.granules[i].iGranules[j].core_similarity(x=x[j])
                            # S.insert(i, 1 - ((1 / (4 * self.n)) * aux))
                            dividend = 4  # (4 - qtd_missing_data)
                            if not dividend: dividend = 4
                            S.insert(i, 1 - ((1 / (dividend * self.n)) * aux))

                        I = [I[S.index(max(S))]]
                    else:
                        I = [I[0]]

                o_granules = copy.deepcopy(I)

                for I in o_granules:

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
                        """ Missing data """
                        if x[j] is None: continue
                        """ /Missing data """

                        ig = self.granules[I].iGranules[j]
                        mp = ig.midpoint(self.mp_total)

                        """""
                        if mp - self.rho / 2 < x[j] < ig.l: ig.l = x[j]  # Support expansion
                        if mp - self.rho / 2 < x[j] < ig.lambd: ig.lambd = x[j]  # core expansion
                        if ig.lambd < x[j] < mp: ig.lambd = x[j]  # Core contraction
                        if mp < x[j] < mp + self.rho / 2: ig.lambd = mp ##
                        if mp - self.rho / 2 < x[j] < mp : ig.Lambd = mp  ##
                        if mp < x[j] < ig.Lambd: ig.Lambd = x[j] ##
                        if ig.Lambd < x[j] < mp + self.rho / 2: ig.Lambd = x[j]
                        if ig.L < x[j] < mp + self.rho / 2: ig.L = x[j]
                        """
                        """ Antes """

                        if mp - self.rho / 2 < x[j] < ig.l:
                            ig.l = x[j]  # Support expansion

                        if self.interval:
                            if mp - self.rho / 2 < x[j] < ig.lambd:  # 2 ok
                                ig.lambd = x[j]  # core expansion

                        if ig.l < x[j] < ig.lambd:
                            ig.lambd = x[j]  # Core expansion
                        if ig.lambd < x[j] < mp:
                            ig.lambd = x[j]  # Core contraction
                        if mp < x[j] < ig.Lambd:
                            ig.Lambd = x[j]  # Core contraction
                        if ig.Lambd < x[j] < ig.L:
                            ig.Lambd = x[j]  # Core expansion

                        if self.interval:
                            if ig.Lambd < x[j] < mp + self.rho / 2:
                                ig.Lambd = x[j]  # core expansion

                        if ig.L < x[j] < mp + self.rho / 2:
                            ig.L = x[j]  # Support expansion
                        """

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
                    """ Check if support contraction is needed """
                    for j in range(0, self.n):
                        if x[j] is not None:
                            ig = self.granules[I].iGranules[j]
                            mp = self.granules[I].iGranules[j].midpoint(self.mp_total)

                            # Inferior support
                            if mp - self.rho / 2 > ig.l:
                                ig.l = mp - self.rho / 2
                                if mp - self.rho / 2 > ig.lambd:
                                    ig.lambd = mp - self.rho / 2

                            # Superior Support
                            if mp + self.rho / 2 < ig.L:
                                ig.L = mp + self.rho / 2
                                if mp + self.rho / 2 < ig.Lambd:
                                    ig.Lambd = mp + self.rho / 2

                    """ Adapting consequent granule I """
                    for k in range(0, self.m):
                        og = self.granules[I].oGranules[k]
                        mp = self.granules[I].oGranules[k].midpoint(self.mp_total)

                        """
                        if mp - self.rho / 2 < y[k] < og.u:  # 1 ok
                            og.u = y[k]  # Support expansion
                        if mp - self.rho / 2 < y[k] < og.ups:  # 2 ok
                            og.ups = y[k]  # core expansion
                        if og.ups < y[k] < mp:  # 3 ok
                            og.ups = y[k]  # core contraction
                        if mp < y[k] < mp + self.rho / 2:  # 4 ok
                            og.ups = mp  # core contraction
                        if mp - self.rho / 2 < y[k] < mp:
                            og.Ups = mp  # Core contraction
                        if mp < y[k] < og.Ups:
                            og.Ups = y[k]  # core contraction
                        if og.Ups < y[k] < mp + self.rho / 2:
                            og.Ups = y[k]  # core expansion
                        if og.U < y[k] < mp + self.rho / 2:
                            og.U = y[k]  # support expansion

                        """
                        if len(self.ys) >= 700:
                            stop = 0
                        if mp - self.rho / 2 < y[k] < og.u:
                            og.u = y[k] - self.margin  # Support expansion

                        if self.interval:
                            if mp - self.rho / 2 < y[k] < og.ups:  # 2 ok
                                og.ups = y[k]  # core expansion

                        if og.u < y[k] < og.ups:
                            og.ups = y[k]  # Core expansion
                        if og.ups < y[k] < mp:
                            og.ups = y[k]  # Core contraction
                        if mp < y[k] < og.Ups:
                            og.Ups = y[k]  # Core contraction
                        if og.Ups < y[k] < og.U:
                            og.Ups = y[k]  # Core expansion

                        if self.interval:
                            if og.Ups < y[k] < mp + self.rho / 2:
                                og.Ups = y[k]  # core expansion

                        if og.U < y[k] < mp + self.rho / 2:
                            og.U = y[k] + self.margin  # Support expansion

                        """
                        if mp - self.rho / 2 < y[k] < og.u: og.u = y[k]  # Support expansion
                        if mp - self.rho / 2 < y[k] < og.ups: og.ups = y[k]  # core expansion
                        if og.ups < y[k] < mp: og.ups = y[k]  # Core contraction
                        if mp < y[k] < mp + self.rho / 2: og.ups = mp ##
                        if mp - self.rho / 2 < y[k] < mp : og.Ups = mp  ##
                        if mp < y[k] < og.Ups: og.Ups = y[k] ##
                        if og.Ups < y[k] < mp + self.rho / 2: og.Ups = y[k]
                        if og.U < y[k] < mp + self.rho / 2: og.U = y[k]
                        """

                    """ Check if support contraction is needed """

                    for k in range(0, self.m):
                        og = self.granules[I].oGranules[k]
                        mp = self.granules[I].oGranules[k].midpoint(self.mp_total)

                        # Inferior support
                        if mp - self.rho / 2 > og.u:
                            og.u = mp - self.rho / 2
                            if mp - self.rho / 2 > og.ups:
                                og.ups = mp - self.rho / 2

                        # Superior support
                        if mp + self.rho / 2 < og.U:
                            og.U = mp + self.rho / 2
                            if mp + self.rho / 2 < og.Ups:
                                og.Ups = mp + self.rho / 2

                    """ Storing sample """
                    self.granules[I].points.append(self.h)
                    self.granules[I].act = 0

                    # if not self.is_there_missing_data(x):

                    if self.use_only_complete_obs_ls:
                        if not self.is_there_missing_data(x):
                            self.granules[I].ys.append(y[0])
                            self.granules[I].xs.append(x)

                        """ Least Squares """
                        self.granules[I].calculate_rls_com()

                        """ Least Squares for functional consequent with a term less """
                        self.granules[I].calculate_rls_q_com()
                    else:
                        self.granules[I].ys.append(y[0])
                        self.granules[I].xs.append(x)

                        """ Least Squares """
                        self.granules[I].calculate_rls()

                        """ Least Squares for functional consequent with a term less """
                        self.granules[I].calculate_rls_q()

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
        for K in range(0, self.c):
            self.granules[K].act += 1
            if self.granules[K].act >= self.hr and self.c > 1:
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

        self.h += 1

        """ Check if support contraction is needed """
        for i in range(0, self.c):
            for j in range(0, self.n):

                ig = self.granules[i].iGranules[j]
                mp = self.granules[i].iGranules[j].midpoint(self.mp_total)

                # Inferior support
                if mp - self.rho / 2 > ig.l:
                    ig.l = mp - self.rho / 2
                    if mp - self.rho / 2 > ig.lambd:
                        ig.lambd = mp - self.rho / 2

                # Superior Support
                if mp + self.rho / 2 < ig.L:
                    ig.L = mp + self.rho / 2
                    if mp + self.rho / 2 < ig.Lambd:
                        ig.Lambd = mp + self.rho / 2
