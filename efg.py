from input_granule import InputGranule
from output_granule import OutputGranule
from granule import Granule
import numpy as np
import copy


# FBeM Class
class EFGClassifier:
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
            newGranule = InputGranule()
            newGranule.l = x[i]
            newGranule.lambd = x[i]
            newGranule.Lambd = x[i]
            newGranule.L = x[i]

            g.input_granules.insert(i, newGranule)

        """ Output granules """
        for k in range(0, self.m):
            newOGranule = OutputGranule()
            newOGranule.u = y[k]
            newOGranule.ups = y[k]
            newOGranule.Ups = y[k]
            newOGranule.U = y[k]

            """ Coefficients alpha """
            newOGranule.coef = y[0]

            g.points.append(self.h)  #
            g.act = 0
            g.output_granules.insert(k, newOGranule)

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
            newOGranule = OutputGranule()
            gran_1 = granule1.output_granules[k]
            gran_2 = granule2.output_granules[k]

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

            g.output_granules.insert(k, newOGranule)

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
                    if self.granules[i].input_granules[j].fits(xj=x[j], rho=self.rho):
                        J += 1

                if J == self.n:
                    """ granule i fits x! """
                    I.append(i)

            """ Check degree of similarity """
            S = np.array([])
            # for i in range(0, self.c):
            #    aux = 0
            #    for j in range(0, self.n):
            #        aux += self.granules[i].input_granules[j].com_similarity(x=x[j])

            #    S.insert(i, 1 - ((1 / (4 * self.n)) * aux))

            # I = [S.index(max(S))]

            # mesclar esse for ao de cima.
            for i in range(self.c):
                aux = sum([self.granules[i].input_granules[j].com_similarity(x=x[j]) for j in range(self.n)])

                S = np.insert(S, i, 1 - ((1 / (4 * self.n)) * aux))

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

                    # print(S)
                    I = I[S.index(max(S))]
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



