import numpy as np


class OutputGranule:

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
