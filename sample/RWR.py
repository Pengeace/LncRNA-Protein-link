import numpy as np
from numpy.linalg import norm


class RWR:

    def __init__(self, W, seeds, r=0.15, max_iter=2000, epsilon=1e-7):
        """
        Using the below iteration to computer p until the L1 norm between p^{n} and p^{n+1} is less than epsilon.
        p^{n+1} = (1-r)Wp^{n} + rp^{0}.

        :param W: numpy matrix, element W_{i,j} denotes whether node i is linked to node j
        :param seeds: a list of initial nodes with p^{0}_{k} = 1/len(seeds)
        :param r: the probability of jumping to the initial state
        :param max_iter: maximal iteration times
        :param epsilon: convergence criterion
        """

        self.W = W
        self.seeds = seeds
        self.r = r
        self.max_iter = max_iter
        self.epsilon = epsilon

        self.n = len(W)
        self.p = np.zeros([self.n])
        for s in seeds:
            self.p[s] = 1.0 / len(seeds)

    def compute(self):
        p_0 = self.p
        p = self.p
        p_next = self.p

        for i in range(self.max_iter):
            p_next = (1 - self.r) * (self.W.dot(p)) + self.r * p_0
            if norm(p_next - p, 1) < self.epsilon:
                # print("The vector p is convergent after %d iterations." % (i+1))
                break
            p, p_next = p_next, p

        return p_next
