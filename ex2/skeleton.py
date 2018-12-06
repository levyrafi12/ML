#################################
# Your name:
#################################

import numpy as np
import matplotlib.pyplot as plt
from intervals import find_best_interval


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    """NEW_COMMENT 1"""
    def sample_y(self, x):
        """
        Calculate y accroding the distribution
        Input: x - single x sampling
        Returns: y - single y sampling 
        """
        p = np.random.uniform(low=0.0, high=1.0, size=1)
        if x <= 0.2 or 0.4 <= x and x <= 0.6 or x >= 0.8:
            y = 1 if p <= 0.8 else 0
        else: 
            y = 1 if p <= 0.1 else 0
        return y

    def overlap_penalty(self, I1, I2, p):
        """
        Calculate the overlapping penalty
        Input: I1, I2 - intervals
               p -  probability
        Returns: penalty - the overlapping interval multiplied by probabilty p
        """
        [L1, R1] = I1
        [L2, R2] = I2

        # non overlapping
        if R1 < L2 or R2 < L1:
            return 0
        res = (min(R1, R2) - max(L1, L2)) * p
        return res

    def calc_true_error(self, h):
        """
        Caclulate the true error (Ep(h))
        Input: h - the hypothesis

        Returns: Ep(h)
        """

        compl_h = []

        L2 = 0.0
        for [L1, R1] in h:
            R2 = L1
            if R2 > L2:
                compl_h.append([L2, R2])
            L2 = R1

        if L2 < 1.0:
            compl_h.append([L2, 1.0])
            
        best_h = [[0.0, 0.2], [0.4, 0.6], [0.8, 1.0]]
        compl_best_h = [[0.2, 0.4], [0.6, 0.8]]

        penalty = 0.0
        for I1 in h:
            for I2 in best_h:
                penalty += self.overlap_penalty(I1, I2, 0.2)
            for I2 in compl_best_h:
                penalty += self.overlap_penalty(I1, I2, 0.9)

        for I1 in compl_h:
            for I2 in best_h:
                penalty += self.overlap_penalty(I1, I2, 0.8)
            for I2 in compl_best_h:
                penalty += self.overlap_penalty(I1, I2, 0.1)

        return penalty

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        A = np.zeros((2, m), dtype=float)
        xs = np.random.uniform(low=0.0, high=1.0, size=m)
        xs = sorted(xs)
        A[0,:] = xs
        for i in range(0,len(xs)):
            x = xs[i]
            A[1, i] = self.sample_y(x)

        return A     

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """

        S = self.sample_from_D(m)

        points = np.random.uniform(low=0.0, high=1.0, size=2 * k)
        points = sorted(points)

        intervals, besterror = find_best_interval(S[0,:], S[1,:] ,k)
        for inter in intervals:
            plt.hlines(0.8, inter[0], inter[1], 'b', lw=2) # print horizontal line
        
        print("besterror = {}".format(besterror))

        plt.plot(S[0,:], S[1, :], 'ro')
        """i = 0
        while i < len(points):
            plt.hlines(0.8, points[i], points[i + 1], 'b', lw=2)
            intervals.append([points[i], points[i + 1]])
            i += 2
        """

        plt.axvline(x=0.2)
        plt.axvline(x=0.4)
        plt.axvline(x=0.6)
        plt.axvline(x=0.8)
        plt.axis([0, 1, -0.1, 1.1])
        plt.show()
        return

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        n_steps = int((m_last - m_first) / step + 1)

        E = np.zeros((n_steps, 2), dtype=float)

        for t in range(T):
            print("t {}".format(t))
            i = 0
            for m in range(m_first, m_last + step, step):
                S = self.sample_from_D(m)
                intervals, besterror = find_best_interval(S[0,:], S[1,:] ,k)
                E[i, 0] += (besterror / m)
                E[i, 1] += self.calc_true_error(intervals)
                i += 1

        for i in range(n_steps):
            for j in range(2):
                E[i, j] /= T

        m_vals = np.arange(m_first, m_last + step, step)
    
        plt.plot(m_vals, E[:, 0], 'r-', m_vals, E[:, 1], 'b--')
        plt.axis([0, m_last, 0, 1])
        plt.text(10, 0.8, 'red = Es')
        plt.text(10, 0.7, 'blue = Ep')
        plt.show()

        return E

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,20.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        # TODO: Implement the loop
        pass

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        # TODO: Implement the loop
        pass

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # TODO: Implement me
        pass

    #################################
    # Place for additional methods


    #################################


if __name__ == '__main__':
    ass = Assignment2()
    # ass.draw_sample_intervals(100, 3)
    # print(ass.calc_true_error([[0.1,0.3],[0.4,0.6],[0.8,1]]))
    ass.experiment_m_range_erm(10, 100, 5, 3, 10)
    """
    ass.experiment_k_range_erm(1500, 1, 20, 1)
    ass.experiment_k_range_srm(1500, 1, 20, 1)
    ass.cross_validation(1500, 3)
    """

