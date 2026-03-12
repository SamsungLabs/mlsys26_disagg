#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================
from scipy.stats import hypergeom
import math
import numpy as np
from functools import lru_cache


class Committee():
    '''
    Class for calculating the minimum committee size for security and correctness
    '''
    def __init__(self, N, sigma, eta, gamma, delta, rho = 1, BFT = False):
        '''
        Argumnets:
        N (int): Total number of clients
        sigma (float): Security parameter
        eta (float): Correctness parameter
        gamma (float): Fraction of corrupt clients
        delta (float): Fraction of dropout clients
        rho (int): The desired rho value for the secret length, it must be > 1
        BFT (bool): Use calculations with Byzantine Fault Tolerance

        Returns:
        The tuple (A_min, t_c, t_r, rho_max)
        A_min: The minimum committee size < N
        t_c: The bounded corrupted committee members (aggregators)
        t_r: The reconstruction threshold for committee members (aggregators)
        rho_max: The maximum found value <= rho (input)
        '''
        self.N = N
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.delta = delta
        self.rho = rho
        self.BFT = BFT

    def search_t_range(self, k):

        # start with initial threshold values (t) according to the two distributions
        # left-side for corrupt committee members, right-side for surviving ones
        # the minimum t (t_left_c) is larger than the center of the left distr. close to the right tail
        # and the maximum t (t_right_s) is smaller than the center of the right distr. close to the left tail
        # initial t values are away from the centers by a fraction of the half-width of distr.
        # start by a large fraction 95% and refine to smaller, so initially the prob thresholds are not reached

        percs = range(95, 0, -5)
        for perc in percs:

            fraction = perc / 100

            # init values for the thresholds, t_left_c: min t, t_right_s: max t
            t_left_c = int((1 + fraction) * self.gamma * k)        # corrupt committee members
            t_right_s = int((1 - self.delta * (1 + fraction) )*k)   # surviving committee members

            # initial probabilities according to the left-right t min-max values
            c_prob = 1 - hypergeom.cdf(t_left_c - 1, self.N, int(self.gamma*self.N), k)
            s_prob = hypergeom.cdf(t_right_s - 1, self.N, int((1-self.delta)*self.N), k)
            
            # we do not want from the start to meet the prob. condition
            # this will be found below with more accuracy
            # continue until there are large enough tails left for fine-search
            if c_prob >= self.p_c or s_prob >= self.p_d:
                break

        if t_left_c >= t_right_s:     # no space between distributions
            return None

        if k <= t_right_s:            # not large enough committee size
            return None

        # search minimum t (from corrupts)
        t_c = None
        for t in range(t_left_c, t_right_s):
            # calc the probability
            c_prob = 1 - hypergeom.cdf(t - 1, self.N, int(self.gamma*self.N), k)
            if c_prob < self.p_c:
                t_c = t
                break
        if t_c is None:
            return None

        # search maximum t (from dropouts)
        t_s = None
        for t in range(t_right_s, t_left_c, -1):
            # calc the probability
            s_prob = hypergeom.cdf(t - 1, self.N, int((1-self.delta)*self.N), k)
            if s_prob < self.p_d:
                t_s = t
                break
        if t_s is None:
            return None

        # check that there is space between min-max
        if t_c <= t_s:
            return (t_c, t_s)
    
        return None

    def search_t(self, k, reverse=False):

        t = self.search_t_range(k)

        if t is None:
            return None
        
        if not reverse:
            t = t[0]
        else:
            t = t[1]

        return t

    def binary_search_k(self, start_k, end_k, condition_func, reverse=False):

        left, right = start_k, end_k
        first_valid_k = None
        first_valid_t = None

        while left <= right:
            mid = (left + right) // 2
            t = self.search_t(mid, reverse=reverse)

            if t is not None and condition_func(t):
                # Found a valid k, search for smaller ones
                first_valid_k = mid
                first_valid_t = t
                right = mid - 1
            else:
                # Need larger k
                left = mid + 1

        if first_valid_k is not None:
            search_start = max(start_k, left - 2)
            search_end = min(end_k, first_valid_k)
            for check_k in range(search_start, search_end + 1):
                check_t = self.search_t(check_k, reverse=reverse)
                if check_t is not None and condition_func(check_t):
                    first_valid_k = check_k
                    first_valid_t = check_t
                    break

        return first_valid_k, first_valid_t

    def get_A_start_BFT(self):
        # print('Calculating minimum committee size for BFT...')
        k = self.gamma + self.delta
        if k >= 1 / 3:
            raise ValueError("Wrong corrupt-dropout fractions!")

        # Binary search for minimum A where BFT condition is satisfied
        left, right = 1, self.N - 1
        found_A = None

        while left <= right:
            mid = (left + right) // 2
            bft_tail_probability = get_bft_prob(self.N, mid, self.gamma, self.delta)

            if bft_tail_probability <= self.p_c:
                # Valid, search for smaller A
                found_A = mid
                right = mid - 1
            else:
                # Need larger A
                left = mid + 1

        if found_A is not None:
            # Small exhaustive search
            search_start = max(1, left - 3)
            for a in range(search_start, found_A + 1):
                bft_tail_probability = get_bft_prob(self.N, a, self.gamma, self.delta)
                if bft_tail_probability <= self.p_c:
                    A_start = a
                    break

            if A_start == 1:
                A_start = found_A
        
        return A_start

    @lru_cache(maxsize=4096)
    def get_committee_size(self):
        '''
        Function to calculate the minimum committee size for security and correctness
        '''

        self.p_c = 2**(-self.sigma)   # probability threshold for security
        self.p_d = 2**(-self.eta)     # probability threshold for correctness

        if self.rho <= 0 or self.rho >= self.N:
            raise ValueError('Wrong RHO value!')

        A_start = 1

        if self.BFT:
            A_start = self.get_A_start_BFT()

        # search for k,t (committee size, thresholds)
        # First binary search: find the first valid k where t is not None
        first_valid_k, first_valid_t = self.binary_search_k(
            A_start, self.N - 1,
            lambda _: True,
            reverse=False
        )

        if first_valid_k is None:
            return None, None, None, None

        k = first_valid_k
        t_c = first_valid_t

        # extend starting A by rho to speed up the search
        A_tmp = min(k + self.rho, self.N-1)

        # Second binary search: find the first k where t >= t_c + rho
        first_valid_k, first_valid_t = self.binary_search_k(
            A_tmp, self.N - 1,
            lambda t: t >= (t_c + self.rho),
            reverse=True
        )

        if first_valid_k is None:
            return None, None, None, None

        t_r = first_valid_t

        A_min = first_valid_k
        rho_max = t_r - t_c
        return A_min, t_c, t_r, rho_max


def get_bft_prob(N, A, gamma, delta):

    # https://www.sciencedirect.com/science/article/pii/S2215016121003009
    # Method (1) : Direct convolution of the two independent hypergeometric distributions

    T = int(gamma*N)    # corrupt aggregators
    D = int(delta*N)    # dropout aggregators

    bft_max_corrupt_and_dropouts = math.floor(A/3)  # BFT threshold

    # observed values for X
    xx = np.arange(0, A + 1, dtype=int)

    # PMF distributions for corrupt and dropouts
    pmf_corrupt = hypergeom.pmf(xx, N, T, A)
    pmf_dropout = hypergeom.pmf(xx, N, D, A)

    # PMF distribution of X_sum = X_corrupt + X_dropout
    pmf_sum = np.convolve(pmf_corrupt, pmf_dropout)

    # probability of X = (γ + δ) > 1/3
    bft_tail_probability = np.sum(pmf_sum[bft_max_corrupt_and_dropouts + 1:])

    return bft_tail_probability


def get_committee_size(N, sigma, eta, gamma, delta, rho = 1, BFT = False):
    '''
    Wrapper function to calculate the minimum committee size for security and correctness

    Argumnets:
    N (int): Total number of clients
    sigma (float): Security parameter
    eta (float): Correctness parameter
    gamma (float): Fraction of corrupt clients
    delta (float): Fraction of dropout clients
    rho (int): The desired rho value for the secret length, it must be > 1
    BFT (bool): Use calculations with Byzantine Fault Tolerance

    Returns:
    The tuple (A_min, t_c, t_r, rho_max)
    A_min: The minimum committee size < N
    t_c: The bounded corrupted committee members (aggregators)
    t_r: The reconstruction threshold for committee members (aggregators)
    rho_max: The maximum found value <= rho (input)
    '''
    cmt = Committee(N, sigma, eta, gamma, delta, rho, BFT)
    return cmt.get_committee_size()

