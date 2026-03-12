#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================
import sys
import math
from typing import Any, Sequence, cast

import networkx as nx
import numpy as np
from networkx.generators.harary_graph import *
from scipy.stats import hypergeom

class HararyGraphGenerator:
    def __init__(
            self,
            nodes: Sequence[Any],
            security_parameter: float,
            correctness_parameter: float,
            max_corrupt_fraction: float,
            max_dropout_fraction: float,
            random_seed: int = 42
    ):
        '''
            Arguments
            --------
            
            nodes:
                A list of items representing the nodes of the graph
            security_parameter:
                Information-theoretic parameter, bounding the probability of bad events (sigma in paper)
            correctness_parameter:
                Parameter bounding the failure probability (eta in paper)
            max_corrupt_fraction:
                Maximum fraction of corrupted clients (gamma in paper)
            max_dropout_fraction:
                Maximum fraction of dropout clients (delta in paper)
            random_seed:
                Seed for random generator
        '''
        self.nodes = nodes
        self.security_parameter = security_parameter
        self.correctness_parameter = correctness_parameter
        self.max_corrupt_fraction = max_corrupt_fraction
        self.max_dropout_fraction = max_dropout_fraction
        self.harrary_graph_generator = nx.generators.harary_graph
        self.random_seed = random_seed
        self.graph = cast(nx.Graph, None)

        self.degree_k, self.threshold = self._compute_degree()
        self.mapping = {i: node for i, node in enumerate(self.nodes)}

    def generate_graph(self) -> nx.Graph:
        self.graph = cast(nx.Graph,
                          self.harrary_graph_generator.hkn_harary_graph(self.degree_k, len(self.nodes)))  # type: ignore
        self.graph = nx.relabel_nodes(self.graph, self.mapping)
        return self.graph

    def permute_graph(self):
        np.random.seed(self.random_seed)
        permutation = np.random.permutation(len(self.nodes))
        permuted_nodes = [self.mapping[i] for i in permutation]
        permuted_mapping = {
            node: permuted_nodes[i] for i, node in enumerate(self.graph.nodes())
        }
        self.graph = nx.relabel_nodes(self.graph, permuted_mapping)
        return self.graph

    def generate_permuted_graph(self) -> nx.Graph:
        self.generate_graph()
        return self.permute_graph()

    def _compute_degree(self):
        """This function computes the degree of the graph. The degree parameter is important
        to ensure that the graph achieves the (σ,η)-goodness property. The computation is not clear as in the paper they
        mention k >= O(log n + σ + η). 
        """
        k,t = binary_search_k_t(len(self.nodes), self.max_corrupt_fraction, 
                                self.max_dropout_fraction, 
                                self.security_parameter,
                                self.correctness_parameter
                                )
        if k == None:
            k = len(self.nodes) - 1 # use complete graph
        print('Graph degree-threshold: ', k, t)
        return k, t


def binary_search_k_t(total_clients, gamma, delta, sigma, eta):
    """
    Perform a binary search to find the minimum k and corresponding t that satisfy
    security and correctness conditions based on hypergeometric distributions, evaluated in the log domain.
    All k,t values are evaluated to satisfy the conditions of Lemma 3.7.
    Args:
    n (int): Total number of clients
    gamma (float): Fraction of corrupt clients
    delta (float): Fraction of dropout clients
    sigma (float): Security parameter (used in 2^-sigma)
    eta (float): Correctness parameter (used in 2^-eta)
    precision (float): Precision for binary search termination
    Returns:
    tuple: The optimal values for k and t
    """

    n = total_clients
    N = n - 1
    # K_corrupt (int): Number of corrupt clients
    K_corrupt = int(gamma * n)
    # K_surviving (int): Number of surviving clients
    K_surviving = int((1 - delta) * n)

    # Right part of the inequalities of Lemma 3.7 in the log space.
    threshold_security = (-sigma * math.log(2)) - math.log(n)
    threshold_correctness = (-eta * math.log(2)) - math.log(n)

    low_k, high_k = 1, N  # Set initial range for k
    optimal_k, optimal_t = None, None

    while high_k >= low_k:
        # mid_k = (low_k + high_k) // 2
        mid_k = low_k + (high_k - low_k) // 2

        # Search for the best t for this k
        low_t, high_t = 1, mid_k
        best_t_for_mid_k = None

        while high_t >= low_t:
            # mid_t = (low_t + high_t) // 2
            mid_t = low_t + (high_t - low_t) // 2

            # Security: P(X >= t) for corrupt clients
            cdf_value_security = hypergeom.cdf(mid_t - 1, N, K_corrupt, mid_k)
            p_security_log = math.log(max(sys.float_info.min, 1 - cdf_value_security + (delta + gamma) ** (mid_k / 2)))

            # Correctness: P(X >= t) for surviving clients
            cdf_value_correctness = hypergeom.cdf(mid_t, N, K_surviving, mid_k)
            p_correctness_log = math.log(max(sys.float_info.min, cdf_value_correctness))

            # Check if both security and correctness conditions are met
            if p_security_log < threshold_security and p_correctness_log < threshold_correctness:
                # Found a valid k and t
                best_t_for_mid_k = mid_t
                high_t = mid_t - 1  # Continue searching for smaller t in the left half
            else:
                low_t = mid_t + 1  # Search in the right half

        # If valid t was found, update optimal k and t
        if best_t_for_mid_k is not None:
            optimal_k, optimal_t = mid_k, best_t_for_mid_k
            high_k = mid_k - 1  # Continue searching for smaller k
        else:
            low_k = mid_k + 1  # Increase k if no valid t found

    return optimal_k, optimal_t
