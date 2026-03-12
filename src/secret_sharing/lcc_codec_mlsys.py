#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================

# Code adapted from: https://github.com/LightSecAgg/MLSys2022_anonymous/blob/master/fedml_api/distributed/lightsecagg/mpc_function.py

import numpy as np
import random

def modular_inv(a, p):
    a = int(a)
    x, y, m = 1, 0, p
    while a > 1:
        q = a // m
        t = m

        m = int(a) % m
        a = t
        t = y

        y, x = x - q * y, t

        if x < 0:
            x = x % p
    return x % p


def div_mod(_num, _den, _p):
    # compute num / den modulo prime p
    _num = int(_num) % _p
    _den = int(_den) % _p
    _inv = modular_inv(_den, _p)
    return (int(_num) * _inv) % _p


def PI(vals, p):  # upper-case PI -- product of inputs
    accum = 1
    for v in vals:
        tmp = int(v) % p
        accum = (accum * tmp) % p
    return accum


def LCC_encoding_with_points(X, alpha_s, beta_s, p, enc_m):
    m, d = np.shape(X)
    U = enc_m
    X_LCC = np.zeros((len(beta_s), d), dtype=object)
    for i in range(len(beta_s)):
        X_LCC[i, :] = np.dot(np.reshape(U[i, :], (1, len(alpha_s))), X)
    return X_LCC % p


def LCC_decoding_with_points(f_eval, eval_points, target_points, p, dec_m):
    alpha_s_eval = eval_points
    beta_s = target_points
    if dec_m is None:
        U_dec = gen_Lagrange_coeffs(beta_s, alpha_s_eval, p)
    else:
        U_dec = dec_m
    f_recon = (U_dec).dot(f_eval) % p

    return f_recon


def gen_Lagrange_coeffs(alpha_s, beta_s, p, is_K1=0):
    if is_K1 == 1:
        num_alpha = 1
    else:
        num_alpha = len(alpha_s)
    U = np.zeros((num_alpha, len(beta_s)), dtype=object)

    w = np.zeros((len(beta_s)), dtype=object)
    for j in range(len(beta_s)):
        cur_beta = beta_s[j]
        den = PI([int(cur_beta) - int(o) for o in beta_s if cur_beta != o], p)
        w[j] = den
    l = np.zeros((num_alpha), dtype=object)
    for i in range(num_alpha):
        l[i] = PI([int(alpha_s[i]) - int(o) for o in beta_s], p)

    for j in range(len(beta_s)):
        for i in range(num_alpha):
            den = ((int(alpha_s[i]) - int(beta_s[j])) % p * int(w[j]) % p) % p
            U[i][j] = div_mod(l[i], den, p)
    return U


def mask_encoding(
    total_dimension,
    num_clients,
    targeted_number_active_clients,
    privacy_guarantee,
    prime_number,
    local_mask,
    enc_m
):
    d = total_dimension
    N = num_clients
    U = targeted_number_active_clients
    T = privacy_guarantee
    p = prime_number

    beta_s = np.array(range(N)) + 1
    alpha_s = np.array(range(U)) + (N + 1)

    k = T * d // (U - T)
    rand_nums = [random.randint(0, p-1) for _ in range(k)]
    rand_nums = np.array(rand_nums, dtype=object)
    n_i = np.reshape(rand_nums, (k, 1))

    LCC_in = np.concatenate([local_mask, n_i], axis=0)
    LCC_in = np.reshape(LCC_in, (U, d // (U - T)))
    encoded_mask_set = LCC_encoding_with_points(LCC_in, alpha_s, beta_s, p, enc_m)

    return encoded_mask_set


def aggregate_mask_reconstruction(
    total_dimension,
    num_clients,
    targeted_number_active_clients,
    privacy_guarantee,
    prime_number,
    aggregate_encoded_mask_buffer,
    active_clients_indices,
    dec_m
):
    """
    Recover the aggregate-mask via decoding
    """
    d = total_dimension
    N = num_clients
    U = targeted_number_active_clients
    T = privacy_guarantee
    p = prime_number

    alpha_s = np.array(range(N)) + 1
    beta_s = np.array(range(U)) + (N + 1)

    eval_points = alpha_s[active_clients_indices]
    aggregate_mask = LCC_decoding_with_points(
        aggregate_encoded_mask_buffer, eval_points, beta_s, p, dec_m
    )

    aggregate_mask = np.reshape(aggregate_mask, (U * (d // (U - T)), 1))
    aggregate_mask = aggregate_mask[0:d]
    return aggregate_mask


def compute_aggregate_encoded_mask(encoded_mask_dict, p, active_clients):
    aggregate_encoded_mask = np.zeros((np.shape(encoded_mask_dict[0])))
    for client_id in active_clients:
        aggregate_encoded_mask += encoded_mask_dict[client_id]
    aggregate_encoded_mask = np.mod(aggregate_encoded_mask, p).astype("int")
    return aggregate_encoded_mask
