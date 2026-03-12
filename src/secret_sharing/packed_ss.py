#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================
import numpy as np
import random

from secret_sharing.lcc_codec_mlsys import gen_Lagrange_coeffs


def lagrange_constants_for_points(points, target_points, p):
    
    C = gen_Lagrange_coeffs(target_points, points, p)

    out = {}
    for i,t in enumerate(target_points):
        out[t] = C[i,:]

    return out


def packed_share(secrets, share_points, T, p, share_coefficients):
    random_values = [random.randrange(p) for _ in range(T)]
    all_values = list(secrets) + random_values
    all_values = np.array(all_values, dtype=object)
    coeff = next(iter(share_coefficients.values()))
    all_values = all_values[:len(coeff)]

    shares = []
    for share_point in share_points:
        coefficients = share_coefficients[share_point]
        share = np.dot(coefficients, all_values)
        shares.append(share % p)

    return shares


def packed_reconstruct(shares, secret_points, reconstruction_coefficients, p):
    reconstructed_secrets = []
    shares = np.array(shares, dtype=object)

    for secret_point in secret_points:
        coefficients = reconstruction_coefficients[secret_point]
        secret_value = np.dot(coefficients, shares)
        reconstructed_secrets.append(secret_value % p)

    return reconstructed_secrets


def packed_mask_encoding(
    total_dimension,
    num_clients,
    targeted_number_active_clients,
    privacy_guarantee,
    prime_number,
    local_mask,
    rho=None,
    share_points=None,
    share_coefficients=None,
):
    d = total_dimension
    N = num_clients
    U = targeted_number_active_clients
    T = privacy_guarantee
    p = prime_number

    secrets_per_poly = rho

    # Validate parameters
    if secrets_per_poly > U - T:
        raise ValueError(
            f"rho ({secrets_per_poly}) cannot be greater than U-T ({U-T}) for security"
        )
    if secrets_per_poly <= 0:
        raise ValueError(f"rho ({secrets_per_poly}) must be positive")

    num_polynomials = d // secrets_per_poly
    if d % secrets_per_poly != 0:
        raise ValueError(f"Dimension {d} must be divisible by rho={secrets_per_poly}")

    encoded_mask_set = np.zeros((N, num_polynomials), dtype=object)

    for poly_idx in range(num_polynomials):
        start_idx = poly_idx * secrets_per_poly
        end_idx = start_idx + secrets_per_poly
        secrets = local_mask[start_idx:end_idx].flatten().tolist()
        secrets = [int(s) % p for s in secrets]

        shares = packed_share(secrets, share_points, T, p, share_coefficients)

        for client_idx in range(N):
            encoded_mask_set[client_idx, poly_idx] = shares[client_idx]

    return encoded_mask_set


def packed_aggregate_mask_reconstruction(
    total_dimension,
    num_clients,
    targeted_number_active_clients,
    privacy_guarantee,
    prime_number,
    aggregate_encoded_mask_buffer,
    active_clients_indices,
    rho=None,
    secret_points=None,
    reconstruction_coefficients=None,
):
    d = total_dimension
    N = num_clients
    U = targeted_number_active_clients
    T = privacy_guarantee
    p = prime_number

    secrets_per_poly = rho

    # Validate parameters
    if secrets_per_poly > U - T:
        raise ValueError(
            f"rho ({secrets_per_poly}) cannot be greater than U-T ({U-T}) for security"
        )

    num_polynomials = d // secrets_per_poly

    reconstructed_mask = []

    for poly_idx in range(num_polynomials):
        shares = []
        for active_idx in range(len(active_clients_indices)):
            shares.append(aggregate_encoded_mask_buffer[active_idx, poly_idx])

        reconstructed_secrets = packed_reconstruct(
            shares, secret_points, reconstruction_coefficients, p
        )
        reconstructed_mask.extend(reconstructed_secrets)

    aggregate_mask = np.array(reconstructed_mask, dtype=object)
    aggregate_mask = np.reshape(aggregate_mask, (d, 1))

    return aggregate_mask
