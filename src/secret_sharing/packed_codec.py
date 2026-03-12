#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================
import secret_sharing.packed_ss as packed_ss


class PackedCodec:
    """
    Class that performs encoding and decoding of the secret shares
    according to the packed secret sharing scheme, providing the same
    interface as LagrangeCodec for easy replacement.
    """

    def __init__(self, rho=None):
        self.N = None  # Number of clients
        self.U = None  # Targeted number of active clients
        self.p = None  # Prime number
        self.rho = rho  # Number of secrets per polynomial
        self.share_points = None  # Share points for encoding
        self.secret_points = None  # Secret points for reconstruction
        self.share_coefficients = None  # Precomputed coeffs for sharing
        self.reconstruction_coefficients = None  # Precomputed coeffs for reconstruction

    def create_codec(self, N, U, p):
        self.N = N
        self.U = U
        self.p = p

        # Share points: 1, 2, ..., N (positive integers for clients)
        share_points = list(range(1, N + 1))

        # Secret points: -1, -2, ..., -rho (negative integers for secrets)
        secret_points = [-(i + 1) for i in range(self.rho)]

        # cache encoder for clients
        self.share_coefficients = packed_ss.lagrange_constants_for_points(
            secret_points, share_points, p
        )

        self.share_points = share_points
        self.secret_points = secret_points

    def encode(self, d, N, U, T, p, mask):
        return packed_ss.packed_mask_encoding(
            d,
            N,
            U,
            T,
            p,
            mask,
            rho=self.rho,
            share_points=self.share_points,
            share_coefficients=self.share_coefficients,
        )

    def decode(self, d, N, U, T, p, mask, idx):

        # the server must construct the decoder each time
        reconstruction_points = list(range(1, U + 1))
        self.reconstruction_coefficients = (
            packed_ss.lagrange_constants_for_points(
                reconstruction_points, self.secret_points, p
            )
        )

        return packed_ss.packed_aggregate_mask_reconstruction(
            d,
            N,
            U,
            T,
            p,
            mask,
            idx,
            rho=self.rho,
            secret_points=self.secret_points,
            reconstruction_coefficients=self.reconstruction_coefficients,
        )
