#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================

from typing import List, Tuple

from Cryptodome.Protocol import SecretSharing
from Cryptodome.Util.Padding import pad, unpad


class Shamir:
    """
    Shamir class.

    Attributes:
    -----------
    threshold : int
        The sufficient number of shares to reconstruct the secret (``threshold < splits``).
    splits : int
        The number of shares that this method will create.
    ssss : Optional[bool]
        Bool value. Enabling the ssss value the polynomial used for splitting is slightly different and is compatible
        with the popular tool ssss (http://point-at-infinity.org/ssss/) when used with 128 bit security level and no
        dispersion. Defaults to False.
    padding : Optional[int]
        Integer value for the padding used. The underlying library Cryptodome uses 16 and is not configurable.
        The parameter is currently not usable. Defaults to 16.

    Methods:
    --------
    create_shares(secret: bytes) -> List[List[Tuple[int, bytes]]]
        Returns a list of lists with tuples containing an identifier (integer) and a share of the secret (bytes object)
         for each participant.
    combine_shares(shares: List[List[Tuple[int, bytes]]]) -> bytes
        Returns the recombined secret of the list of shares.
    _create_padded_chunks(self, secret: bytes) -> List[bytes]
        Returns a list of padded chunks (bytes).
    _shamir_split(self, chunk: bytes) -> List[Tuple[int, bytes]]
        Returns a list of tuples containing an identifier (integer) and a share of the secret (bytes object)
         for each participant.
    _shamir_combine(self, shares: List[Tuple[int, bytes]]) -> bytes
        Returns the recombined secret after combining shares and removing the standard padding.
    """
    # Cryptodome uses 16 byte padding as default and it is not configurable.
    padding: int = 16  # Defined for future use if padding is configurable

    def __init__(
        self, threshold: int, n_splits: int, ssss: bool = False, padding: int = 16
    ):
        """
        The constructor of Shamir class.
        If the ``ssss`` attribute is ``True`` the share generation and combination process is slightly adjusted.
        Specifically, it adds an extra term to the share, which is calculated as the index raised to the power of the
        polynomial's length. The extra term is designed to align with the well-known tool ssss
        (http://point-at-infinity.org/ssss/).

        Args:
        threshold : int
            The sufficient number of shares to reconstruct the secret (``threshold < splits``).
        splits : int
            The number of shares that this method will create.
        ssss : Optional[bool]
            Bool value. Enabling the ssss value the polynomial used for splitting is slightly different and is compatible
            with the popular tool ssss (http://point-at-infinity.org/ssss/) when used with 128 bit security level and no
            dispersion. Defaults to False.
        padding : Optional[int]
            Integer value for the padding used. The underlying library Cryptodome uses 16 and is not configurable.
            The parameter is currently not usable. Defaults to 16.
        Returns:
        None
        """
        # padding is not configurable in Cryptodome, so it is not used atm.
        self.threshold = threshold
        self.splits = n_splits
        self.ssss = ssss

    def create_shares(self, secret: bytes) -> List[List[Tuple[int, bytes]]]:
        """
        Function to create shares for the provided secret

        Args:
        secret: bytes
            The bytes to split and create a list of shares from.

        Returns:
        List of shares (bytes).
        """
        secret_padded_chunk = self._create_padded_chunks(secret)
        share_list: List[List[Tuple[int, bytes]]] = []

        # Loop can be parallelized
        for chunk in secret_padded_chunk:
            shares = self._shamir_split(chunk)
            share_list.append(shares)

        share_list = list(zip(*share_list))
        return share_list

    def combine_shares(self, shares: List[List[Tuple[int, bytes]]]) -> bytes:
        """
        Function to combine shares of secret and remove padding.

        Args:
        shares: List[List[Tuple[int, bytes]]]
            List of the shares to combine to a single secret.

        Returns:
        The combined secret
        """
        combined_secret = bytearray(0)

        # Loop can be parallelized
        for chunk_shares in shares:
            combined_secret += self._shamir_combine(chunk_shares)
        return unpad(combined_secret, self.padding)  # type: ignore

    def _create_padded_chunks(self, secret: bytes) -> List[bytes]:
        """
        Function to pad the secret and split in chunks

        Args:
        secret: bytes
           The bytes to split and create a list of padded chunks from.

        Returns:
        List of padded chunks (bytes)
        """
        secret_padded = pad(secret, self.padding)
        return [
            secret_padded[i : i + self.padding]
            for i in range(0, len(secret_padded), self.padding)
        ]

    def _shamir_split(self, chunk: bytes) -> List[Tuple[int, bytes]]:
        """
        Function to split a chunk of bytes

        If the ``ssss`` attribute is ``True`` an extra term is added to the share, which is calculated
        as the index raised to the power of the polynomial's length.
        The extra term is designed to align with the well-known tool ssss (http://point-at-infinity.org/ssss/).

        Args:
        chunk: bytes
           The padded chunk of bytes to split.

        Returns:
        A list of #(self.splits) tuples for equal participants.
        Each tuple contains an integer as the unique identifier and a share (a byte string, 16 bytes)
        """
        return SecretSharing.Shamir.split(
            self.threshold, self.splits, chunk, ssss=self.ssss
        )

    def _shamir_combine(self, shares: List[Tuple[int, bytes]]) -> bytes:
        """
        Function to recombine a secret, if enough shares are presented

        If the ``ssss`` attribute is ``True`` the extra term that was added to the share, which is calculated
        as the index raised to the power of the polynomial's length, will be considered in the combination.
        The extra term is designed to align with the well-known tool ssss (http://point-at-infinity.org/ssss/).

        Args:
        shares: List[Tuple[int, bytes]]
            A list of  tuples, each containing the index (an integer) and
            the share (a byte string, 16 bytes long) that were assigned to
            a participant.
          ssss (bool):
            If ``True``, the shares were produced by the ``ssss`` utility.
            Defaults to ``False``.

        Returns:
        The original secret, as a byte string (16 bytes long).
        """
        if len(shares) < self.threshold:
            raise ValueError("Shamir's secret sharing inadequate shares!")
        return SecretSharing.Shamir.combine(shares, ssss=self.ssss)
