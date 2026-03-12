#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================

import numpy as np
import sympy
from typing import Any

npArray = np.ndarray[Any, np.dtype[Any]]


class Quantizer:
    '''
    Class for performing quantization/dequantization on floating point parameters
    
    Attributes:
    -----------
    clip_value:
        The maximum absolute value of allowed float parameters. The parameters
        are limited in the range [-clip_value, clip_value]
    clients_scale_factor:
        The scaling factor for float parameters to the maximum integer
    num_bits:
        The number of bits for the finite field range
    protocol:
        The name of the secure protocol to be used    
    '''
    def __init__(
        self,
        clip_value: float,
        clients_scale_factor: int,
        num_bits: int,
    ):
        '''
        Arguments:
        
        clip_value:
            The maximum absolute value of allowed float parameters. The parameters
            are limited in the range [-clip_value, clip_value]
        clients_scale_factor:
            The scaling factor for float parameters to the maximum integer
        num_bits:
            The number of bits for the finite field range
        protocol:
            The name of the secure protocol to be used
        '''
        self.clip_value = clip_value
        self.clients_scale_factor = clients_scale_factor
        self.num_bits = num_bits
        self.num_levels, self.field_prime_number = self._get_num_levels()
        self.scale_factor = self.num_levels / (2 * self.clip_value)

    def _get_max_prime(self, num_bits):
        '''Find the maximum prime number in the unsigned integer range of num_bits bits
        '''
        max_p = 2 ** num_bits
        min_p = max_p - 100000
        primes = list(sympy.primerange(min_p, max_p))
        return primes[-1]

    def _get_num_levels(self):
        '''Return the final number of levels
        '''
        q = self._get_max_prime(self.num_bits)
        n = (q - 1) // self.clients_scale_factor
        return n, q

    def print_levels(self):
        '''Print the number of levels for quantization'''
        msg = f'Quantization levels: {self.num_levels} ({self.num_levels.bit_length()} bits)'
        print(msg)

    def _clip_gradients(self, gradients: npArray):
        """Clips the gradients to the range [-clip_value, clip_value]."""
        return np.clip(gradients, -self.clip_value, self.clip_value)

    def _scale_and_shift_gradients(self, gradients: npArray):
        """Shifts the gradients to ensure all values are non-negative
            and scales them by a factor"""
        shift = self.clip_value  # Shift by the clip value to ensure all values are non-negative
        scaled_and_shifted_gradients = (gradients + shift) * self.scale_factor
        return scaled_and_shifted_gradients

    def _quantize_gradients(self, gradients: npArray):
        """Quantizes shifted and scaled gradients.
        Clipping is also applied in the final range {0, ..., num_levels-1}"""
        gradients = np.floor(gradients).astype(int)
        gradients = np.clip(gradients, 0, self.num_levels - 1).astype(int)
        return gradients

    def quantize(self, gradients: npArray):
        """Preprocess gradients by clipping, scaling, shifting, and quantizing."""
        clipped_gradients = self._clip_gradients(gradients)
        scaled_and_shifted_gradients = self._scale_and_shift_gradients(clipped_gradients)
        quantized_gradients = self._quantize_gradients(scaled_and_shifted_gradients)
        return quantized_gradients

    def dequantize(self, gradients: npArray):
        """Dequantize the quantized gradients by reversing the scaling and shifting effect."""
        dequantized_gradients = (gradients / self.scale_factor) - self.clip_value
        return dequantized_gradients


class DummyQuantizer(Quantizer):
    def __init__(
        self,
        clip_value: float,
        clients_scale_factor: int,
        num_bits: int
    ):
        super().__init__(clip_value, clients_scale_factor, num_bits)

    def quantize(self, gradients: npArray):
        return gradients

    def dequantize(self, gradients: npArray):
        return gradients
