#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================
import numpy as np
import secret_sharing.lcc_codec_mlsys as lcc_codec_mlsys


class LagrangeCodec():
    '''
        Class that performs encoding and decoding of the secret shares
        according to the LightSecAgg protocol, based on Lagrange Coded Computing
        (https://arxiv.org/abs/1806.00939)
    '''
    def __init__(self):
        self.encoder_matrix = None

    def _get_module_name(self):
        return lcc_codec_mlsys        

    def load(self, enc_m, dec_m):
        self.encoder_matrix = enc_m
        self.decoder_matrix = dec_m

    def create_codec(self, N, U, p, cache_decoder=False):
        beta_s = np.array(range(N)) + 1
        alpha_s = np.array(range(U)) + (N+1)
        enc_m = self._get_module_name().gen_Lagrange_coeffs(beta_s, alpha_s, p)
        enc_m = enc_m.view(np.ndarray)

        if cache_decoder:
            alpha_s = np.array(range(U)) + 1
            beta_s = np.array(range(U)) + (N+1)
            dec_m = self._get_module_name().gen_Lagrange_coeffs(beta_s, alpha_s, p)
            dec_m = dec_m.view(np.ndarray)
        else:
            dec_m = None

        self.load(enc_m, dec_m)

    def encode(self, d, N, U, T, p, mask):
        return self._get_module_name().mask_encoding(d, N, U, T, p, mask, self.encoder_matrix)

    def decode(self, d, N, U, T, p, mask, idx):
        return self._get_module_name().aggregate_mask_reconstruction(d, N, U, T, p, mask, idx,
                                                                     self.decoder_matrix)

