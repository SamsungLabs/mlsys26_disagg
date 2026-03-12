#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================
import numpy as np
import time
import random
import math
from tqdm import tqdm
from multiprocessing.pool import Pool
import multiprocessing as mp
import pickle

from queue import Empty

from constants import init_parameters, num_proc

import utils.quantization as quantization
import common
from constants import init_parameters, num_proc
from secret_sharing.lcc_codec import LagrangeCodec
from secret_sharing.packed_codec import PackedCodec
import utils.train_utils as train_utils
import utils.committee as cmt

SHARING_SCHEME = "packed"


random.seed(42)
np.random.seed(42)


# custom non-daemon Pool to allow nested multiprocessing (for DataLoader workers)
class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(mp.get_context('spawn'))):
    Process = NoDaemonProcess

class NoDaemonPool(Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NoDaemonPool, self).__init__(*args, **kwargs)


class PRG:
    def __init__(self, lambda_key_dim, M_model_dim, q_field_size, p_model_field_size):
        """
        Initialize PRG with necessary parameters.

        Args:
            lambda_key_dim: Key dimension (lambda)
            M_model_dim: Model dimension (M)
            q_field_size: Field size for the PRG operations (q)
            p_model_field_size: Model field size (p), where p < q
        """
        self.lambda_key_dim = lambda_key_dim
        self.M_model_dim = M_model_dim
        self.q_field_size = q_field_size
        self.p_model_field_size = p_model_field_size

        # Calculate log(p) for bit extraction using Python's math.log2 for arbitrary precision
        self.log_p_bits = int(math.ceil(math.log2(p_model_field_size)))

    @staticmethod
    def Gen(lambda_key_dim, M_model_dim, q_field_size, matrix_seed=42):
        """
        Generate a random matrix A from seed. This is a static method since
        the matrix A should be the same for all PRG operations.

        Args:
            lambda_key_dim: Key dimension (lambda)
            M_model_dim: Model dimension (M)
            q_field_size: Field size for the PRG operations (q)
            matrix_seed: Seed for generating the matrix A (default: 42)

        Returns:
            A: Random matrix in Z_q^(lambda x M)
        """
        np.random.seed(matrix_seed)

        if q_field_size > (2 ** 64):    # for F = 128
            AH = np.random.randint(0, 2**64-1, size=(lambda_key_dim, M_model_dim), dtype=np.uint64).astype(object)
            AL = np.random.randint(0, 2**64-1, size=(lambda_key_dim, M_model_dim), dtype=np.uint64).astype(object)
            A = np.mod((AH << 64) + AL, q_field_size)
        else:
            A = np.random.randint(0, q_field_size, size=(lambda_key_dim, M_model_dim), dtype=np.uint64).astype(object)

        return A

    def Expand(self, sd, A):
        """
        Expand seed to generate vector in Z_p^M using pre-generated matrix A.

        Args:
            sd: Seed vector in Z_q^lambda
            A: Pre-generated random matrix from Gen()

        Returns:
            M: Vector in Z_p^M (rounded to log(p) most significant bits)
        """

        # Ensure sd is a numpy array with dtype=object for large integers
        if not isinstance(sd, np.ndarray):
            sd = np.array(sd, dtype=object)
        else:
            sd = sd.astype(object)

        # Compute X = A^T * sd (resulting in Z_q^M)
        X = np.dot(A.T, sd) % self.q_field_size

        # Ensure X contains Python integers for bit operations
        X = np.array([int(x) for x in X.flatten()], dtype=object)

        # Extract the log(p) most significant bits using Python's bit operations
        shift_amount = int(math.ceil(math.log2(self.q_field_size))) - self.log_p_bits
        if shift_amount >= 0:
            M = np.array(
                [(int(x) >> shift_amount) % self.p_model_field_size for x in X],
                dtype=object,
            )
        else:
            raise ValueError(
                "shift_amount must be positive to extract most significant bits."
            )

        M = np.reshape(M, (len(M),1))
        return M


class Member(common.MemberBase):
    def __init__(self, parameters, codec, q_quantizer, p_quantizer, m_quantizer):
        super().__init__(parameters)
        self.codec = codec
        self.q_quantizer = q_quantizer
        self.p_quantizer = p_quantizer
        self.m_quantizer = m_quantizer
        self.model = None
        self.scaled_model = None
        self.all_keys = None

    def _save_model(self, b_scaled = False):
        if not self.parameters['USE_DISK']: return
        if not b_scaled:
            fname = f'{self.temp_dir}/{self.cid}_model.npy'
            np.save(fname, self.model)
            self.model = None
        else:
            fname = f'{self.temp_dir}/{self.cid}_scaled_model.npy'
            np.save(fname, self.scaled_model)
            self.scaled_model = None

    def _load_model(self, b_scaled = False):
        if not b_scaled:
            if not self.parameters['USE_DISK']: return self.model
            fname = f'{self.temp_dir}/{self.cid}_model.npy'
        else:
            if not self.parameters['USE_DISK']: return self.scaled_model
            fname = f'{self.temp_dir}/{self.cid}_scaled_model.npy'
        A = np.load(fname, allow_pickle=True)
        return A

    def get_model(self, b_scaled = False):
        return self._load_model(b_scaled)

    def _save_A_matrix(self, A):
        fname = f'{self.temp_dir}/a_matrix.npy'
        np.save(fname, A)

    def _load_A_matrix(self):
        fname = f'{self.temp_dir}/a_matrix.npy'
        A = np.load(fname, allow_pickle=True)
        return A

    def _save_codec(self):
        if not self.parameters['USE_DISK']: return
        fname = f'{self.temp_dir}/ss_codec.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(self.codec.share_coefficients, f)
        self.codec.share_coefficients = None

    def _load_codec(self):
        if not self.parameters['USE_DISK']: return
        fname = f'{self.temp_dir}/ss_codec.pkl'
        with open(fname, 'rb') as f:
            A = pickle.load(f)
        return A

    def _save_masked_model(self, masked_model):
        if not self.parameters['USE_DISK']: return masked_model
        fname = f'{self.temp_dir}/{self.cid}_masked_model.npy'
        np.save(fname, masked_model)
        return fname

    def _load_masked_model(self, path = None):
        if path is None:
            fname = f'{self.temp_dir}/{self.cid}_masked_model.npy'
        else:
            fname = path
        tstart = time.time()
        A = np.load(fname, allow_pickle=True)
        dt = time.time() - tstart
        self.disk_timing += dt
        return A


class Client(Member):
    def __init__(self, parameters, codec, q_quantizer, p_quantizer, m_quantizer, cid):
        super().__init__(parameters, codec, q_quantizer, p_quantizer, m_quantizer)
        self.cid = cid
        self.sd = None  # Will be set in _prepare_model
        if not self.parameters['USE_TRAINING']:
            self._prepare_model()

    def _prepare_model(self):
        # random model
        m = np.random.random(size=(self.parameters["M"]))
        self.model = self.m_quantizer.quantize(m)
        self.model = np.reshape(self.model, (len(self.model),1))
        self._encode_and_scale_model()
        self._save_model(False)
        self._save_model(True)

    def _encode_and_scale_model(self):
        #  ENCODE
        scaling_factor = self.parameters["N"]
        self.scaled_model = self.model * scaling_factor + 1

        lambda_key_dim = self.parameters["lambda_key_dim"]  # Security parameter
        q_field_size = self.q_quantizer.field_prime_number  # Use quantizer's field size

        # Use Python's random for arbitrary-precision support
        self.sd = np.array(
            [random.randrange(q_field_size) for _ in range(lambda_key_dim)], dtype=object
        )

    def fit(self):
        train_utils.set_params(self.parameters)
        gm, sh = train_utils.load_global_model()
        self.comtime['down'] = self._get_com_time(gm, 'd', 'N')
        self.comtime['up'] = self._get_com_time(gm, 'u', 'N')
        self.quantizer = self.m_quantizer

        tstart = time.time()
        self.train_and_quantize_model()
        self.elt = time.time() - tstart
        
        self._save_model(False)

    def mask_model(self, A_matrix):
        """
        Compute masked model: m + mask where m is the client's model
        and mask is generated from PRG.Expand(self.sd, A_matrix)

        Args:
            A_matrix: Pre-generated PRG matrix A

        Returns:
            masked_model: The model with pseudorandom mask applied
        """
        if self.parameters['USE_DISK']:
            A_matrix = self._load_A_matrix()

        if not self.parameters['USE_TRAINING']:
            scaled_model = self._load_model(True)
            self.comtime['down'] = self._get_com_time(scaled_model, 'd', 'N')
        else:
            train_utils.set_params(self.parameters)
            gm, sh = train_utils.load_global_model()
            self.comtime['down'] = self._get_com_time(gm, 'd', 'N')

        tstart = time.time()

        if self.parameters['USE_TRAINING']:
            self.quantizer = self.m_quantizer
            self.train_and_quantize_model()
            self._encode_and_scale_model()
            scaled_model = self.scaled_model

        if scaled_model is None or self.sd is None:
            raise ValueError(
                "Model and seed must be prepared first. Call _prepare_model()."
            )

        # Create PRG instance with appropriate parameters
        lambda_key_dim = self.parameters["lambda_key_dim"]
        M_model_dim = self.parameters["M"]
        q_field_size = self.q_quantizer.field_prime_number
        p_model_field_size = self.p_quantizer.field_prime_number

        prg = PRG(lambda_key_dim, M_model_dim, q_field_size, p_model_field_size)

        # Generate mask using PRG.Expand with pre-generated matrix A
        mask = prg.Expand(self.sd, A_matrix)

        # Ensure shapes match
        if mask.shape != scaled_model.shape:
            raise ValueError(
                f"Mask shape {mask.shape} doesn't match model shape {scaled_model.shape}"
            )

        # Compute masked model: m + mask (mod p) using Python arithmetic
        masked_model = np.array(
            [
                (int(m) + int(mask_val)) % p_model_field_size
                for m, mask_val in zip(scaled_model, mask)
            ],
            dtype=object,
        )
        elt = time.time() - tstart

        self.comtime['up'] = self._get_com_time(masked_model, 'u', 'N')

        # if USE_DISK it returns a file name of the masked_model
        masked_model = self._save_masked_model(masked_model)

        if self.parameters['USE_TRAINING']:
            self._save_model(False)
            self._save_model(True)

        if self.dropped:    # dropped client
            masked_model = None

        return masked_model, elt

    def share_key(self):
        """
        Produce shares of sd for each committee member (aggregator).
        Uses mask_encoding with specific parameters for key sharing.

        Returns:
            key_shares: Encoded shares of the seed for committee members
        """
        if self.parameters['USE_DISK']:
            self.codec.share_coefficients = self._load_codec()

        tstart = time.time()
        if self.sd is None:
            raise ValueError("Seed must be prepared first. Call _prepare_model().")

        # Parameters for mask_encoding
        lambda_key_dim = self.parameters["lambda_key_dim"]
        A = self.parameters["A"]  # committee size (same notation as aggregators)
        q = self.q_quantizer.field_prime_number

        total_dimension = lambda_key_dim
        num_clients = A
        targeted_number_active_clients = self.parameters['U']
        
        privacy_guarantee = self.parameters["T_c"]
        prime_number = q

        U = targeted_number_active_clients

        if SHARING_SCHEME == "packed":
            divisor = self.parameters['RHO']
        elif SHARING_SCHEME == "lcc":
            divisor = U - privacy_guarantee
        else:
            raise ValueError(f"Unknown sharing scheme: {SHARING_SCHEME}")

        # Pad dimension to be divisible by the appropriate divisor
        chunk_size = total_dimension // divisor
        if total_dimension % divisor != 0:
            chunk_size += 1
            padded_dimension = chunk_size * divisor
            padded_sd = np.zeros(padded_dimension, dtype=object)
            padded_sd[:lambda_key_dim] = self.sd
            local_mask = padded_sd.reshape(-1, 1)
            total_dimension = padded_dimension
        else:
            local_mask = self.sd.reshape(-1, 1)

        key_shares = self.codec.encode(
            total_dimension,
            num_clients,
            targeted_number_active_clients,
            privacy_guarantee,
            prime_number,
            local_mask,
        )
        if self.parameters['USE_DISK']:
            self.codec.share_coefficients = None

        elt = time.time() - tstart

        # add to previous masked_model upload time
        self.comtime['up'] += self._get_com_time(key_shares, 'u', 'N')

        if self.dropped:    # dropped client
            key_shares = None

        return key_shares, elt, self.comtime


class Committee(Member):
    def __init__(self, parameters, codec, q_quantizer, p_quantizer, m_quantizer):
        super().__init__(parameters, codec, q_quantizer, p_quantizer, m_quantizer)

    def aggregate_shares(self, client_shares_dict):
        """
        Aggregate shares from multiple clients by summing them.

        Args:
            client_shares_dict: Dictionary with client_id as keys and their shares as values
                               Format: {client_id: shares_matrix, ...}

        Returns:
            aggregated_shares: Sum of all client shares (mod prime_number)
        """
        self.comtime['down'] = self._get_com_time(client_shares_dict, 'd', 'A')
        tstart = time.time()
        if not client_shares_dict:
            raise ValueError("No client shares provided for aggregation.")

        # Get the prime number from quantizer
        q = self.q_quantizer.field_prime_number

        # Get the first shares to determine the shape
        first_client_id = next(iter(client_shares_dict))
        first_shares = client_shares_dict[first_client_id]

        # Initialize aggregated shares with zeros of the same shape
        aggregated_shares = np.zeros_like(first_shares, dtype=object)
        
        # Sum all client shares
        for client_id, shares in client_shares_dict.items():
            # Validate shape consistency
            if shares.shape != aggregated_shares.shape:
                raise ValueError(
                    f"Client {client_id} shares shape {shares.shape} "
                    f"doesn't match expected shape {aggregated_shares.shape}"
                )

            # Add shares using Python arithmetic for large moduli
            aggregated_shares += np.array(shares, dtype=object)

        # Apply modulo operation to keep values in the field using Python arithmetic
        aggregated_shares = np.mod(aggregated_shares, q)
        
        elt = time.time() - tstart
        self.comtime['up'] = self._get_com_time(aggregated_shares, 'u', 'A')
        return aggregated_shares, elt, self.comtime


class Server(Member):
    def __init__(self, parameters, codec, q_quantizer, p_quantizer, m_quantizer):
        super().__init__(parameters, codec, q_quantizer, p_quantizer, m_quantizer)

    def fit_clients(self, clients):
        if num_proc > 0:
            with mp.Manager() as manager:
                lock = manager.Lock()
                for i in range(len(clients)):
                    clients[i].lock = lock
                with NoDaemonPool(num_proc) as p:
                    all_res = list(tqdm(p.imap(client_fit_worker, clients),
                                        total=len(clients), desc='Fit clients'))
                    p.close()
                    p.join()
                for i in range(len(clients)):
                    clients[i].lock = None
        else:
            all_res = [client_fit_worker(c) for c in tqdm(clients, desc='Fit clients')]

        for res in all_res:
            cid, elt, comtime = res
            self.clk.add('Client_Down', comtime['down'])
            self.clk.add('Client_Proc', elt)
            self.clk.add('Client_Up', comtime['up'])

    def reconstruct(self, committee_aggregated_shares_dict, active_committee_indices):
        """
        Reconstruct the aggregated seed from committee members' aggregated shares.

        Args:
            committee_aggregated_shares_dict: Dictionary with committee_id as keys and
                                            their aggregated shares as values
                                            Format: {committee_id: aggregated_shares_matrix, ...}
            active_committee_indices: List/array of indices of active committee members

        Returns:
            reconstructed_aggregate_seed: The reconstructed aggregated seed (trimmed to original size)
        """
        if not committee_aggregated_shares_dict:
            raise ValueError(
                "No committee aggregated shares provided for reconstruction."
            )

        # Parameters for reconstruction (same as used in share_key)
        lambda_key_dim = self.parameters["lambda_key_dim"]
        A = self.parameters["A"]  # committee size (aggregators)
        q = self.q_quantizer.field_prime_number

        # Calculate the padded dimension (same logic as in share_key)
        U = self.parameters['U']
        privacy_guarantee = self.parameters["T_c"]

        if SHARING_SCHEME == "packed":
            divisor = self.parameters['RHO']
        elif SHARING_SCHEME == "lcc":
            divisor = U - privacy_guarantee
        else:
            raise ValueError(f"Unknown sharing scheme: {SHARING_SCHEME}")

        # Pad dimension to be divisible by the appropriate divisor (same logic as in share_key)
        chunk_size = lambda_key_dim // divisor
        if lambda_key_dim % divisor != 0:
            chunk_size += 1
            padded_dimension = chunk_size * divisor
        else:
            padded_dimension = lambda_key_dim

        # Get the first aggregated shares to determine the shape
        first_committee_id = next(iter(committee_aggregated_shares_dict))
        first_aggregated_shares = committee_aggregated_shares_dict[first_committee_id]

        # Create array for active committee shares (extract only active ones)
        active_aggregated_shares = np.zeros(
            (len(active_committee_indices), first_aggregated_shares.shape[1]),
            dtype=object,
        )

        # Fill in the shares from active committee members
        for i, committee_idx in enumerate(active_committee_indices):
            committee_id = f"committee_{committee_idx}"
            if committee_id in committee_aggregated_shares_dict:
                # Each committee member has aggregated their specific shares into a single row
                active_aggregated_shares[i, :] = committee_aggregated_shares_dict[
                    committee_id
                ][0, :]
            else:
                raise ValueError(
                    f"Committee {committee_id} not found in aggregated shares"
                )

        # Reconstruct using aggregate_mask_reconstruction
        reconstructed_aggregate_seed = self.codec.decode(
            padded_dimension,  # total_dimension (padded)
            A,  # num_clients (committee size)
            U,  # targeted_number_active_clients
            privacy_guarantee,  # privacy_guarantee
            q,  # prime_number
            active_aggregated_shares,  # aggregate_encoded_mask_buffer
            active_committee_indices,  # active_clients_indices
        )

        # Extract the relevant part from reconstruction (remove padding)
        reconstructed_seed_trimmed = reconstructed_aggregate_seed[
            :lambda_key_dim
        ].flatten()

        return reconstructed_seed_trimmed

    def recover_model(
        self, reconstructed_aggregate_seed, client_masked_models_dict, A_matrix
    ):
        """
        Recover the aggregated model from the reconstructed seed and client masked models.

        Args:
            reconstructed_aggregate_seed: The reconstructed aggregated seed from reconstruct()
            client_masked_models_dict: Dictionary with client_id as keys and their masked models as values
                                     Format: {client_id: masked_model, ...}
            A_matrix: Pre-generated PRG matrix A

        Returns:
            recovered_aggregated_model: The recovered aggregated model
        """
        if reconstructed_aggregate_seed is None:
            raise ValueError("Reconstructed aggregate seed is required.")

        if not client_masked_models_dict:
            raise ValueError("Client masked models dictionary is required.")

        self.disk_timing = 0    # store accumulated disk loading timings

        # Aggregate the masked models (CT = sum of all client masked models)
        first_client_id = 0
        # search for the first non None item
        for i in client_masked_models_dict:
            if client_masked_models_dict[i] is None:
                continue
            else:
                first_client_id = i
                break

        first_masked_model = client_masked_models_dict[first_client_id]
        if self.parameters['USE_DISK']:
            # masked model data are replaced by paths (strings)
            first_masked_model = self._load_masked_model(first_masked_model)

        # Get field sizes
        q = self.q_quantizer.field_prime_number  # larger field for seed operations
        p = self.p_quantizer.field_prime_number  # smaller field for model operations

        # Initialize aggregated masked models with zeros
        aggregated_masked_models = np.zeros_like(first_masked_model, dtype=object)

        # Sum all client masked models using Python arithmetic
        for client_id, masked_model in client_masked_models_dict.items():
            if masked_model is None:
                continue
            if self.parameters['USE_DISK']:
                masked_model = self._load_masked_model(masked_model)
            if masked_model.shape != aggregated_masked_models.shape:
                raise ValueError(
                    f"Client {client_id} masked model shape {masked_model.shape} "
                    f"doesn't match expected shape {aggregated_masked_models.shape}"
                )
            aggregated_masked_models += np.array(masked_model, dtype=object)

        # Apply modulo p for model field using Python arithmetic
        aggregated_masked_models = np.mod(aggregated_masked_models, p)

        aggregated_masked_models = np.reshape(aggregated_masked_models,
                                              (len(aggregated_masked_models),1))

        # Create PRG instance with appropriate parameters
        lambda_key_dim = self.parameters["lambda_key_dim"]
        M_model_dim = self.parameters["M"]
        q_field_size = q  # larger field for PRG operations
        p_model_field_size = p  # smaller field for model operations

        prg = PRG(lambda_key_dim, M_model_dim, q_field_size, p_model_field_size)

        # Generate aggregate mask (AUX) using PRG.Expand with reconstructed seed and pre-generated matrix A
        AUX = prg.Expand(reconstructed_aggregate_seed, A_matrix)

        # Ensure shapes match
        if AUX.shape != aggregated_masked_models.shape:
            raise ValueError(
                f"AUX shape {AUX.shape} doesn't match aggregated masked models shape {aggregated_masked_models.shape}"
            )

        # Recover aggregated model: CT - AUX (mod p) using Python arithmetic
        recovered_aggregated_model = np.array(
            [(int(ct) - int(aux)) % p for ct, aux in zip(aggregated_masked_models, AUX)],
            dtype=object,
        )

        # DECODE
        scaling_factor = self.parameters["N"]
        recovered_aggregated_model = np.array(
            [int(val) // scaling_factor - 1 for val in recovered_aggregated_model],
            dtype=object,
        )

        return recovered_aggregated_model


def create_client(input_args):
    parameters, codec, q_quantizer, p_quantizer, m_quantizer, i = input_args
    return Client(parameters, codec, q_quantizer, p_quantizer, m_quantizer, i)


def client_worker(input_args):
    c, a = input_args
    ret1, t1 = c.mask_model(a)
    ret2, t2, t3 = c.share_key()
    return c.cid, ret1, ret2, (t1 + t2), t3


def client_fit_worker(c):
    c.fit()
    return (c.cid, c.elt, c.comtime)


def worker_process(client_sublist, A_matrix, queue):
    for client in client_sublist:
        cid, client_mask, client_secret_shares, elt, comtime = client_worker((client, A_matrix))
        queue.put((cid, client_mask, client_secret_shares, elt, comtime))
    return

def watcher(queue, masked_models_dict, client_shares_dict, clock, total_items):
    """Watch the queue and update the progress bar in the main thread."""
    pbar = tqdm(total=total_items, desc="Clients computation")
    processed_count = 0

    while True:
        try:
            result = queue.get(timeout=1)
            if result is None:  # Sentinel value to stop the watcher
                break
            cid = result[0]
            _, masked_models_dict[cid], client_shares_dict[cid], elt, comtime = result
            clock.add("Client_Down", comtime['down'])
            clock.add("Client_Proc", elt)
            clock.add("Client_Up", comtime['up'])
            pbar.update(1)
            processed_count += 1
        except Empty:
            continue

    pbar.close()
    print(f"Processed {processed_count} items")



def start_simulation(parameters):

    parameters = common.parse_parameters(parameters)

    print("=== Secure Aggregation Protocol Test ===")
    print(f"Using secret sharing scheme: {SHARING_SCHEME.upper()}")

    # Initialize clock for timing measurements
    clock = common.Clock()

    # === SETUP ===
    clock.tic()

    q_quantizer = quantization.Quantizer(
        clip_value=2.0,
        clients_scale_factor=parameters["N"],
        num_bits=parameters["F"]
    )
    p_quantizer = quantization.Quantizer(
        clip_value=2.0,
        clients_scale_factor=parameters["N"],
        num_bits=parameters["P"]
    )
    plaintext_field_bits = int(parameters["P"] - np.ceil(np.log2(parameters["N"])))
    print(f"plaintext_field_bits = {plaintext_field_bits}")
    m_quantizer = quantization.Quantizer(
        clip_value=2.0,
        clients_scale_factor=parameters["N"],
        num_bits=plaintext_field_bits
    )
    clock.toc("Setup_Quantizer")

    if parameters['USE_QUANTIZATION'] == False:
        m_quantizer = quantization.DummyQuantizer(
            clip_value=2.0,
            clients_scale_factor=parameters["N"],
            num_bits=plaintext_field_bits
        )

    # chose committee size and threshold
    A_min, t_c, t_r, rho_max = cmt.get_committee_size(parameters['N'],
                                          parameters['sigma'],
                                          parameters['eta'],
                                          parameters['T'],
                                          parameters['D'],
                                          parameters['RHO'],   # desired rho
                                          parameters['USE_BFT_CALC'])
    print(f"A_min = {A_min}")

    A = A_min

    if rho_max < parameters['RHO']:
        parameters['RHO'] = rho_max

    parameters["A"] = A
    clock.toc("Setup_Graph")

    # calculate optimized rho
    RHO = parameters['RHO']
    parameters['Q'] = parameters['N'] // RHO
    parameters['A_min_X'] = A // A_min
    parameters['T_c'] = t_c   # corruption threshold
    parameters['T_r'] = t_r   # reconstruction threshold
    parameters['U'] = t_r   # reconstruction threshold alias

    # check if simulated dropouts exceed the limit
    if int(parameters['drop_frac'] * A) > (A - t_r):
        raise ValueError('Too many dropout committee members!')

    # set model size if training
    if parameters['USE_TRAINING']:
        parameters['M'], model_shape = train_utils.get_model_size(parameters)

    print("ALL PARAMETERS:\n", parameters)

    if SHARING_SCHEME == "packed":
        print(f"Packed scheme configuration: rho={RHO}")

    if SHARING_SCHEME == "packed":
        codec = PackedCodec(rho=RHO)
        codec_name = f"Setup_PackedCodec"
    elif SHARING_SCHEME == "lcc":
        codec = LagrangeCodec(False)
        codec_name = "Setup_LCC"
    else:
        raise ValueError(
            f"Unknown sharing scheme: {SHARING_SCHEME}. Use 'lcc' or 'packed'"
        )

    N = parameters["N"]
    U = parameters['U']
    codec.create_codec(A, U, q_quantizer.field_prime_number)
    clock.toc(codec_name)

    print(f"N = {N}, M = {parameters['M']}, A = {parameters['A']}")
    print(f"q = {q_quantizer.field_prime_number}, p = {p_quantizer.field_prime_number}")

    # Generate the PRG matrix A once for all operations
    print('Generating A matrix...')
    A_matrix = PRG.Gen(
        parameters["lambda_key_dim"],
        parameters["M"],
        q_quantizer.field_prime_number,
        matrix_seed=42,
    )
    A_matrix_client = A_matrix

    clock.toc("Setup_A_matrix")

    if parameters['USE_DISK']:
        temp_member = Member(parameters, codec, q_quantizer, p_quantizer, m_quantizer)
        temp_member._save_A_matrix(A_matrix)
        temp_member._save_codec()
        A_matrix_client = None
        clock.tic()

    # initialize clients
    if num_proc > 0:
        inputs = [(parameters, codec, q_quantizer, p_quantizer, m_quantizer, i) for i in range(N)]
        with NoDaemonPool(num_proc) as p:
            clients = list(tqdm(p.imap(create_client, inputs),
                                    total=len(inputs), desc='Creating client objects'))
            p.close()
            p.join()
    else:
        clients = [
            Client(parameters, codec, q_quantizer, p_quantizer, m_quantizer, i) \
                                for i in tqdm(range(N), desc='Creating client objects')
        ]

    start_time = time.time()

    tmp_drop_member = common.MemberBase(parameters)

    metrics = {'round':[], 'loss':[], 'acc':[]}
    for rnd in range(parameters['TRAINING_ROUNDS']):

        # define dropout clients per round
        clients = tmp_drop_member._mark_dropped_clients(clients)

        # === STEP 1: CREATE CLIENTS AND MASK MODELS ===
        clock.tic()

        masked_models_dict = {}
        client_shares_dict = {}

        if parameters['USE_SECURITY']:

            if num_proc > 0:

                inputs = [(c, A_matrix_client) for c in clients]
                with NoDaemonPool(num_proc) as p:
                    rets = list(tqdm(p.imap(client_worker, inputs), total=len(inputs), desc="Clients computation"))
                    p.close()
                    p.join()

                for i, r in enumerate(rets):
                    _, masked_models_dict[i], client_shares_dict[i], elt, comtime = r
                    clock.add("Client_Down", comtime['down'])
                    clock.add("Client_Proc", elt)
                    clock.add("Client_Up", comtime['up'])
            else:
                # process clients sequentially
                for i, client in enumerate(tqdm(clients, desc="Clients")):
                    # Mask model and share keys
                    masked_models_dict[i], t1 = client.mask_model(A_matrix_client)
                    client_shares_dict[i], t2, t3 = client.share_key()
                    clock.add("Client_Down", t3['down'])
                    clock.add("Client_Proc", (t1+t2))
                    clock.add("Client_Up", t3['up'])

            # === STEP 2: COMMITTEE MEMBERS AGGREGATE THEIR SHARES ===
            clock.tic()
            committee_aggregated_shares_dict = {}

            # Each committee member aggregates only the shares sent to them
            for committee_idx in tqdm(range(parameters["A"]), desc="Committee"):
                committee = Committee(parameters, codec, q_quantizer, p_quantizer, m_quantizer)

                # Extract shares for this specific committee member from all clients
                committee_specific_shares = {}
                for client_id, client_shares in client_shares_dict.items():
                    if client_shares is None:   # dropout client
                        continue
                    # Each client's shares is a matrix where row i corresponds to committee member i
                    committee_specific_shares[client_id] = client_shares[
                        committee_idx : committee_idx + 1, :
                    ]

                # Aggregate shares for this committee member
                committee_aggregated_shares, elt, comtime = committee.aggregate_shares(
                    committee_specific_shares
                )
                committee_aggregated_shares_dict[f"committee_{committee_idx}"] = (
                    committee_aggregated_shares
                )

                # clock.toc("Committee")
                clock.add("Committee_Down", comtime['down'])
                clock.add("Committee_Proc", elt)
                clock.add("Committee_Up", comtime['up'])

            client_shares_dict = None

            # === STEP 3: SERVER RECONSTRUCTS SEED (WITH DROPOUTS) ===

            server = Server(parameters, codec, q_quantizer, p_quantizer, m_quantizer)

            # Simulate committee dropouts
            dropout_count = 0
            # active_committee_indices = np.arange(dropout_count, parameters["A"])
            active_committee_indices = np.arange(dropout_count, U)  # least needed

            clock.tic()
            reconstructed_seed = server.reconstruct(
                committee_aggregated_shares_dict, active_committee_indices
            )

            # === STEP 4: SERVER RECOVERS MODEL ===
            recovered_model = server.recover_model(
                reconstructed_seed, masked_models_dict, A_matrix
            )

            clock.toc("Server", parameters['k_comp'])

            print('DISK LOAD FRACTION:', (server.disk_timing * parameters['k_comp']) / clock.stats["Server"][0])

            # subtract disk loading time from server's timings
            clock.stats["Server"][0] -= (server.disk_timing * parameters['k_comp'])
        
        else:
            # FL fitting
            server = Server(parameters, codec, q_quantizer, p_quantizer, m_quantizer)
            server.clk = clock
            server.fit_clients(clients)

        clock.tic()   # reset timer for plaintext FL aggregation
        true_aggregated_model = None
        for c in tqdm(clients, desc='Aggregating client models for true_aggregated_model'):
            if c.dropped:
                continue
            m = c.get_model()
            if true_aggregated_model is None:
                true_aggregated_model = m
            else:
                true_aggregated_model += m
            del m
        if true_aggregated_model.dtype != np.float32 and true_aggregated_model.dtype != np.float64:
            true_aggregated_model = true_aggregated_model % p_quantizer.field_prime_number

        # plaintext server's aggregation time
        if not parameters['USE_SECURITY']:
            clock.toc("Server", parameters['k_comp'])

        clock.add('Wall_Clock_Total', time.time() - start_time)

        res1 = clock.report_stats()

        if parameters['USE_SECURITY']:
            recovered_model = np.reshape(recovered_model, (len(recovered_model),1))
            errors = np.abs(recovered_model - true_aggregated_model)
            print(f"Mean error = {errors.mean()}")
            if np.all(errors < 10000):
                print("ALL GOOD: Recovered model matches true aggregated model")
            else:
                print("ERROR: Recovered model does NOT match true aggregated model")
                print("\nRECOVERED MODEL:", recovered_model[:10])
                print("TRUE AGGREGATED MODEL:", true_aggregated_model[:10])

        if parameters['USE_TRAINING']:
            if not parameters['USE_SECURITY']:
                recovered_model = true_aggregated_model
            loss, acc = train_utils.test_and_save_global_model(
                recovered_model, model_shape, m_quantizer, parameters
            )
            print('Round:', rnd, 'Accuracy:', acc)
            metrics['round'].append(rnd)
            metrics['loss'].append(loss)
            metrics['acc'].append(acc)

            # append timings
            for k,v in res1.items():
                if k not in metrics:
                    metrics[k] = [v]
                else:
                    metrics[k].append(v)

        else:
            metrics = None
            break

    return res1, metrics


def run_simulation(log_name):
    name_val_pairs = common.get_var_params()
    sub_ks = ['M', 'N', 'D', 'T', 'P', 'F', 'RHO', 'A', 'T_c', 'T_r', 'Q', 'D_KBPS', 'U_KBPS', 'k_comp']
    common.run_simulation(log_name, init_parameters, sub_ks, name_val_pairs, start_simulation)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    init_parameters['name'] = 'OPA'
    common.run_with_temp_folder(init_parameters, run_simulation)
