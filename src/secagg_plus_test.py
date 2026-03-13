#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================
import numpy as np
import time
import random
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import copy

from pathlib import Path
from typing import Any, List, Tuple

import utils.quantization as quantization
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key,
)
import common
from constants import init_parameters, num_proc
from utils.harary import HararyGraphGenerator
from secret_sharing.shamir import Shamir
from utils.diffie_hellman import DiffieHellman

USE_SECAGG_PLUS = True     # True for SecAgg+, False for SecAgg


def transpose_shares(
        shares: List[List[Tuple[int, bytes]]]
) -> List[List[Tuple[int, bytes]]]:
    """
    Function that receives a List of Users each containing a List of chunks and transpose it to a
    List of Chunks each containing a List of all Users chunk for the specific chunk index.

    input = List = [
        [user1.chunk1, user1.chunk2],
        [user2.chunk1, user2.chunk2]
    ]

    output = List = [
        [user1.chunk1, user2.chunk1],
        [user1.chunk2, user2.chunk2]
    ]

    Args:
    shares: List[List[Tuple[int, bytes]]]
        List of lists each inner list containing chunks of a user.

    Returns:
    transposed_shares: List[List[Tuple[int, bytes]]]
        List of lists each inner list containing chunks of all users for this index.
    """
    num_clients = len(shares)
    num_chunks = len(shares[0])

    transposed_shares: List[Any] = []
    for chunk_index in range(num_chunks):
        chunk_list: List[Any] = []
        for client_index in range(num_clients):
            chunk_list.append(shares[client_index][chunk_index])
        transposed_shares.append(chunk_list)

    return transposed_shares


class Member(common.MemberBase):
    def __init__(self, parameters, helper_objects):
        super().__init__(parameters)
        quantizer, dh_prms_bytes, shamir, neighborhoods = helper_objects
        self.quantizer = quantizer
        self.shamir = shamir
        self.dh_prms_bytes = dh_prms_bytes
        self.modulus = self.quantizer.field_prime_number
        self.neighborhoods = neighborhoods
        self.key_agreement = DiffieHellman(dh_prms_bytes)
        self.all_keys = None

    def _create_mask(self, seed, n):
        random.seed(seed)
        mask = [random.randint(0, self.modulus - 1)
                        for _ in range(n)]
        mask = np.array(mask, dtype=object)
        mask = np.reshape(mask, (len(mask), 1))
        return mask

    def _temp_keys_file(self):
        fn = 'all_keys.pkl'
        path = Path(self.temp_dir) / Path(fn)
        return path

    def _save_all_keys(self):
        with open(self._temp_keys_file(), 'wb') as f:
            pickle.dump(self.all_keys, f)
            self.all_keys = None

    def _load_all_keys(self):
        with open(self._temp_keys_file(), 'rb') as f:
            data = pickle.load(f)
        return data


class Client(Member):
    def __init__(self, parameters, helper_objects, cid):
        super().__init__(parameters, helper_objects)
        self.cid = cid
        self._prepare_model()

        self.pkeys = {}     # placeholder
        self.bu_bytes = b'0x00'
        self.neighbor_shares = {}

    def _temp_file(self):
        fn = str(self.cid) + '.pkl'
        path = Path(self.temp_dir) / Path(fn)
        return path

    def _save_state(self):
        data = (self.pkeys, self.bu_bytes, self.neighbor_shares)
        with open(self._temp_file(), 'wb') as f:
            pickle.dump(data, f)
        self.pkeys, self.bu_bytes, self.neighbor_shares = None, None, None

    def _load_state(self):
        with open(self._temp_file(), 'rb') as f:
            data = pickle.load(f)
        self.pkeys, self.bu_bytes, self.neighbor_shares = data

    def _prepare_model(self):
        m = np.random.random(size=(self.parameters['M'], 1)).astype(object)
        self.model = self.quantizer.quantize(m)
        self._save_model()

    def _get_derived_key(self, client_id):
        key_name = 'Mask'
        self.key_agreement.deserialize()
        clients_public_key = load_pem_public_key(self.pkeys[client_id])
        self.key_agreement.generate_shared_key(
            clients_public_key,
            key_name,
            client_id,
        )
        derived_key = self.key_agreement.derive_keys(
            key_name,
            client_id,
        )
        self.key_agreement.serialize()
        return derived_key

    def get_public_mask_key(self):
        tstart = time.time()
        self.key_agreement.generate_private_key('Mask')
        self.key_agreement.generate_public_key('Mask')
        self.key_agreement.serialize()
        self.elt = time.time() - tstart
        pkey = self.key_agreement.get_public_key('Mask')
        self.comtime['up'] = self._get_com_time(pkey, 'u')
        return pkey

    def create_secret_shares(self, keys):
        keys = self._load_all_keys()
        self.comtime['down'] = self._get_com_time(keys, 'd')

        tstart = time.time()
        self.pkeys = keys

        mask_key_shares = self.shamir.create_shares(
            self.key_agreement.get_private_key('Mask')
        )

        self.bu = int(random.randint(0, self.modulus - 1))
        self.bu_bytes = self.bu.to_bytes(16, "little")
        bu_shares = self.shamir.create_shares(self.bu_bytes)

        shares = {}
        for i, cid in enumerate(self.neighborhoods[self.cid]):
            sh = self.encrypt((self.cid, cid, bu_shares[i], mask_key_shares[i]))
            shares[cid] = sh

        self.elt = time.time() - tstart

        # leave save-load out of timings, is is for simulation only
        self._save_state()
        self.comtime['up'] = self._get_com_time(shares, 'u')

        return shares

    def get_masked_model(self, shares):
        model = self._load_model()
        self.comtime['down'] = self._get_com_time(shares, 'd')
        
        # leave save-load out of timings, is is for simulation only
        self._load_state()

        tstart = time.time()
        self.neighbor_shares = {}
        derived_keys = {}
        for _,sh in shares:
            sid, cid, bu_shares, mask_key_shares = self.decrypt(sh)
            if cid != self.cid:
                raise ValueError('Wrong recipient ID!')
            self.neighbor_shares[sid] = (bu_shares, mask_key_shares)
            derived_keys[sid] = self._get_derived_key(sid)

        n = len(model)
        private_mask = self._create_mask(self.bu_bytes, n)
        masked_model = np.mod(model + private_mask, self.modulus)
        for i,dk in derived_keys.items():
            pairwise_mask = self._create_mask(dk, n)
            if self.cid < i:
                masked_model = np.mod(masked_model + pairwise_mask, self.modulus)
            else:
                masked_model = np.mod(masked_model - pairwise_mask, self.modulus)
        self.elt = time.time() - tstart

        # leave save-load out of timings, is is for simulation only
        self._save_state()
        self.comtime['up'] = self._get_com_time(masked_model, 'u')

        return masked_model

    def request_shares(self, dropouts):
        self.comtime['down'] = self._get_com_time(dropouts, 'd')
        # leave save-load out of timings, is is for simulation only
        self._load_state()

        tstart = time.time()
        dset = set(dropouts)
        shares = {}
        for i,sh in self.neighbor_shares.items():
            bu, pair =  sh
            if i in dset:
                shares[i] = pair
            else:
                shares[i] = bu
        self.elt = time.time() - tstart
        self.comtime['up'] = self._get_com_time(shares, 'u')

        return shares

    def encrypt(self, msg):
        # simulate encryption, simply serialize with pickle
        return pickle.dumps(msg)

    def decrypt(self, cipher):
        # simulate encryption, simply deserialize with pickle
        dec = pickle.loads(cipher)
        return dec


def client_get_keys_worker(input_args):
    c = input_args
    return (c.get_public_mask_key(), c.cid, c.elt, c.comtime)

def client_shares_worker(input_args):
    c, k = input_args
    return (c.create_secret_shares(k), c.cid, c.elt, c.comtime)

def client_model_worker(input_args):
    c, sh = input_args
    return (c.get_masked_model(sh), c.cid, c.elt, c.comtime)

def client_request_shares_worker(input_args):
    c, d = input_args
    return (c.request_shares(d), c.cid, c.elt, c.comtime)


class Server(Member):
    def __init__(self, parameters, helper_objects):
        super().__init__(parameters, helper_objects)
        self.clk = None
        self.all_keys = {}

    def exchange_mask_keys(self, clients):
        mask_keys = [client_get_keys_worker(c)
                     for c in tqdm(clients, desc='Clients-GetKeys')]
        self.all_keys = {}
        for m,i,t, comtime in mask_keys:
            self.all_keys[i] = m
            self.clk.add('Client_GetKeys_Proc', t)
            self.clk.add('Client_GetKeys_Up', comtime['up'])
        self._save_all_keys()

    def create_secret_shares(self, clients):
        self.neighbors_shares = {}   # shares for the neighbors

        inputs = [(c, self.all_keys) for c in clients]
        if num_proc > 0:
            with Pool(num_proc) as p:
                shares = list(tqdm(p.imap(client_shares_worker, inputs),
                                    total=len(inputs), desc='Clients-CreateShares'))
                p.close()
                p.join()
        else:
            shares = [client_shares_worker(ii)
                      for ii in tqdm(inputs, desc='Clients-CreateShares')]

        for sh in shares:
            shares, cid, elt, comtime = sh
            for i,sh in shares.items():
                msg = (cid, sh)    # msg with sender's id

                if i in self.neighbors_shares:
                    self.neighbors_shares[i].append(msg)
                else:
                    self.neighbors_shares[i] = [msg]
            self.clk.add('Client_CreateShares_Down', comtime['down'])
            self.clk.add('Client_CreateShares_Proc', elt)
            self.clk.add('Client_CreateShares_Up', comtime['up'])

    def get_masked_models(self, clients):
        inputs = [(c, self.neighbors_shares[c.cid]) for c in clients]
        if num_proc > 0:
            with Pool(num_proc) as p:
                masked_models = list(tqdm(p.imap(client_model_worker, inputs),
                                    total=len(inputs), desc='Clients-Mask'))
                p.close()
                p.join()
        else:
            masked_models = [client_model_worker(ii)
                        for ii in tqdm(inputs, desc='Clients-Mask')]

        for mm in masked_models:
            m, cid, elt, comtime = mm
            self.clk.add('Client_Mask_Down', comtime['down'])
            self.clk.add('Client_Mask_Proc', elt)
            self.clk.add('Client_Mask_Up', comtime['up'])

        self.masked_models = masked_models

    def request_shares(self, clients):
        dropouts = []   # dropout IDs
        inputs = [(c, dropouts) for c in clients]
        if num_proc > 0:
            with Pool(num_proc) as p:
                shares = list(tqdm(p.imap(client_request_shares_worker, inputs),
                                    total=len(inputs), desc='Clients-SendShares'))
                p.close()
                p.join()
        else:
            shares = [client_request_shares_worker(ii)
                        for ii in tqdm(inputs, desc='Clients-SendShares')]

        dset = set(dropouts)
        bu_shares = {}
        pairwise_shares = {}
        for sh in shares:
            m, cid, elt, comtime = sh        # id = this client's response
            for nid,s in m.items():     # nid = neighbor id
                if nid in dset:
                    if nid in pairwise_shares:
                        pairwise_shares[nid].append(s)
                    else:
                        pairwise_shares[nid] = [s]
                else:
                    if nid in bu_shares:
                        bu_shares[nid].append(s)
                    else:
                        bu_shares[nid] = [s]
            self.clk.add('Client_SendShares_Down', comtime['down'])
            self.clk.add('Client_SendShares_Proc', elt)
            self.clk.add('Client_SendShares_Up', comtime['up'])
        self.bu_shares = bu_shares
        self.pairwise_shares = pairwise_shares

    def reconstruct(self):
        bu_vals = []
        for i,bs in tqdm(self.bu_shares.items(), desc='Reconstruct'):
            ts = transpose_shares(bs)
            bb = self.shamir.combine_shares(ts)
            bu = int.from_bytes(bb, 'little')
            bu_vals.append(bu)

        # aggregate
        aggregated_model = None
        for m,cid,_,_ in tqdm(self.masked_models, desc='Aggregate'):
            if aggregated_model is None:
                aggregated_model = m
            else:
                aggregated_model = np.mod(aggregated_model + m,
                                           self.modulus)

        # unmasking
        for bu in tqdm(bu_vals, desc='Unmask'):
            n = len(aggregated_model)
            private_mask = self._create_mask(bu.to_bytes(16, "little"), n)
            aggregated_model = np.mod(aggregated_model - private_mask,
                                        self.modulus)
        self.aggregated_model = aggregated_model
        return self.aggregated_model

def create_client(input_args):
    parameters, helper_objects, i = input_args
    return Client(parameters, helper_objects, i)


def start_simulation(parameters):

    temp_dir = parameters['Temp']
    outdir = Path(temp_dir)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    clk = common.Clock()
    clk.tic()

    # keep original parameters as const starting point
    init_params = copy.deepcopy(init_parameters)

    ### Setup
    parameters['P'] = int(init_params["P"] - np.ceil(np.log2(parameters["N"]))) # to match OPA
    print(f"plaintext_field_bits = {parameters['P']}")

    print("ALL PARAMETERS:\n", parameters)

    quantizer = quantization.Quantizer(clip_value=2.0,
                                              clients_scale_factor=parameters['N'],
                                              num_bits=parameters['P'])
    clk.toc('Setup_Quantizer')

    secagg_plus_graph = HararyGraphGenerator(
                            nodes=range(0, parameters['N']),
                            security_parameter=parameters['sigma'],
                            correctness_parameter=parameters['eta'],
                            max_corrupt_fraction=parameters['T'],
                            max_dropout_fraction=parameters['D'])
    
    if not USE_SECAGG_PLUS:
        # set complete graph
        secagg_plus_graph.degree_k = parameters['N'] - 1
        secagg_plus_graph.threshold = int(parameters['N'] * (1 - parameters['D']))

    k,t = secagg_plus_graph.degree_k, secagg_plus_graph.threshold
    if t is None:
        raise ValueError('Threshold error!')
    neighborhoods = {}
    graph = secagg_plus_graph.generate_permuted_graph()
    for node in graph.nodes:
        neighborhoods[node] = list(graph.neighbors(node))

    test_len = len(next(iter(neighborhoods.values())))
    for _,ns in neighborhoods.items():
        if test_len != len(ns):
            raise ValueError('Wrong neighborhoods size')

    print('Graph Degree:', k)
    print('Threshold:', t)

    clk.toc('Setup_Graph')

    # cache the DH parameters as it may be a slow process
    dh_prms_bytes = b'-----BEGIN DH PARAMETERS-----\nMIIBCAKCAQEAvP4x33fTOS+2Wx7qO52iTDDmB7LT3Uo9xCXhKiBXLnYvRI0lP6Ee\nbcVX7c1haBCQEUDA2ISwiZZNzsrDbTrrZip4p1YdjuLZGaj5A3p8vXlucUrQNKZl\ngrURPgI9sDCFAzjkCejGd5i6El7H68UiKk+gsVmy0ISPiEx3mkzx6FzNYFLyfDVD\nB7yhvOg+VERkx4Cu7ma6dKNcwI4DF+fbGr7zhktcccVUJqKmBMcev3yUpLnzPpW9\nKGPWf1Ecyqd7teB9M+y2DlKsiNQ0l67I+IhP7tDrU697AQpkSj4gNYluTPkl1Tu/\niSP6ajEZ8f/Np36TKTUVMOh2wEdZhpnsrwIBAg==\n-----END DH PARAMETERS-----\n'

    shamir = Shamir(t, k)
    clk.toc('Shamir')

    helper_objects = (quantizer, dh_prms_bytes, shamir, neighborhoods)

    # ### initialize clients, aggregators
    ids = list(range(parameters['N']))

    # create clients in this process where all inputs are the same objects
    clients = [create_client((parameters, helper_objects, i))
            for i in tqdm(ids, desc='Initialization')]

    start_time = time.time()

    ### initialize server
    server = Server(parameters, helper_objects)
    server.clk = clk

    clk.tic()
    ### Step 1: exchange keys
    server.exchange_mask_keys(clients)

    ### Step 2: create secret shares
    server.create_secret_shares(clients)

    ### Step 3: fit & get masked model
    server.get_masked_models(clients)

    ### Step 4: request shares (private or pairwise)
    server.request_shares(clients)

    clk.tic()
    ### Step 5: reconstruct
    server.reconstruct()

    clk.toc('Server', parameters['k_comp'])

    clk.add('Wall_Clock_Total', time.time() - start_time)

    res1 = clk.report_stats()

    # correctness check, for quantized models to be accurate
    aggregated_model = None
    for c in clients:
        m = c.get_model()
        if aggregated_model is None:
            aggregated_model = m
        else:
            aggregated_model = np.mod(aggregated_model + m,
                                      quantizer.field_prime_number)
    print('Correctness:', np.all(aggregated_model == server.aggregated_model))

    return res1, {}


def run_simulation(log_name):
    name_val_pairs = common.get_var_params()
    sub_ks = ['M', 'N', 'D', 'T', 'P', 'D_KBPS', 'U_KBPS', 'k_comp']
    common.run_simulation(log_name, init_parameters, sub_ks, name_val_pairs, start_simulation)


if __name__ == '__main__':
    init_parameters['name'] = 'SecAggPlus'
    common.run_with_temp_folder(init_parameters, run_simulation)
