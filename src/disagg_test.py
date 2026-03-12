#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================
import numpy as np
import time
import random
from tqdm import tqdm
import copy
import pickle
from pathlib import Path

import utils.quantization as quantization
from secret_sharing.lcc_codec import LagrangeCodec
import common
from constants import init_parameters, num_proc

import utils.train_utils as train_utils
import utils.committee as cmt

import multiprocessing as mp
from multiprocessing.pool import Pool

# switch to LightSecAgg simulation
USE_LIGHT_SEC_AGG = False

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


class Member(common.MemberBase):
    def __init__(self, parameters, lcc_codec, quantizer):
        super().__init__(parameters)
        self.lcc_codec = lcc_codec
        self.quantizer = quantizer
        ids = list(range(self.parameters['N']))
        self.aggregators = random.sample(ids, self.parameters['A'])
        self.modulus = parameters['SSP']    # modulus for secret sharing
        self.lock = None
        self.elt_agg = {}

    def _get_params(self):
        A = self.parameters['A']
        U = self.parameters['U']

        T = self.parameters['T_c']

        # padding d
        d = self.parameters['M']
        n = d // (U - T)
        if d % (U - T) != 0:
            n += 1
        d = n * (U - T)
        p = self.modulus
        k = d - self.parameters['M']
        return A, U, T, d, p, k

    def _save_codec(self):
        if not self.parameters['USE_DISK']: return
        fname = f'{self.temp_dir}/lcc_codec.npy'
        np.save(fname, self.lcc_codec.encoder_matrix)
        self.lcc_codec.encoder_matrix = None

    def _load_codec(self):
        if not self.parameters['USE_DISK']: return
        fname = f'{self.temp_dir}/lcc_codec.npy'
        A = np.load(fname, allow_pickle=True)
        return A

    def _append_aggregator_share(self, a_id, sh):
        if self.dropped:    # dropped client
            return
        if self.lock is None:
            self._append_aggregator_share_locked(a_id, sh)
        else:
            with self.lock:
                self._append_aggregator_share_locked(a_id, sh)

    def _clear_aggregator_shares(self):
        for a_id in self.aggregators:
            fname = f'{self.temp_dir}/{a_id}_shares.pkl'
            if Path(fname).exists():
                Path(fname).unlink()

    def _append_aggregator_share_locked(self, a_id, sh):
        fname = f'{self.temp_dir}/{a_id}_shares.pkl'
        if not self.parameters['USE_CLIENT_AGGREGATION'] or not Path(fname).exists():
            with open(fname, 'ab') as f:
                pickle.dump(sh, f)
            self.elt_agg[a_id] = 0
        else:
            with open(fname, 'rb') as f:
                (_, decip) = pickle.load(f)

            shares = self._get_from_fixed_byte_size( [(0, (0, decip))] )
            share = shares[0][1][1]

            shares = self._get_from_fixed_byte_size( [(0, sh)] )
            shsh = shares[0][1][1]

            tstart = time.time()
            share = np.mod(shsh + share, self.modulus)
            self.elt_agg[a_id] = time.time() - tstart

            shares = self._set_fixed_byte_size( {0: (0, share)} )
            share = shares[0]

            with open(fname, 'wb') as f:
                pickle.dump(share, f)

    def _read_aggregator_shares(self, a_id):
        fname = f'{self.temp_dir}/{a_id}_shares.pkl'
        shares = []
        N = self.parameters['N']
        if self.parameters['USE_CLIENT_AGGREGATION']:
            N = 1
        with open(fname, 'rb') as f:
            for _ in range(N):
                (c_id, sh) = pickle.load(f)
                sh = (c_id, (c_id, sh))
                shares.append(sh)
        return shares


class Client(Member):
    def __init__(self, parameters, lcc_codec, quantizer, cid):
        super().__init__(parameters, lcc_codec, quantizer)
        self.cid = cid
        if not self.parameters['USE_TRAINING']:
            self._prepare_model()

    def _prepare_model(self):
        # random model
        m = np.random.random(size=(self.parameters['M'], 1)).astype(object)
        self.model = self.quantizer.quantize(m)
        self._save_model()

    def _pad_array(self, array):
        A, U, T, d, p, k = self._get_params()

        if k != 0:
            rand_nums = [random.randint(0, p-1) for _ in range(k)]
            rand_nums = np.array(rand_nums, dtype=object)
            rand_nums = np.reshape(rand_nums, (k, 1))
            padded_array = np.concatenate([array, rand_nums],axis=0)
        else:
            padded_array = array
        return A, U, T, d, p, k, padded_array

    def fit(self):
        train_utils.set_params(self.parameters)
        gm, sh = train_utils.load_global_model()
        self.comtime['down'] = self._get_com_time(gm, 'd', 'N')
        self.comtime['up'] = self._get_com_time(gm, 'u', 'N')

        tstart = time.time()
        self.train_and_quantize_model()
        self.elt = time.time() - tstart

        self._save_model()

    def get_shares(self):
        if not self.parameters['USE_TRAINING']:
            if USE_LIGHT_SEC_AGG:
                model = self._load_mask()                
            else:
                model = self._load_model()
            self.comtime['down'] = self._get_com_time(model, 'd', 'N')
        else:
            train_utils.set_params(self.parameters)
            gm, sh = train_utils.load_global_model()
            self.comtime['down'] = self._get_com_time(gm, 'd', 'N')

        if self.parameters['USE_DISK']:
            self.lcc_codec.encoder_matrix = self._load_codec()

        tstart = time.time()

        if self.parameters['USE_TRAINING']:
            self.train_and_quantize_model()
            model = self.model

        A, U, T, d, p, k, padded_model = self._pad_array(model)

        parts = self.lcc_codec.encode(d, A, U, T, p, padded_model)
        shares = {}
        
        if self.parameters['USE_DISK']:
            self.lcc_codec.encoder_matrix = None

        for i,part in enumerate(parts):
            a_id = self.aggregators[i]
            sh = (self.cid, part)
            shares[a_id] = sh

        self.elt = time.time() - tstart

        shares = self._set_fixed_byte_size(shares)
        self.comtime['up'] = self._get_com_time(shares, 'u', 'N')

        if self.parameters['USE_TRAINING']:
            self._save_model()

        if self.parameters['USE_DISK']:
            for a_id,sh in shares.items():
                self._append_aggregator_share(a_id, sh)
                shares[a_id] = (self.cid, None)

        if self.dropped:
            return None

        return shares

    def aggregate(self, shares):

        if self.parameters['USE_DISK']:
            shares = self._read_aggregator_shares(self.cid)

        shares = self._get_from_fixed_byte_size(shares)
        self.comtime['down'] = self._get_com_time(shares, 'd', 'A')

        if self.parameters['USE_CLIENT_AGGREGATION']:
            self.comtime['down'] = self.comtime['down'] * self.parameters['N']
            time.sleep(self.comtime['down'])
            sum_share = shares[0][1][1]
            self.elt = 0
        else:
            tstart = time.time()
            sum_share = None
            for cid,cipher in shares:
                (sid, share) = cipher
                if cid != sid:
                    raise ValueError("Error in sender's ID")
                if sum_share is None:
                    sum_share = share
                else:
                    sum_share = np.mod(sum_share + share, self.modulus)

            self.elt = time.time() - tstart

        self.comtime['up'] = self._get_com_time(sum_share, 'u', 'A')
        return sum_share

    def _set_fixed_byte_size(self, shares):
        for k in shares:
            cid,arr = shares[k]
            p = 2 ** 64
            H = (arr // p).astype(np.uint64)
            L = (arr % p).astype(np.uint64)
            T = np.array((H,L))
            shares[k] = (cid,T)
        return shares

    def _get_from_fixed_byte_size(self, shares):
        for i,sh in enumerate(shares):
            cid = sh[0]
            arr = sh[1][1]
            H = arr[0].astype(object)
            L = arr[1].astype(object)
            T = (H << 64) + L
            shares[i] = (cid, (cid, T))
        return shares


class LightSecAggClient(Client):
    def __init__(self, parameters, lcc_codec, quantizer, cid):
        super().__init__(parameters, lcc_codec, quantizer, cid)
        if not self.parameters['USE_TRAINING']:
            self._prepare_mask()

    def _prepare_mask(self):
        self.mask = [random.randint(0, self.modulus - 1)
                    for _ in range(self.parameters['M'])]
        self.mask = np.array(self.mask, dtype=object)
        self.mask = np.reshape(self.mask, (len(self.mask), 1))
        self._save_mask()

    def _save_mask(self):
        if not self.parameters['USE_DISK']: return
        fname = f'{self.temp_dir}/{self.cid}_mask.npy'
        np.save(fname, self.mask)
        self.mask = None

    def _load_mask(self):
        if not self.parameters['USE_DISK']: return self.mask
        fname = f'{self.temp_dir}/{self.cid}_mask.npy'
        A = np.load(fname, allow_pickle=True)
        return A

    def get_masked_model(self):
        model = self._load_model()
        mask = self._load_mask()
        tstart = time.time()
        masked_model = np.mod(model + mask, self.modulus)
        self.elt = time.time() - tstart
        self.comtime['up'] = self._get_com_time(masked_model, 'u', 'N')
        return masked_model


def client_worker(c):
    shares = c.get_shares()
    return (shares, c.cid, c.elt, c.comtime, c.elt_agg)


def client_fit_worker(c):
    c.fit()
    return (c.cid, c.elt, c.comtime)


def client_model_worker(c):
    mm = c.get_masked_model()
    return (mm, c.cid, c.elt, c.comtime)


def aggregator_worker(input_args):
    c, sh = input_args
    agg = c.aggregate(sh)
    return (agg, c.cid, c.elt, c.comtime)


class Server(Member):
    def __init__(self, parameters, lcc_codec, quantizer):
        super().__init__(parameters, lcc_codec, quantizer)
        self.clk = None

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

    def request_shares(self, clients):
        self.a_shares = {}   # shares for the aggregators
        self.clk.tic()

        if num_proc > 0:
            with mp.Manager() as manager:
                lock = manager.Lock()
                for i in range(len(clients)):
                    clients[i].lock = lock
                with NoDaemonPool(num_proc) as p:
                    all_shares = list(tqdm(p.imap(client_worker, clients),
                                        total=len(clients), desc='Clients'))
                    p.close()
                    p.join()
                for i in range(len(clients)):
                    clients[i].lock = None
        else:
            all_shares = [client_worker(c) for c in tqdm(clients, desc='Clients')]

        elt_agg_sum = {}    # store timing sums per aggregator
        for sh in all_shares:
            shares, cid, elt, comtime, elt_agg = sh
            if shares is None:  # dropped client
                continue
            for i,sh in shares.items():
                msg = (cid, sh)    # msg with sender's id
                if i in self.a_shares:
                    self.a_shares[i].append(msg)
                else:
                    self.a_shares[i] = [msg]
            self.clk.add('Client_Down', comtime['down'])
            self.clk.add('Client_Proc', elt)
            self.clk.add('Client_Up', comtime['up'])

            if self.parameters['USE_CLIENT_AGGREGATION']:
                if not elt_agg_sum:
                    elt_agg_sum = elt_agg
                else:
                    elt_agg_sum = {k: (elt_agg_sum[k] + v) for k,v in elt_agg.items()}

        if self.parameters['USE_CLIENT_AGGREGATION']:
            for _,v in elt_agg_sum.items():
                self.clk.add('Aggregator_Proc', v)

    def request_aggregated_shares(self, clients):
        self.aggregated_parts = None
        self.clk.tic()

        if num_proc > 0:
            inputs = [(clients[cid], self.a_shares[cid]) for cid in self.aggregators]
            with NoDaemonPool(num_proc) as p:
                all_aggs = list(tqdm(p.imap(aggregator_worker, inputs),
                                    total=len(inputs), desc='Aggregators'))
                p.close()
                p.join()
        else:
            all_aggs = [aggregator_worker((clients[cid], self.a_shares[cid]))
                        for cid in tqdm(self.aggregators, desc='Aggregators')]

        self.a_shares = None

        for aggs in all_aggs:
            agg, cid, elt, comtime = aggs
            agg = np.reshape(agg, (1,len(agg)))
            if self.aggregated_parts is None:
                self.aggregated_parts = agg
            else:
                self.aggregated_parts = np.concatenate(
                    (self.aggregated_parts, agg), axis=0)
            if USE_LIGHT_SEC_AGG:
                self.clk.add('Client_Aggregate_Down', comtime['down'])
                self.clk.add('Client_Aggregate_Proc', elt)
                self.clk.add('Client_Aggregate_Up', comtime['up'])
            else:
                self.clk.add('Aggregator_Down', comtime['down'])
                if not self.parameters['USE_CLIENT_AGGREGATION']:
                    self.clk.add('Aggregator_Proc', elt)
                self.clk.add('Aggregator_Up', comtime['up'])

    def reconstruct(self):
        A, U, T, d, p, k = self._get_params()
        idxs = range(U)
        self.aggregated_parts = self.aggregated_parts[:U] # drop extra shares so we only have the minimum required U
        self.aggregated_model = self.lcc_codec.decode(d, A, U, T, p,
                                self.aggregated_parts, idxs)
        self.aggregated_model = self.aggregated_model[:self.parameters['M']]


class LightSecAggServer(Server):
    def __init__(self, parameters, lcc_codec, quantizer):
        super().__init__(parameters, lcc_codec, quantizer)

    def request_shares(self, clients):
        super().request_shares(clients)
        ### Step 1.1: get masked models
        self.request_masked_models(clients)

    def request_masked_models(self, clients):
        self.clk.tic()
        if num_proc > 0:
            
            with NoDaemonPool(num_proc) as p:
                masked_models = list(tqdm(p.imap(client_model_worker, clients),
                                    total=len(clients), desc='Clients_Mask'))
                p.close()
                p.join()
        else:
            masked_models = [client_model_worker(c)
                             for c in tqdm(clients, desc='Clients_Mask')]

        aggregated_masked_model = None
        for mm in masked_models:
            masked_model, _, elt, comtime = mm
            if aggregated_masked_model is None:
                aggregated_masked_model = mm[0]
            else:
                aggregated_masked_model = np.mod(aggregated_masked_model + masked_model,
                                                  self.modulus)

            self.clk.add('Client_Mask_Proc', elt)
            self.clk.add('Client_Mask_Up', comtime['up'])
                
        self.aggregated_masked_model = aggregated_masked_model

    def reconstruct(self):
        super().reconstruct()
        self.aggregated_mask = self.aggregated_model
        self.aggregated_model = np.mod(self.aggregated_masked_model - self.aggregated_mask,
                                       self.modulus)


if USE_LIGHT_SEC_AGG:
    Client = LightSecAggClient
    Server = LightSecAggServer


def create_client(input_args):
    parameters, lcc_codec, quantizer, i = input_args
    return Client(parameters, lcc_codec, quantizer, i)


def start_simulation(parameters):

    parameters = common.parse_parameters(parameters)

    clk = common.Clock()
    clk.tic()
    
    # keep original parameters as const starting point
    init_params = copy.deepcopy(init_parameters)

    ### Setup
    # to match opa style reduce the weight quantization to accommodate for N clients
    parameters['P'] = int(init_params["P"] - np.ceil(np.log2(parameters["N"]))) # 64 - math.ceil(math.log2(parameters['N']))
    print(f"plaintext_field_bits = {parameters['P']}")
    
    quantizer = quantization.Quantizer(clip_value=2.0,
                                              clients_scale_factor=parameters['N'],
                                              num_bits=parameters['P'])
    if parameters['USE_QUANTIZATION'] == False:
        quantizer = quantization.DummyQuantizer(clip_value=2.0,
                                                clients_scale_factor=parameters['N'],
                                                num_bits=parameters['P'])

    clk.toc('Setup_Quantizer')

    # chose committee size and threshold
    if not USE_LIGHT_SEC_AGG:
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
    elif USE_LIGHT_SEC_AGG:
        # set aggregators all the clients as LightSecAgg
        # and the thresholds as the protocol's paper
        A = parameters['N']
        A_min = A
        t_c = int(A * parameters['T'])  # LightSecAgg's T parameter, all corrupts
        D = int(A * parameters['D'])    # LightSecAgg's D parameter, all dropouts
        t_r = (A - D + t_c) // 2        # LightSecAgg's U parameter
        parameters['RHO'] = t_r - t_c   # LightSecAgg's (U-T) parameter, secret length

    # check if simulated dropouts exceed the limit
    if int(parameters['drop_frac'] * A) > (A - t_r):
        raise ValueError('Too many dropout aggregators!')

    print('Aggregators:', A)
    print('Thresholds:', t_c, t_r)
    # update parameters for aggregators (A) and target aggregators (U)
    parameters['A'] = A
    parameters['T_r'] = t_r   # reconstruction threshold
    parameters['U'] = t_r   # reconstruction threshold alias for LCC
    parameters['T_c'] = t_c   # corruption threshold
    parameters['Q'] = parameters['N'] // parameters['RHO']
    parameters['A_min_X'] = A // A_min

    # set model size if training
    if parameters['USE_TRAINING']:
        parameters['M'], model_shape = train_utils.get_model_size(parameters)

    print("ALL PARAMETERS:\n", parameters)

    clk.toc('Setup_Graph')

    lcc_codec = LagrangeCodec()
    A = parameters['A']
    U = parameters['U']

    # calc secret sharing prime number according to OPA's F field size
    ss_quantizer = quantization.Quantizer(clip_value=2.0,
                                              clients_scale_factor=parameters['N'],
                                              num_bits=parameters['F'])
    ss_modulus = ss_quantizer.field_prime_number
    lcc_codec.create_codec(A, U, ss_modulus)     # no decoder caching by default, can handle dropouts
    parameters['SSP'] = ss_modulus  # secret sharing prime

    temp_member = Member(parameters, lcc_codec, quantizer)
    temp_member._save_codec()

    clk.toc('Setup_LCC')

    ### initialize clients, aggregators
    ids = list(range(parameters['N']))
    if num_proc > 0:
        inputs = [(parameters, lcc_codec, quantizer, i) for i in ids]
        with NoDaemonPool(num_proc) as p:
            clients = list(tqdm(p.imap(create_client, inputs),
                                    total=len(inputs), desc='Creating client objects'))
            p.close()
            p.join()
    else:
        clients = [create_client((parameters, lcc_codec, quantizer, i))
                                for i in tqdm(ids, desc='Creating client objects')]

    start_time = time.time()

    ### initialize server
    server = Server(parameters, lcc_codec, quantizer)
    server.clk = clk
    
    metrics = {'round':[], 'loss':[], 'acc':[]}

    tmp_drop_member = common.MemberBase(parameters)

    for r in range(parameters['TRAINING_ROUNDS']):
        clk.tic()

        server._clear_aggregator_shares()     # clear previous round aggregator shares if exist

        # define dropout clients per round
        clients = tmp_drop_member._mark_dropped_clients(clients)

        if parameters['USE_SECURITY']:
            ### Step 1: request shares
            server.request_shares(clients)

            ### Step 2: send for aggregation
            server.request_aggregated_shares(clients)

            clk.tic()
            ### Step 3: reconstruct
            server.reconstruct()
            clk.toc('Server', parameters['k_comp'])
        else:
            # FL fitting
            server.fit_clients(clients)

        clk.tic()   # reset timer for plaintext FL aggregation
        aggregated_model = None
        for c in tqdm(clients, desc='Aggregating client models for true_aggregated_model'):
            if c.dropped:   # do not aggregate dropped clients models
                continue
            m = c.get_model()
            if aggregated_model is None:
                aggregated_model = m
            else:
                if m.dtype == np.float32 or m.dtype == np.float64:
                    aggregated_model = aggregated_model + m
                else:
                    aggregated_model = np.mod(aggregated_model + m, ss_modulus)

        # plaintext server's aggregation time
        if not parameters['USE_SECURITY']:
            clk.toc('Server', parameters['k_comp'])

        clk.add('Wall_Clock_Total', time.time() - start_time)

        res1 = clk.report_stats()

        if parameters['USE_SECURITY']:
            # correctness check, for quantized models to be accurate
            print('Check correctness...')
            print(np.all(aggregated_model == server.aggregated_model))

        if parameters['USE_TRAINING']:
            loss, acc = train_utils.test_and_save_global_model(
                aggregated_model, model_shape, quantizer, parameters
            )
            print('Round:', r, 'Accuracy:', acc)
            metrics['round'].append(r)
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
            break # if not training only run for one round

    return res1, metrics


def run_simulation(log_name):
    name_val_pairs = common.get_var_params()
    sub_ks = ['M', 'N', 'D', 'T', 'P', 'F', 'RHO', 'A', 'T_c', 'T_r', 'Q', 'D_KBPS', 'U_KBPS', 'k_comp']
    common.run_simulation(log_name, init_parameters, sub_ks, name_val_pairs, start_simulation)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    init_parameters['name'] = 'DisAgg'
    common.run_with_temp_folder(init_parameters, run_simulation)

