#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================
import tempfile
from pathlib import Path
import itertools
import sys
import numpy as np
import time
import random
import pickle
import pandas as pd
import datetime
import copy

from constants import var_params
from utils import train_utils
from utils import heterogeneity_sweep_2d as heterogeneity_calc


def get_var_params():
    '''
    Function to read the variables that will be used in the simulation
    and are defined in the structure var_params in constants.py
    
    Returns two lists one for the names and one for the values of the variables
    '''

    i = 0
    args = sys.argv
    if len(args) == 2:
        a = int(args[1])
        if a < 0 or a >= len(var_params):
            raise ValueError('Wrong argument, must be a valid list index!')
        else:
            i = a

    names = [k for k,_ in var_params[i].items()]
    lists = [v for _,v in var_params[i].items()]
    vals = list(itertools.product(*lists))
    return names, vals


def run_with_temp_folder(params, func):
    '''
    Helper function to run another function with a temporary folder
    that automatically will be deleted after the function finishes

    Input arguments:
    params (dict): The parameters for the simulation
    func (callable): The function to be run
    name (str): Th ename of the outpout csv file
    '''

    name = params['name']

    temp_dir = Path('temp')
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(dir=temp_dir) as temp_dir:
        params['Temp'] = temp_dir
        func(name)


def get_rho(params):
    '''
    Function to obtain the optimum theoretical RHO value according to M,N

    Input arguments:
    params (dict): The parameters for the simulation
    disagg_opa (str): The name for the protocol, either 'opa' or 'disagg'
    '''
    
    disagg_opa = params['name']

    if params['RHO'] > 0:
        # needed for manually varying RHO from variables
        return params['RHO']

    M = params['M']
    N = params['N']
    k = params['D'] + params['T']
    k = int(k * 100)    # to avoid float errors

    opa_rho = {
         1000: {1000: 110, 3000: 160, 5000: 206, 10000: 260},
        10000: {1000: 110, 3000: 160, 5000: 206, 10000: 260, 50000: 515},
        50000: {50000: 515},
        100_000: {100_000: 669}
    }

    disagg_rho = {
         1000: {1000: 110, 3000: 154, 5000: 184, 10000: 238},
        10000: {1000: 122, 3000: 176, 5000: 228, 10000: 305, 50000: 659},
        50000: {50000: 697},
        100_000: {100_000: 996}
    }

    if disagg_opa == 'DisAgg':
        r = disagg_rho[M][N]
    elif disagg_opa == 'OPA':
        r = opa_rho[M][N]
    else:
        raise ValueError('Wrong protocol!')

    return r


def parse_parameters(parameters):
    '''
    Modify the parameters according to specific parameters defined in constants.var_params
    These parameters can be dynamically defined and not be present in constants.init_parameters
    In general define experiment variants that affect more than one other parameters
    '''


    # set theoretical rho if OPA or DisAgg
    if parameters['name'] in ['OPA', 'DisAgg']:
        parameters['RHO'] = get_rho(parameters)

    if 'stragglers_case' in parameters:
        # 'D'           # (1) OPA=DisAgg=0.1 | (2) OPA=0.1, DisAgg=0.2  | (3) OPA=DisAgg=0.2
        # 'drop_frac'   # (1) OPA=DisAgg=0   | (2) OPA=0, DisAgg=0.1    | (3) OPA=DisAgg=0.1
        # 'NET_GEN_S'   # (1) OPA=DisAgg=3   | (2) OPA=3, DisAgg=4      | (3) OPA=DisAgg=4
        cases = {'OPA':   [(0.1, 0, 3), (0.1, 0, 3), (0.2, 0.1, 4)],
                'DisAgg': [(0.1, 0, 3), (0.2, 0.1, 4), (0.2, 0.1, 4)]
                }
        c = parameters['stragglers_case']
        n = parameters['name']
        if c not in [1, 2, 3]:
            raise ValueError('Wrong stragglers_case value!')
        if n not in ['OPA', 'DisAgg']:
            raise ValueError('Wrong protocol name value!')

        (parameters['D'],
         parameters['drop_frac'],
         parameters['NET_GEN_S']) = cases[n][c-1]

    # override network speeds for the stragglers if NET_GEN_S != 0
    if parameters['NET_GEN_S'] == 5:       # 5G
        parameters['D_KBPS_S'] = 20_000
        parameters['U_KBPS_S'] = 2_000
    elif parameters['NET_GEN_S'] == 4:     # 4G
        parameters['D_KBPS_S'] = 2_000
        parameters['U_KBPS_S'] = 200
    elif parameters['NET_GEN_S'] == 3:     # 3G
        parameters['D_KBPS_S'] = 500
        parameters['U_KBPS_S'] = 50
    else:
        if parameters['NET_GEN_S'] != 0:
            raise ValueError('Wrong NET_GEN_S value!')

    return parameters


class Stats():
    '''
    Class to hold statistics regarding timings in each protocol phase
    '''

    def __init__(self):
        self.stats = {}     # measurements lists per keyword

    def frmt(self, x):
        o = '{:.3f}'.format(x)
        return o

    def report_stats(self):
        res = {}
        for k,v in self.stats.items():
            a = sum(v) / len(v)
            aa = self.frmt(a)
            if len(v) > 1:
                std = np.std(np.array(v, dtype=float))
                std = self.frmt(std)
                mn = np.min(np.array(v, dtype=float))
                mn = self.frmt(mn)
                mx = np.max(np.array(v, dtype=float))
                mx = self.frmt(mx)
                print(k, ':', aa, '(min=', mn, '| max=', mx, '| std=', std, ')')
            else:
                print(k, ':', aa)
            res[k] = a
        return res


class Clock(Stats):
    '''
    Class to provide clock timings functions and store them for computing statistics
    '''

    def __init__(self):
        super().__init__()
        self.clock_time_stamp = 0

    def tic(self):
        self.clock_time_stamp = time.time()

    def add(self, s, t):
        if s in self.stats:
            self.stats[s].append(t)
        else:
            self.stats[s] = [t]

    def toc(self, s, k_comp = 1.0):   # s = keyword for stats & print
        elt = time.time() - self.clock_time_stamp
        elt *= k_comp
        self.add(s, elt)
        self.tic() 

    def report_stats(self):
        sep = '----------------------------------'
        print(sep)
        print('Timings (sec):')
        print(sep)
        res = super().report_stats()
        print(sep)
        return res


class MemberBase():
    '''
    Base class for any member (client or server) for the execution of the protocol
    '''

    def __init__(self, parameters):
        self.parameters = parameters
        random.seed(self.parameters['S'])
        self.cid = None
        self.comtime = {}   # holds up-down timings per function
        self.temp_dir = self.parameters['Temp']
        self.dropped = False

    def _apply_server_speed_limit(self, kbps, size_key='N'):
        srv_speed = self.parameters['SRV_KBPS']
        if size_key != 'N' and size_key != 'A':
            raise ValueError('Wrong client/committee size argument!')
        n = self.parameters[size_key]
        srv_per_client_speed = srv_speed / n
        return min(srv_per_client_speed, kbps)

    def _get_com_time(self, data, up_down='u', size_key='N'):
        kbps = 0
        if up_down == 'u':
            kbps = self.parameters['U_KBPS']
        elif up_down == 'd':
            kbps = self.parameters['D_KBPS']
        else:
            raise ValueError('Wrong up/down argument!')

        kbps = self._apply_server_speed_limit(kbps, size_key)
        bps = kbps * 1024
        bb = len(pickle.dumps(data))
        t = bb / bps

        if self.parameters['USE_HETEROGENEOUS_CLIENTS']:
            kb = bb / 1024
            t = self._get_heterogeneous_time(kb, up_down, size_key)

        if self.parameters['ADD_COMM_DELAY']:
            time.sleep(t)

        return t

    def _get_heterogeneous_time(self, d, up_down, size_key):

        if size_key != 'N' and size_key != 'A':
            raise ValueError('Wrong client/committee size argument!')
        N = self.parameters[size_key]

        if up_down == 'u':
            s1_max = self.parameters['U_KBPS']
            s2_max = self.parameters['U_KBPS_S']
        elif up_down == 'd':
            s1_max = self.parameters['D_KBPS']
            s2_max = self.parameters['D_KBPS_S']
        else:
            raise ValueError('Wrong up/down argument!')

        slow_frac = self.parameters['slow_frac']

        srv_bw = self.parameters['SRV_KBPS']

        tt = heterogeneity_calc.run(d, N, slow_frac, s1_max, s2_max, srv_bw)
        min_t = min(tt['total_time'])

        return min_t


    def train_and_quantize_model(self):
        # lock each client with a constant seed across multiple processes
        model_name = train_utils.dataset_cfg.model
        if model_name == 'cnn_mnist' or model_name == 'cnn_cifar10':
            self.model, self.model_shape = train_utils.train_model(str(self.cid))
        elif model_name == 'nlp':
            self.model, self.model_shape = train_utils.train_model_nlp(str(self.cid))
        elif model_name == 'efficientnet' or model_name == 'tinynet':
            self.model, self.model_shape = train_utils.train_model_vision(str(self.cid))
        else:
            raise ValueError('Wrong model name!')
        self.model = self.quantizer.quantize(self.model)
        lm = len(self.model)
        self.model = np.reshape(self.model, (lm, 1))

    def _save_model(self):
        if not self.parameters['USE_DISK']: return
        fname = f'{self.temp_dir}/{self.cid}_model.npy'
        np.save(fname, self.model)
        self.model = None

    def _load_model(self):
        if not self.parameters['USE_DISK']: return self.model
        fname = f'{self.temp_dir}/{self.cid}_model.npy'
        A = np.load(fname, allow_pickle=True)
        return A

    def get_model(self):
        return self._load_model()

    def _mark_dropped_clients(self, clients):
        N = self.parameters['N']
        ids = list(range(N))
        dropped = int(self.parameters['drop_frac'] * N)
        if dropped == 0:
            print(f'No dropout clients')
            return clients
        items = random.sample(ids, dropped)
        for i in ids:
            if i in items:
                clients[i].dropped = True
            else:
                clients[i].dropped = False  # reset previous state
        print(f'Dropout clients: {dropped} / {N}')
        return clients


def run_simulation(log_name, init_parameters, sub_ks, name_val_pairs, start_simulation):
    '''
    Function that starts the simulation for each protocol

    Argumnets:
    log_name (str): Total number of clients
    init_parameters (dict): The initial parameters passed to the simulation
    sub_ks (list): List of named columns (str) for the output csv file
    name_val_pairs (): The return of the get_var_params() function,
        with lists of names-values pairs for the variables
    start_simulation (callable):
    '''

    def _get_timestamp_name():
        now = datetime.datetime.now()
        ts = now.strftime('%Y-%m-%d_%H-%M-%S')
        ts_log_name = 'log_' + ts + '_' + log_name + '.csv'
        return ts_log_name

    ts_log_name = _get_timestamp_name()
    ts_log_name_out = ts_log_name

    params = copy.deepcopy(init_parameters)

    df_all = None

    out_dir = Path('outputs')
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)

    names, vals = name_val_pairs

    for vv in vals:
        # set variable parameters
        for i,v in enumerate(vv):
            params[names[i]] = v

        if params['USE_TRAINING']:
            
            # override dataset temp dir to be inside auto-delete temp folder
            train_utils.set_params(params)

            train_utils.use_torch_seed(0)

            train_utils.remove_temp_model_dir()

            model_name = train_utils.dataset_cfg.model
            if model_name == 'cnn_mnist' or model_name == 'cnn_cifar10':
                print('Creating datasets and initial model...')
                # partition the dataset for clients
                train_utils.create_datasets_and_init_model(params['N'])
            elif model_name == 'nlp':
                print('Creating initial NLP model...')
                train_utils.save_initial_nlp_model()
            elif model_name == 'efficientnet' or model_name == 'tinynet':
                print('Creating initial Vision model...')
                train_utils.save_initial_vision_model()
            else:
                raise ValueError('Wrong model name!')


        res, metrics = start_simulation(params)
        dft = pd.DataFrame(res, index=[0])

        pp = {k: params[k] for k in sub_ks if k in params}
        dfp = pd.DataFrame(pp, index=[0])

        rest_keys = [k for k in params.keys() if k not in sub_ks]
        ppp = {k: params[k] for k in rest_keys if k in params}
        dfrest = pd.DataFrame(ppp, index=[0])

        df = pd.concat([dfp, dft, dfrest], axis=1)

        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], ignore_index=True, axis=0)

        if params['USE_TRAINING']:
            dfm = pd.DataFrame(metrics)
            dfm.to_csv(out_dir / Path('metrics_' + ts_log_name))
            ts_log_name_out = 'last_round_' + ts_log_name
            # reset timestamp for another set of training if exists
            ts_log_name = _get_timestamp_name()

        # override current results
        df_all.to_csv(out_dir / Path(ts_log_name_out))

        if params['USE_TRAINING']:
            # reset accumulated measurements
            df_all = None

    print('OK')
