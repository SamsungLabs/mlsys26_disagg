#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================

# set non-zero to process clients in parallel with num_proc = number of parallel jobs
num_proc = 16

# set names and values of variable parameters to iterate, order will be preserved
# default parameters are listed below in "init_parameters"
# for multiple experiments run the command with an index argument
# pointing to the variable list index below, e.g.:
# "python src/disagg_test.py 1",
# to get the 2nd variable set from the list (if exists)
# NOTE: training works only with DisAgg & OPA
var_params = [
    {# 0    simple example with low resources
        'M': [1_000],
        'N': [100],
        'RHO': [10]
    },
    {# 1    parameters for the results in Figure 6 of the paper
    'M': [10_000],
    'N': [10_000],
    'RHO': [-1, 5, 10, 25, 50, 100, 250, 500, 1_000],
    },
    {# 2    parameters for the results in Figure 7 of the paper
    'M': [1_000, 10_000],
    'N': [1_000, 3_000, 5_000, 10_000],
    },
    {# 3    parameters for the results in Table 3 of the paper
    'M': [100_000],
    'N': [100_000],
    'RHO': [-1, 5, 10, 25, 50, 100, 250, 500, 1_000],
    },
    {# 4    parameters for the results in Figure 8 of the paper
    'M': [10_000],
    'N': [10_000],
    'D': [0.01, 0.05, 0.1, 0.15],
    'T': [0.01, 0.05, 0.1, 0.15],
    },
    {# 5    parameters for the results in Figure 9 of the paper
    'N': [10],
    'RHO': [2],
    'lora_r': [16, 64],          # 16=>~297k params (DisAgg or OPA), 64=>~1.1M params (DisAgg only)
    'dataset': ['sst2'],
    'USE_SECURITY': [True, False],
    'USE_TRAINING': [True],
    'TRAINING_ROUNDS': [30]
    },
    {# 6    parameters for the results in Figure 9 of the paper
    'N': [10],
    'RHO': [2],
    'dataset': ['celeba'],
    'USE_SECURITY': [True, False], # False (=> plain-text FL) can only be used with src/disagg_test.py
    'USE_TRAINING': [True],
    'TRAINING_ROUNDS': [30]
    },
    {# 7    parameters for the results in Figure 9 of the paper
    'N': [100],
    'RHO': [10],
    'dataset': ['mnist', 'cifar10'],
    'USE_SECURITY': [True, False], # False (=> plain-text FL) can only be used with src/disagg_test.py
    'USE_TRAINING': [True],
    'TRAINING_ROUNDS': [30]
    },
    {# 8    parameters for the results in Figure 9 of the paper
    'N': [10],
    'RHO': [2],
    'dataset': ['cifar100'],
    'USE_SECURITY': [True, False], # False (=> plain-text FL) can only be used with src/disagg_test.py
    'USE_TRAINING': [True],
    'TRAINING_ROUNDS': [30]
    },
    {# 9    parameters for the results in Figure 12 of the paper
    'N': [100],
    'RHO': [10],
    'USE_TRAINING': [True],
    'TRAINING_ROUNDS': [30],
    'dataset': ['cifar10'],
    'USE_HETEROGENEOUS_CLIENTS': [True],
    'ADD_COMM_DELAY': [True],
    'stragglers_case': [1, 2, 3],      # dynamically created, elaborates in common.parse_parameters
    # 'D': [0.1],         # (1) OPA=DisAgg=0.1 | (2) OPA=0.1, DisAgg=0.2  | (3) OPA=DisAgg=0.2
    # 'drop_frac': [0],   # (1) OPA=DisAgg=0   | (2) OPA=0, DisAgg=0.1    | (3) OPA=DisAgg=0.1
    # 'NET_GEN_S': [3],   # (1) OPA=DisAgg=3   | (2) OPA=3, DisAgg=4      | (3) OPA=DisAgg=4
    },
]


# initial parameters for all protocols
init_parameters = {
    'M': 10_000,   # model size as number of parameters
    'N': 100,      # number of selected clients
    'D': 0.2,       # delta = maximum dropout fraction (tolerance)
    'drop_frac': 0.0,    # the actual simulated dropouts applied in clients & committee/aggregators sets
    'T': 0.1,       # gamma = maximum corrupt fraction (tolerance)
    'sigma': 40,    # security parameter
    'eta': 40,      # correctness parameter
    'P': 53,        # plaintext field size in bits, to match with OPA
    'F': 128,       # q field size in bits for OPA & DisAgg
    'lambda_key_dim': 2048, # security parameter for PRG in OPA
    'S': 42,         # global seed
    'D_KBPS': 20_000,      # Download speed (kB/s) limit per client/committee
    'U_KBPS': 2_000,       # Upload speed (kB/s) limit per client/committee
    'SRV_KBPS': 25_000_000,   # Server speed limit (kB/sec) (Up/Down)
    'D_KBPS_S': 2_000,       # Slow download speed limit (kB/s) 4G
    'U_KBPS_S': 200,       # Slow upload speed limit (kB/s) 4G
    'NET_GEN_S': 0,         # network generation for the stragglers, 4 = 4G, 3 = 3G, 0 = custom
    'slow_frac': 0.3,        # Slow clients/committee fraction
    'Temp': 'temp',          # temp dir
    'k_comp': 0.66,         # Server / Client computation cost ratio
    'RHO': -1,              # desired ρ value for the secret length. -1 = use theoretical best value
    'TRAINING_ROUNDS': 2,   # the training iterations = FL iterations
    'USE_BFT_CALC': False,  # If True use calculations with Byzantine fault tolerance constraint
    'USE_DISK': True,        # save large data per client to disk instead of ram
    'USE_SECURITY': True,     # Select whether use security or just plain FL
    'USE_TRAINING': False,   # select for training model
    'dataset': 'mnist',        # the name of the dataset (currently [mnist, cifar10, cifar100, sst2, celeba])
    'learning_rate': -1,        # the learning rate value for training, set > 0 to override defaults
    'USE_QUANTIZATION': True,   # in training with FL (no security) optionally disable quantization
    'USE_CLIENT_AGGREGATION': True,   # simulate DisAgg aggregation at the client side to reduce disk space for the shares
    'USE_HETEROGENEOUS_CLIENTS': False, # use mixed network speeds for the clients
    'ADD_COMM_DELAY': True,    # put delay at clients side equal to communication timings
}


# checks
if init_parameters['USE_TRAINING'] == False:
    init_parameters['USE_SECURITY'] = True

if init_parameters['USE_SECURITY'] == True:
    init_parameters['USE_QUANTIZATION'] = True

if init_parameters['USE_CLIENT_AGGREGATION'] == True and init_parameters['USE_DISK'] == False:
    init_parameters['USE_CLIENT_AGGREGATION'] = False
