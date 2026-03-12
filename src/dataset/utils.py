#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================
import numpy as np
import torch
import random
import os
import pickle

def use_torch_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True    
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def save_dataloaders(partitions_dir: str, trainloaders, valloaders):
    '''
    Function to save the dataloaders to disk per user ID 
    '''
    for i,t in enumerate(trainloaders):
        with open(f'{partitions_dir}/train_{i}.pkl', 'wb') as f:
            pickle.dump(t, f)
    for i,t in enumerate(valloaders):
        with open(f'{partitions_dir}/val_{i}.pkl', 'wb') as f:
            pickle.dump(t, f)


def read_dataloaders(partitions_dir: str, cid: str):
    '''
    Function to load the dataloaders from disk per user ID 
    '''
    with open(f'{partitions_dir}/train_{cid}.pkl', 'rb') as f:
        trainloader = pickle.load(f)
    with open(f'{partitions_dir}/val_{cid}.pkl', 'rb') as f:
        valloader = pickle.load(f)
    return trainloader, valloader
