# Reconstructs federated dataset splits from pre-computed index files (.npz).
# Client data partitions were generated using Dirichlet LDA, producing
# approximately IID splits across clients.
import argparse, torch, gc, numpy as np
from datasets import load_dataset, Dataset
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', choices=['cifar100', 'celeba'], required=True)
args = parser.parse_args()

BATCH_SIZE = 250
CONFIG = {
    'cifar100': dict(dataset_path='uoft-cs/cifar100',    output_dir=Path('dataset/cifar100-processed'), img_attr_name='img',   valid_split='train'),
    'celeba':   dict(dataset_path='flwrlabs/celeba',      output_dir=Path('dataset/celeba-processed'),   img_attr_name='image', valid_split='valid'),
}

cfg           = CONFIG[args.dataset_name]
output_dir    = cfg['output_dir']
img_attr_name = cfg['img_attr_name']

splits  = np.load(Path('dataset') / f'{args.dataset_name}-splits.npz')
dataset = load_dataset(cfg['dataset_path'], streaming=False)

train_data = dataset['train']
valid_data = dataset[cfg['valid_split']]
test_data  = dataset['test']
attribute_names = [col for col in train_data.column_names if col != img_attr_name]

def save_split(data, indices, path):
    all_data = {img_attr_name: [], **{attr: [] for attr in attribute_names}}
    for i in range(0, len(indices), BATCH_SIZE):
        batch = data.select(indices[i:i+BATCH_SIZE].tolist())
        d = batch.to_dict()
        all_data[img_attr_name].extend(d[img_attr_name])
        for attr in attribute_names:
            all_data[attr].extend(d[attr])
        del batch, d; gc.collect()
    torch.save(Dataset.from_dict(all_data), path)

num_clients = sum(1 for k in splits if k.startswith('client_'))
for cid in tqdm(range(num_clients), desc="Saving client splits"):
    (output_dir / "federated" / str(cid)).mkdir(parents=True, exist_ok=True)
    save_split(train_data, splits[f'client_{cid}'], output_dir / "federated" / str(cid) / "train.pt")

(output_dir / "central" / "0").mkdir(parents=True, exist_ok=True)
save_split(valid_data, splits['valid_indices'], output_dir / "central" / "0" / "valid.pt")
save_split(test_data,  splits['test_indices'],  output_dir / "central" / "0" / "test.pt")
print(f"OK! Output: {output_dir}")
