# DisAgg v1.0.0-alpha

## Hardware Requirements

CPU with 10+ cores, Nvidia GPU with at least 10 GB VRAM and minimum CUDA 12.2 support, at least 128 GB RAM, and 0.5 TB disk space.

## Installation

The following assumes a virtual environment with Python 3.10 installed and bash as the default terminal.

```bash
# install dependencies
python -m pip install -r requirements.txt

# create/update python path
echo "PYTHONPATH=$PYTHONPATH:." >> ~/.bashrc
source ~/.bashrc
```

## Datasets

- **MNIST** / **CIFAR-10**: Downloaded automatically via `torchvision` on first run. No manual steps needed.
- **CIFAR-100** / **CelebA**: Downloaded from HuggingFace and partitioned into federated splits. Pre-computed index files (`dataset/*-splits.npz`) are included in the repo. To build the dataset files, run:
  ```bash
  python src/dataset/build_dataset.py --dataset_name cifar100
  python src/dataset/build_dataset.py --dataset_name celeba
  ```
- **SST-2**: Pre-split data is already included in the repo under `dataset/sst2-processed/`. No download needed.

## Running Experiments

Experiments are configured via `src/constants.py`. Default parameters are in `init_parameters`. To run a specific experiment, pass its index from `var_params` as an argument:

```bash
PYTHONPATH=$PYTHONPATH:. python src/disagg_test.py <index>
PYTHONPATH=$PYTHONPATH:. python src/opa_test.py <index>
PYTHONPATH=$PYTHONPATH:. python src/light_secagg_test.py <index>
PYTHONPATH=$PYTHONPATH:. python src/secagg_plus_test.py <index>
```

For plain-text FL experiments, set `USE_SECURITY=False` in `src/constants.py` and use the `src/disagg_test.py` script.

| Index | Description                                        | Recommended num_proc | Scripts to run                                                      |
|-------|----------------------------------------------------|----------------------| ------------------------------------------------------------------- |
| 0     | Quick test (N=100, M=1k)                           | 16                   | disagg_test.py                                                      |
| 1     | Set M=N=10k and vary ρ (Figure 6)                  | 16                   | {disagg|opa}_test.py                                                |
| 2     | Sweep M, N with all protocols (Figure 7)           | 16                   | {disagg|opa|light_secagg|secagg_plus}_test.py                       |
| 3     | Set M=N=100k and vary ρ (Table 3)                  | 4                    | {disagg|opa}_test.py                                                |
| 4     | Set M=N=10k and vary γ,δ (Figure 8)                | 16                   | {disagg|opa}_test.py                                                |
| 5     | Train NLP model with SST2 (Table 6)                | 4                    | {disagg|opa}_test.py for lora_r=16,  disagg_test.py for lora_r=64   |
| 6     | Train EfficientNet model with CELEBA (Table 6)     | 4                    | {disagg|opa}_test.py                                                |
| 7     | Train CNN models on MNIST & CIFAR10 (Figure 9)     | 16                   | {disagg|opa}_test.py                                                |
| 8     | Train TinyNet model on CIFAR100 (Figure 9)         | 16                   | {disagg|opa}_test.py                                                |
| 9     | Experiment with stragglers on CIFAR10 (Figure 12)  | 16                   | {disagg|opa}_test.py                                                |

Example — run SST2 experiment with DisAgg:
```bash
PYTHONPATH=$PYTHONPATH:. python src/disagg_test.py 5
```

## Experiment customization

Experiments can be customized by modifying the configuration parameters in `src/constants.py`. Edit `init_parameters` to change default values, or add new sweep configurations to `var_params`. Parallelism is controlled by `num_proc`; set to `0` for sequential execution.
