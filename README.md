# DisAgg: Distributed Aggregators for Efficient Secure Aggregation in Federated Learning

## Hardware Requirements

CPU with 10+ cores, Nvidia GPU with at least 10 GB VRAM and minimum CUDA 12.2 support, at least 128 GB RAM, and 0.5 TB disk space.

**Reference hardware:** All experiments were conducted on a single machine using an Intel(R) Xeon(R) Gold 5220 @ 2.20GHz with 128GB RAM, and an NVIDIA GeForce RTX 2080 Ti with 12GB VRAM.

## Installation

A virtual environment with Python 3.10 installed and bash as the default terminal is recommended.

**Note:** NumPy <2.x is required (breaking changes in NumPy 2.x).

```bash
# install as an editable package to allow modifying constants.py
pip install -e .
```

Confirm installation by executing: `pip show disagg`

## Datasets

- **MNIST** / **CIFAR-10**: Downloaded automatically via `torchvision` on first run. No manual steps needed.
- **CIFAR-100** / **CelebA**: Downloaded from HuggingFace and partitioned into federated splits. Pre-computed index files (`dataset/*-splits.npz`) are included in the repo. To build the dataset files, run:
  ```bash
  python -m dataset.build_dataset --dataset_name cifar100
  python -m dataset.build_dataset --dataset_name celeba
  ```
  **Note:** The disk usage for downloading & processing CelebA is ~28GB.

- **SST-2**: Pre-split data is already included in the repo under `dataset/sst2-processed/`. No download needed.

## Running Experiments

Experiments are configured via `src/constants.py`. Default parameters are in `init_parameters`. To run a specific experiment, pass its index from `var_params` as an argument:

Predefined experiments without editing variables can be executed directly from the command line, as shown below. `<index>` specifies an experiment in the range [1-9] with `0` defaulting to a simple test.

Parallelism is controlled by the optional command line argument `--num_proc`; set to `0` or omit this argument for sequential execution. See the table below for recommended values for `<num_proc>`.

### Commands:

```bash
python -m disagg_test --exp_index=<index> --num_proc=<num_proc>
python -m opa_test --exp_index=<index> --num_proc=<num_proc>
python -m light_secagg_test --exp_index=<index> --num_proc=<num_proc>
python -m secagg_plus_test --exp_index=<index> --num_proc=<num_proc>
```

For plain-text FL experiments, set `USE_SECURITY=False` in `constants.py` and use the `disagg_test` module.

**Note:** Wall clock times below refer to the full experiment required to produce the paper plots, measured on the reference hardware above.

| Index | Description | Recommended num_proc | Wall Clock Time | Scripts to run |
|-------|-------------|----------------------|-----------------|----------------|
| 0 | Quick test (N=100, M=1k) | 16 | <1m | disagg_test.py |
| 1 | Set M=N=10k and vary ρ (Figure 7) | 16 | ~13.5h | {disagg\|opa}_test.py |
| 2 | Sweep M, N with all protocols (Figure 6) | 16 | ~25.7h | {disagg\|opa\|light_secagg\|secagg_plus}_test.py |
| 3 | Set M=N=100k and vary ρ (Table 2) | 4 | >1d | {disagg\|opa}_test.py |
| 4 | Set M=N=10k and vary γ,δ (Figure 8) | 16 | ~23.2h | {disagg\|opa}_test.py |
| 5 | Train NLP model with SST2 (Figure 9c-d) | 2 | ~3.5h | {disagg\|opa}_test.py for lora_r=16, disagg_test.py for lora_r=64 |
| 6 | Train EfficientNet model with CELEBA (Figure 9e) | 2 | ~4.3h | {disagg\|opa}_test.py |
| 7 | Train CNN models on MNIST & CIFAR10 (Figure 9a-b) | 16 | ~2.9h | {disagg\|opa}_test.py |
| 8 | Train TinyNet model on CIFAR100 (Figure 9f) | 4 | ~2.1h | {disagg\|opa}_test.py |
| 9 | Experiment with stragglers on CIFAR10 (Figure 10) | 16 | ~6.4h | {disagg\|opa}_test.py |

Example — run SST2 experiment using DisAgg:
```bash
python -m disagg_test --exp_index=5 --num_proc=2
```

**Notes:**
- When training with CIFAR10/100, the accuracy after 30 rounds is expected to be ~0.3/~0.5, due to the non-IID dataset splits used.
- Experiments create temporary files in the folder `./temp`. This is created automatically at the beginning and its contents are deleted automatically once each experiment finishes. It is recommended to have at least 200-300GB free disk space for temporary files.

## Experiment customization

Experiments can be customized by modifying the configuration parameters in `src/constants.py`. Edit `init_parameters` to change default values, or add new sweep configurations to `var_params`.


## Paper

[Arxiv Link](https://arxiv.org/abs/2605.13708)

## Citation

```
Mehmood, H., Tatsis, G., Alexopoulos, D., Saravanan, K., Xu, J., Drosou, A., and Ozay, M. DisAgg: Distributed Aggregators for Efficient Secure Aggregation in Federated Learning. To appear in Proceedings of the Ninth Annual Conference on Machine Learning and Systems, MLSys 2026.
```
