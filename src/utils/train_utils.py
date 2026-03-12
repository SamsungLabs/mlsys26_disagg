#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================
import os
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Any, List, Union
from tempfile import mkdtemp
import pickle
import shutil
import torch
import evaluate as evaluate_package
import numpy as np
import re

from dataset.dataset_lda import load_datasets
from dataset.utils import save_dataloaders, read_dataloaders, use_torch_seed
from dataset import model
from dataset.data_structures import Parameters

import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]
Scalar = Union[bool, bytes, float, int, str]

from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainingArguments,
    Trainer,
    default_data_collator,
    DataCollatorWithPadding,
    TrainerCallback,
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
    AutoImageProcessor,
    AutoTokenizer,
    EvalPrediction,
)

from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType,
)

from typing import Union, List


class CFG():
    name = 'mnist'               # Dataset name, 'mnist', 'cifar10', 'cifar100', 'celeba', 'sst2'
    model = 'cnn_mnist'          # model name ('cnn_mnist', 'cnn_cifar10', 'nlp', 'efficientnet', 'tinynet')
    method = 2                   # 1 = IID, 2 = Dirichlet, 3 = Pathological
    alpha = 0.5                  # for Dirichlet, parameters required: 'alpha', 'equal_samples'
    equal_samples = False         # equal_samples refers to the total number of samples per client
    num_classes = 3              # for Pathological, parameters required: 'num_classes' (per partition)
    flower_dataset = False       # Use the flower's federated datasets
                                # If False use old (torchvision) datasets with LDA partitioning (faster)
    temp_dir = 'tmp'            # placeholder, it will be changed dynamically
    model_file_name = 'model.pkl'
    testloader_file_name = 'test.pkl',
    params = None               # parameters placeholder from constants.py

dataset_cfg = CFG()

def set_params(params):
    dataset_cfg.temp_dir = params['Temp'] + '/datasets_temp'
    dataset_cfg.params = params
    dataset_cfg.name = params['dataset']

    dataset_list = ['mnist', 'cifar10', 'cifar100', 'sst2', 'celeba']

    if dataset_cfg.name not in dataset_list:
        raise ValueError('Unknown dataset name!')

    # map model to dataset
    if dataset_cfg.name == 'mnist':
        dataset_cfg.model = 'cnn_mnist'
    elif dataset_cfg.name == 'cifar10':
        dataset_cfg.model = 'cnn_cifar10'
    elif dataset_cfg.name == 'cifar100':
        dataset_cfg.model = 'tinynet'
    elif dataset_cfg.name == 'sst2':
        dataset_cfg.model = 'nlp'
    elif dataset_cfg.name == 'celeba':
        dataset_cfg.model = 'efficientnet'
    else:
        raise ValueError('Unknown dataset name!')

    if dataset_cfg.model == 'cnn_mnist':
        model.Net = model.Net_MNIST
    elif dataset_cfg.model == 'cnn_cifar10':
        model.Net = model.Net_CIFAR10
    else:
        pass


def remove_temp_model_dir():
    # remove temp folder for multiple experiments
    if Path(dataset_cfg.temp_dir).exists():
        shutil.rmtree(dataset_cfg.temp_dir, ignore_errors=True)


def get_device_per_client_name(cid=None):
    if not torch.cuda.is_available():
        print('CUDA not available! The CPU will be used instead!')
        dev_name = "cpu"
    else:
        if cid is None:
            dev_name = "cuda"
        else:
            num_gpus = torch.cuda.device_count()
            gpu_id = int(cid) % num_gpus
            dev_name = f"cuda:{gpu_id}"
    return dev_name


def get_device_per_client(cid=None):
    dev_name = get_device_per_client_name(cid)
    DEVICE = torch.device(dev_name)
    return DEVICE


def freeze_params(model: torch.nn.Module, regex: Union[str, List[str]], level: int = 0):
    """
    Freezes the parameters of a model based on the provided regex pattern(s).

    This function iterates through the named parameters of the model and sets
    `requires_grad` to `False` for parameters whose names match any of the provided regex patterns.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters are to be frozen.
    regex : Union[str, List[str]]
        A single regex pattern or a list of regex patterns to match against parameter names.
    level : int, optional
        The maximum levels (entries in `regex` as list) to freeze. Default is 0, which only freezes the first.

    Returns
    -------
    None

    Example Usage
    -------------
    >>> model = torchvision.models.resnet18(pretrained=True)
    >>> freeze_params(model, ["layer1", "layer2"])

    Exceptions
    ----------
    AssertionError
        If `regex` is not of type `str` or `List[str]`.

    Notes
    -----
    - The function converts a single regex string to a list for uniform processing.
    - The function uses Python's `re.search` to match patterns in parameter names.
    - If `level` is less than 0, all patterns in `regex` will be frozen, otherwise
      only upto `level` patterns in `regex` will be frozen.
    """
    if not isinstance(regex, list):
        regex = [regex]
    if level >= 0: # if level < 0, use all levels
        regex = regex[:level + 1] # only freeze upto a certain level
    assert isinstance(regex[0], str), "regex must be of type str or List[str]"
    for name, param in model.named_parameters():
        for reg in regex:
            if re.search(reg, name):
                param.requires_grad = False


def create_datasets_and_init_model(N):
    temp_dir = dataset_cfg.temp_dir
    outdir = Path(temp_dir)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    temp_dir = dataset_cfg.temp_dir
    trainloaders, valloaders, testloader = load_datasets(
        dataset_cfg=dataset_cfg, num_clients=N, batch_size=32
    )
    # override previous files
    save_dataloaders(temp_dir, trainloaders, valloaders)

    # save an initial global model
    net = get_model()
    parameters = get_parameters(net, {})
    params = Parameters(parameters)
    model_flat = params.flat
    model_shape = params.shape
    with open(f'{temp_dir}/{dataset_cfg.model_file_name}', 'wb') as f:
        pickle.dump((model_flat, model_shape), f)

    # save testloader
    save_test_loader(testloader)


def save_test_loader(testloader):
    temp_dir = dataset_cfg.temp_dir
    with open(f'{temp_dir}/{dataset_cfg.testloader_file_name}', 'wb') as f:
        pickle.dump(testloader, f)


def load_testloader(parameters):
    model_name = dataset_cfg.model
    if model_name == 'cnn_mnist' or model_name == 'cnn_cifar10':
        temp_dir = dataset_cfg.temp_dir
        with open(f'{temp_dir}/{dataset_cfg.testloader_file_name}', 'rb') as f:
            testloader = pickle.load(f)
    elif model_name == 'nlp':
        testloader = torch.load(Path('dataset/sst2-processed/central/0') / ("valid" + ".pt"), weights_only=False)
    elif model_name == 'efficientnet' or model_name == 'tinynet':
        if parameters['dataset'] == 'celeba':
            testloader = torch.load(Path('dataset/celeba-processed/central/0') / ("valid" + ".pt"), weights_only=False)
        elif parameters['dataset'] == 'cifar100':
            testloader = torch.load(Path('dataset/cifar100-processed/central/0') / ("valid" + ".pt"), weights_only=False)
        else:
            raise ValueError('Wrong dataset for vision model!')
    else:
        raise ValueError('Wrong model name!')

    return testloader


def get_parameters(net, config: Dict[str, Scalar]) -> NDArrays:
    """Returns the parameters of the current net."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: NDArrays) -> None:
    """Changes the parameters of the model using the given ones."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def state_dict_2_params(model):
    """Only return requires_grad=True subset"""
    if hasattr(model, 'peft_config'):
        state_dict = get_peft_model_state_dict(model)
    else:
        state_dict = model.state_dict()

    # NOTE: replacement is for workaround in lora weights for tinyllama, may not work with other models, need to test!
    trainable_params_names = \
        set([n.replace('.default', '').replace('.modules_to_save', '') \
                            for n, p in model.named_parameters() if p.requires_grad])
    params = []
    for k,v in state_dict.items():
        if k in trainable_params_names:
            params.append(v.cpu().numpy())

    return params


def params_2_state_dict(model, params,):
    """This function should only be used for LoRA models"""
    if hasattr(model, 'peft_config'):
        peft_state = get_peft_model_state_dict(model)
    else:
        peft_state = model.state_dict()
    trainable_params_names_set = \
        set([n.replace('.default', '').replace('.modules_to_save', '') \
                            for n, p in model.named_parameters() if p.requires_grad])

    trainable_params_names = []
    # ordered list of keys based on original order
    for k,_ in peft_state.items():
        if k in trainable_params_names_set:
            trainable_params_names.append(k)

    trainable_params_dict = OrderedDict(zip(trainable_params_names, params))
    # the returned state_dict retains non-trainable weights and only repales trainable weights with those in params
    state_dict = OrderedDict({})
    for (k,v) in peft_state.items():
        if k in trainable_params_names_set:
            state_dict[k] = torch.from_numpy(np.copy(trainable_params_dict[k]))
        else:
            state_dict[k] = v

    return state_dict


def set_flat_model(net, model_flat, shape, parameters=None):
    if parameters is None:
        model_name = 'cnn_mnist'
    else:
        model_name = dataset_cfg.model
    if model_flat is not None:
        p = Parameters.from_flat_array(model_flat, shape)
        pp = p.tolist()
        if model_name == 'cnn_mnist' or model_name == 'cnn_cifar10':
            set_parameters(net, pp)
        else:
            if hasattr(net, 'peft_config'):
                set_peft_model_state_dict(
                    net, params_2_state_dict(net, pp)
                )
            else:
                params_dict = params_2_state_dict(net, pp)
                net.load_state_dict(params_dict, strict=True)


def test_nlp_model(model, tokenizer, testloader, device, test_batch_size=64, is_regression=False):
    if test_batch_size > 1:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

    metric = evaluate_package.load('glue', 'sst2', cache_dir=None)
    def compute_metrics_glue(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    compute_metrics_fn = compute_metrics_glue

    if dataset_cfg.params['learning_rate'] < 0:
        l_r = 5e-5
    else:
        l_r = dataset_cfg.params['learning_rate']

    temp_dir = mkdtemp(suffix=None, prefix='temp_dir_dp_nlp_eval_exp', dir=None)
    train_args = Seq2SeqTrainingArguments(
        gradient_accumulation_steps=0,
        gradient_checkpointing=False,
        learning_rate=l_r,
        logging_strategy="steps",
        logging_steps=10,
        max_steps=-1,
        optim='adamw_torch',
        output_dir=temp_dir,
        log_level='warning',
        per_device_train_batch_size=test_batch_size,
        per_device_eval_batch_size=test_batch_size,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        compute_metrics=compute_metrics_fn,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    with torch.backends.cuda.sdp_kernel(enable_flash=True,
                                        enable_math=True,
                                        enable_mem_efficient=True):
        results = trainer.evaluate(eval_dataset=testloader)

    return results['eval_loss'], results['eval_accuracy']


def save_initial_nlp_model():
    """
        Instantiates nlp model with pretrained and random weights then save it to disk
    """
    temp_dir = dataset_cfg.temp_dir
    if not Path(temp_dir).exists():
        Path(temp_dir).mkdir(parents=True)

    # save the entire pretrained model to disk
    net, tokenizer = get_nlp_model()
    net.save_pretrained(os.path.join(temp_dir, 'pretrained_model'))
    tokenizer.save_pretrained(os.path.join(temp_dir, 'pretrained_model'))
    
    # save an initial global model only trainable params
    parameters = state_dict_2_params(net)
    params = Parameters(parameters)
    model_flat = params.flat
    model_shape = params.shape
    with open(f'{temp_dir}/{dataset_cfg.model_file_name}', 'wb') as f:
        pickle.dump((model_flat, model_shape), f)


def test_and_save_global_model(model_flat, shape, quantizer, parameters):
    temp_dir = dataset_cfg.temp_dir

    contributors = parameters['N'] - int(parameters['drop_frac'] * parameters['N'])

    # compute average & dequantize ; model_flat is already a sum
    if model_flat.dtype == np.float32 or model_flat.dtype == np.float64:
        model_flat = model_flat / contributors
    else:
        model_flat = model_flat.astype(int)
        model_flat = model_flat // contributors
    dequantized_model = quantizer.dequantize(model_flat)

    testloader = load_testloader(parameters)

    DEVICE = get_device_per_client()

    model_name = dataset_cfg.model
    if model_name == 'cnn_mnist' or model_name == 'cnn_cifar10':
        net = get_model()
    elif model_name == 'nlp':
        net, tokenizer = get_nlp_model()
        net.to(DEVICE)
    elif model_name == 'efficientnet' or model_name == 'tinynet':
        net, processor = get_vision_model()
        net.to(DEVICE)
    else:
        raise ValueError('Wrong model name!')

    set_flat_model(net, dequantized_model, shape, parameters)

    if model_name == 'cnn_mnist' or model_name == 'cnn_cifar10':
        loss, acc = model.test(net, testloader, DEVICE)
    elif model_name == 'nlp':
        loss, acc = test_nlp_model(net, tokenizer, testloader, DEVICE)
    elif model_name == 'efficientnet' or model_name == 'tinynet':
        loss, acc = test_vision_model(net, processor, testloader, DEVICE)
    else:
        raise ValueError('Wrong model name!')

    with open(f'{temp_dir}/{dataset_cfg.model_file_name}', 'wb') as f:
        pickle.dump((dequantized_model, shape), f)

    return loss, acc


def load_global_model():
    temp_dir = dataset_cfg.temp_dir
    mp = Path(f'{temp_dir}/{dataset_cfg.model_file_name}')
    if mp.exists():
        with open(mp, 'rb') as f:
            model, shape = pickle.load(f)
        return model, shape
    else:
        raise ValueError('Empty global model!')


def train_model(cid):
    
    use_torch_seed(int(cid))
    
    temp_dir = dataset_cfg.temp_dir
    trainloader, valloader = read_dataloaders(temp_dir, cid)

    net = get_model(cid)

    gm, sh = load_global_model()
    set_flat_model(net, gm, sh)

    DEVICE = get_device_per_client(cid)
    model.train(
        net,
        trainloader,
        DEVICE,
        epochs=1,
        learning_rate=0.01,
        momentum=0.9,

    )
    parameters = get_parameters(net, {})

    params = Parameters(parameters)
    model_flat = params.flat
    model_shape = params.shape

    return model_flat, model_shape


def get_model(cid=None):
    if cid is None:
        use_torch_seed(0)
    else:
        use_torch_seed(int(cid))

    DEVICE = get_device_per_client(cid)
    net = model.Net().to(DEVICE)
    return net


def get_nlp_model(cid=None):
    # Load from saved pretrained model if it exists (for clients), otherwise use default (for initial server setup)
    temp_dir = dataset_cfg.temp_dir
    saved_model_path = os.path.join(temp_dir, 'pretrained_model')
    if os.path.exists(saved_model_path):
        model_path = saved_model_path
    else:
        model_path = 'distilbert/distilbert-base-uncased'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=4096)
    tokenizer.padding_side = "right"

    dev_name = get_device_per_client_name(cid)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        device_map=dev_name,
        num_labels=2,
    )

    _lora_alpha = {16: 32, 64: 256}  # 16=>~297k params, 64=>~1.1M params
    lora_r = (dataset_cfg.params or {}).get('lora_r', 16)
    peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=_lora_alpha[lora_r],
            target_modules=["q_lin", "k_lin"],
            lora_dropout=0.001,
    )
    model = get_peft_model(model, peft_config)

    # freeze the 768*768 matrix -- all base layers of lora modules are automatically frozen
    freeze_params(model, '^.*pre_classifier.*weight$')

    print("num_trainable_params_nlp:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model, tokenizer


def train_model_nlp(cid):
    temp_dir = dataset_cfg.temp_dir

    DEVICE = get_device_per_client(cid)
    trainloader = torch.load(Path('dataset/sst2-processed/federated') / cid / ("train" + ".pt"), weights_only=False)

    model, tokenizer = get_nlp_model(cid)
    model.to(DEVICE)

    # flat trainable weights only
    gm, sh = load_global_model()
    set_flat_model(model, gm, sh, dataset_cfg.params)

    temp_output_dir = mkdtemp(suffix=None, prefix='temp_dir_dp_nlp_client_exp', dir=None)

    if dataset_cfg.params['learning_rate'] < 0:
        l_r = 5e-5 # use 5e-5 for smaller models
    else:
        l_r = dataset_cfg.params['learning_rate']

    train_args = Seq2SeqTrainingArguments(
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        learning_rate=l_r, 
        logging_strategy="no",
        logging_steps=10,
        num_train_epochs=1,
        optim='adamw_torch',
        output_dir=temp_output_dir,
        log_level='warning',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=trainloader,
        eval_dataset=None,
        args=train_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    train_results = trainer.train()
    shutil.rmtree(train_args.output_dir, ignore_errors=True)

    parameters = state_dict_2_params(model)

    params = Parameters(parameters)
    model_flat = params.flat
    model_shape = params.shape

    return model_flat, model_shape


def get_vision_model(cid=None):
    # Load from saved pretrained model if it exists (for clients), otherwise use default (for initial server setup)
    temp_dir = dataset_cfg.temp_dir
    saved_model_path = os.path.join(temp_dir, 'pretrained_model')
    if os.path.exists(saved_model_path):
        model_path = saved_model_path
    else:
        if dataset_cfg.model == 'efficientnet':
            model_path = 'google/efficientnet-b0'
        elif dataset_cfg.model == 'tinynet':
            model_path = 'timm/tinynet_a.in1k'
        else:
            raise ValueError('Wrong dataset for vision model!')

    dev_name = get_device_per_client_name(cid)

    if dataset_cfg.model == 'tinynet':
        processor = AutoImageProcessor.from_pretrained(
            'timm/tinynet_a.in1k',
            use_fast=True,
            device_map=dev_name,
            
        )

    else:
        processor = AutoImageProcessor.from_pretrained(
            model_path,
            use_fast=True,
            device_map=dev_name
        )

    if dataset_cfg.name == 'cifar100':
        num_labels = 100
        problem_type = "single_label_classification"
    elif dataset_cfg.name == 'celeba':
        num_labels = 40
        problem_type = "multi_label_classification"
    else:
        raise ValueError('Wrong dataset for vision model!')

    model = AutoModelForImageClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        problem_type=problem_type,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    if dataset_cfg.model == 'tinynet':
        freeze_params(model, '^((?!blocks\.6\.0\.se\.conv_exp)(?!_model\.classifier).)*$')
    else:
        peft_config = LoraConfig(
            inference_mode=False,
            r=20,
            lora_alpha=40,
            target_modules=["project_conv"],
            modules_to_save=["classifier"],
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)

    print("num_trainable_params_vision:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, processor


# module-level collate function for vision models
class VisionCollator:
    """Collate function for vision models that can be pickled for multiprocessing"""
    def __init__(self, processor):
        self.processor = processor
        self.attribute_names = None
        if dataset_cfg.name == 'celeba':
            self.tags = ['image', 'celeb_id']
            self.label_dtype = torch.float32
        if dataset_cfg.name == 'cifar100':
            self.tags = ['img', 'coarse_label'] # 'fine_label'
            self.label_dtype = torch.long
    
    def __call__(self, examples):
        if self.attribute_names is None:
            self.attribute_names = [k for k in examples[0].keys() 
                                   if k not in self.tags]

        # handle both preprocessed tensor data and raw image data
        if isinstance(examples[0].get('pixel_values', None), torch.Tensor):
            pixel_values = torch.stack([example['pixel_values'] for example in examples])
            labels = torch.tensor([[example[attr] for attr in self.attribute_names] for example in examples], dtype=self.label_dtype)
            return {'pixel_values': pixel_values, 'labels': labels}
        else:
            images = []
            for example in examples:
                img = example.get(self.tags[0], example.get('pixel_values', None))
                if isinstance(img, dict) and 'bytes' in img:
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(img['bytes']))
                images.append(img)
            labels = [[example[attr] for attr in self.attribute_names] for example in examples]
            inputs = self.processor(images=images, return_tensors='pt')
            inputs['labels'] = torch.tensor(labels, dtype=self.label_dtype)
            return inputs


def train_model_vision(cid):
    
    use_torch_seed(int(cid))
    
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    DEVICE = get_device_per_client(cid)
    
    if dataset_cfg.name == 'celeba':
        trainloader = torch.load(Path('dataset/celeba-processed/federated') / cid / ("train" + ".pt"), weights_only=False)
    elif dataset_cfg.name == 'cifar100':
        trainloader = torch.load(Path('dataset/cifar100-processed/federated') / cid / ("train" + ".pt"), weights_only=False)
    else:
        raise ValueError('Wrong dataset for vision model!')

    model, processor = get_vision_model(cid)
    model.to(DEVICE)

    gm, sh = load_global_model()
    set_flat_model(model, gm, sh, dataset_cfg.params)

    temp_output_dir = mkdtemp(suffix=None, prefix='temp_dir_dp_vision_client_exp', dir=None)

    collate_fn = VisionCollator(processor)

    if dataset_cfg.params['learning_rate'] < 0:
        if dataset_cfg.name == 'cifar100':
            l_r = 1e-3
        else:
            l_r = 3e-5
    else:
        l_r = dataset_cfg.params['learning_rate']

    train_args = TrainingArguments(
        gradient_accumulation_steps=2,
        dataloader_num_workers=2,
        learning_rate=l_r,
        logging_strategy="no",
        num_train_epochs=1,
        optim='adamw_torch',
        output_dir=temp_output_dir,
        log_level='warning',
        per_device_train_batch_size=14,
        per_device_eval_batch_size=14,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=trainloader,
        args=train_args,
        data_collator=collate_fn,
    )
    trainer.train()
    shutil.rmtree(train_args.output_dir, ignore_errors=True)

    parameters = state_dict_2_params(model)
    params = Parameters(parameters)

    # Free GPU memory
    model.to('cpu')
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return params.flat, params.shape


def test_vision_model(model, processor, testloader, device, test_batch_size=64):
    def compute_metrics(p: EvalPrediction):
        if dataset_cfg.name == 'celeba':
            # Hamming accuracy (mean accuracy across all labels) for multi-label classification
            preds = (torch.sigmoid(torch.tensor(p.predictions)) > 0.5).float().numpy()
            labels = p.label_ids
            accuracy = (preds == labels).mean()
        else:
            # Top-1 accuracy for single-label classification
            preds = np.argmax(p.predictions, axis=1)
            labels = p.label_ids.squeeze() if p.label_ids.ndim > 1 else p.label_ids
            accuracy = (preds == labels).mean()
        return {'accuracy': accuracy}

    temp_dir = mkdtemp(suffix=None, prefix='temp_dir_dp_vision_eval_exp', dir=None)
    train_args = TrainingArguments(
        output_dir=temp_dir,
        per_device_eval_batch_size=test_batch_size,
        dataloader_num_workers=0,
        log_level='warning',
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    collate_fn = VisionCollator(processor)

    trainer = Trainer(
        model=model,
        args=train_args,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    results = trainer.evaluate(eval_dataset=testloader)
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Free GPU memory
    model.to('cpu')
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results['eval_loss'], results['eval_accuracy']


def save_initial_vision_model():
    temp_dir = dataset_cfg.temp_dir
    if not Path(temp_dir).exists():
        Path(temp_dir).mkdir(parents=True)

    # save the entire pretrained model to disk
    net, processor = get_vision_model()
    net.save_pretrained(os.path.join(temp_dir, 'pretrained_model'))
    processor.save_pretrained(os.path.join(temp_dir, 'pretrained_model'))
    
    parameters = state_dict_2_params(net)
    params = Parameters(parameters)
    model_flat = params.flat
    model_shape = params.shape
    with open(f'{temp_dir}/{dataset_cfg.model_file_name}', 'wb') as f:
        pickle.dump((model_flat, model_shape), f)


def get_model_size(params):
    model_name = dataset_cfg.model
    DEVICE = get_device_per_client()
    if model_name == 'cnn_mnist' or model_name == 'cnn_cifar10':
        net = model.Net().to(DEVICE)
        parameters = get_parameters(net, {})
    elif model_name == 'nlp':
        net, tokenizer = get_nlp_model()
        parameters = state_dict_2_params(net)
    elif model_name == 'efficientnet' or model_name == 'tinynet':
        net, processor = get_vision_model()
        parameters = state_dict_2_params(net)
    else:
        raise ValueError('Wrong model name!')

    params = Parameters(parameters)
    model_flat = params.flat
    model_shape = params.shape
    return len(model_flat), model_shape
