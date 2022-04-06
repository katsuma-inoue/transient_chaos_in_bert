from dataclasses import dataclass, field
from typing import Dict, Optional
import os
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    DefaultDataCollator,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    GlueDataset,
    GlueDataTrainingArguments,
    Trainer,
)
import torch
import math


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=f"albert-large-v2",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def load_glue_pretrained_models(
        task_name, num_layer,
        take_steps=None,
        data_dir="/data/glue_data",
        root_dir="./out/glue",
        max_seq_len=128,
        output_hidden_states=True):
    """Load finetuned models on GLUE tasks.

    Parameter:
    =========
    task_name(str) : name of task. {"CoLA", "MNLI", "MRPC",
                        "SST-2", "STS-B", "QQP", "QNLI", "RTE", "WNLI"}
    num_layer(int) : Number of layers used during finetuning.
    take_steps(int) : How many times you run the emuration.
    data_dir(str) : The base directroy for glue dataset.
    root_dir(str) : The base directory for saved models.
    max_seq_len(str): max_seq_length.
    output_hidden_states(bool): Whether to get hidden states.
    """
    if take_steps is None:
        take_steps = num_layer
    model_path = os.path.join(root_dir, f"{task_name}/{num_layer}_2")
    data_dir = os.path.join(data_dir, task_name)
    model_args = ModelArguments()
    model_args.model_name_or_path = model_path
    data_args = GlueDataTrainingArguments(
        task_name='CoLA', data_dir=data_dir, max_seq_length=max_seq_len)
    num_labels = glue_tasks_num_labels[data_args.task_name]
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task='CoLA',
        cache_dir=model_args.cache_dir,
    )
    config.update({
        "output_hidden_states": output_hidden_states
    })
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model.config.num_hidden_layers = take_steps  # change the number of layers
    eval_dataset = GlueDataset(
        data_args, tokenizer=tokenizer, local_rank=-1, evaluate=True)
    return model, tokenizer, eval_dataset


def get_batch(ids, dataset, tokenizer):
    """Get batch from dataset
    """
    labels = []
    input_ids = []
    attention_mask = []
    token_type_ids = []
    original_sentence = []
    for id in ids:
        org_sentence = tokenizer.decode(dataset[id].input_ids).replace("<pad>", "")
        original_sentence.append(org_sentence)
        labels.append(dataset[id].label)
        input_ids.append(dataset[id].input_ids)
        attention_mask.append(dataset[id].attention_mask)
        token_type_ids.append(dataset[id].token_type_ids)
    batch = {
        'labels': torch.tensor(labels, dtype=torch.int64),
        'input_ids': torch.tensor(input_ids, dtype=torch.int64),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.int64),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.int64)
    }
    return batch, original_sentence


def load_pretrained_model(
        model_name, output_hidden_states=True, initial_model=False,
        masked_lm=False):
    """Load pretrained ALBERT model from Transformers.

    Parameters:
    =========
    model_name (str or path): Model name or model path.
    output_hidden_states (boolean): Whether to output hidden state of model.
    initial_model (boolean): If True, this function will return randomly
        initialized parameters.
    masked_lm (boolean): If True, this function will return AlbertMaskedLM

    Returns:
    ========
    model (AlbertModel): Loaded model.
    tokenizer (AlbertTokenizer): Loaded Tokenizer.
    """
    model_args = ModelArguments(model_name_or_path=model_name)
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=model_args.cache_dir,
    )
    config.update({
        "output_hidden_states": output_hidden_states
    })
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    if initial_model and not masked_lm:
        model = AutoModel.from_config(config)
    elif not masked_lm:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config)
    elif initial_model:
        model = AutoModelWithLMHead.from_config(config)
    else:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config
        )
    return model, tokenizer


def reset_parameters(model):
        """Implement reset_parameters according to nn.Linear.
        See also
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        """
        for param in model.parameters():
            bound = 1 / math.sqrt(param.size(-1))
            torch.nn.init.uniform_(param, -bound, bound)


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False
