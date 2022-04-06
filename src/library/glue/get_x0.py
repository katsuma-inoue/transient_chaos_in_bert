"""The implementation of gettin x0 states on each data."""
import dataclasses
import logging
import os
import json
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn

from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, GlueDataset,
                          GlueDataTrainingArguments, TrainingArguments,
                          HfArgumentParser, glue_tasks_num_labels,
                          glue_output_modes)

from src.glue.trainer import FinetuneTrainer


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help":
            "Path to pretrained model or model identifier from huggingface.co/models"
        })
    num_layer: int = field(
        metadata={"help": "The custom number of layers of Albert"})

    init_model: int = field(
        metadata={
            "help": "Whether to use randomly initialized parameters of ALBERT."
        },
        default=0,
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where do you want to store the pretrained models downloaded from s3"
        })


@dataclass
class SaveArguments:
    """Arguments for saving x0 states"""
    save_dir: str = field(metadata={"help": "Path to save the x0 states."})


logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():

    parser = HfArgumentParser(
        (ModelArguments, GlueDataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Setup model
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    logger.info(f"Specified num hidden layers : {model_args.num_layer}")
    config.update({"num_hidden_layers": model_args.num_layer})
    config.update({"output_hidden_states": True})
    logger.info(f"model configuration : {config}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    logger.info(f"Loading pre-trained model.")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    # Get datasets
    train_dataset = (GlueDataset(data_args, tokenizer=tokenizer))
    eval_dataset = (GlueDataset(data_args, tokenizer=tokenizer, mode='dev'))

    def reset_parameters(model):
        """Implement reset_parameters according to nn.Linear.
        See also
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        """
        for param in model.parameters():
            bound = 1 / math.sqrt(param.size(-1))
            if param.ndim == 1:
                nn.init.uniform_(param, -bound, bound)
            else:
                nn.init.orthogonal_(param, gain=1)

    if model_args.init_model == 1:
        logger.info(f"Initializing parameters in AlbertModel")
        reset_parameters(model.albert)

    trainer = FinetuneTrainer(model=model, args=training_args)
    result = trainer.get_x0s(dataset=eval_dataset,
                             tokenizer=tokenizer,
                             save_dir=training_args.output_dir,
                             mode="eval")
    result = trainer.get_x0s(dataset=train_dataset,
                             tokenizer=tokenizer,
                             save_dir=training_args.output_dir,
                             mode="train")


if __name__ == "__main__":
    main()