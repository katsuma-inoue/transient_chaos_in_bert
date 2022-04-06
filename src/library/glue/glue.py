# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE
"""

import dataclasses
import logging
import os
import json
import math
from dataclasses import dataclass, field
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import sys

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
    GlueDataTrainingArguments,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)


sys.path.append(".")

from src.library.glue.classifier import Classifier
from src.library.glue.trainer import FinetuneTrainer

logger = logging.getLogger(__name__)


def _initialize_truncated_normal(param: torch.Tensor,
                                 stddev: float = 0.02,
                                 upper_bound: float = 2.0,
                                 lower_bound: float = -2.0):
    upper_bound = torch.Tensor([upper_bound])
    lower_bound = torch.Tensor([lower_bound])
    param_shape = param.size()
    Uniform_distribution = torch.distributions.uniform.Uniform(0, 1)
    u = Uniform_distribution.sample(param.size())

    Fa = 0.5 * (1 + torch.erf(lower_bound / np.sqrt(2)))
    Fb = 0.5 * (1 + torch.erf(upper_bound / np.sqrt(2)))
    with torch.no_grad():
        param.copy_(
            np.sqrt(2) * torch.erfinv(2 * ((Fb - Fa) * u + Fa) - 1) * stddev)



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
    fix_encoder: int = field(metadata={
        "help":
        "Whether or not to fix the parameters of ALBERT encoder layers"
    },
        default=1)
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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, GlueDataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # to save evaluate results in each epoch,
    # we should change the training args.
    # training_args.num_train_epochs = 1

    if (os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    logger.info(f"Specified num hidden layers : {model_args.num_layer}")
    config.update({"num_hidden_layers": model_args.num_layer})
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
    model.classifier = nn.Linear(config.hidden_size, config.num_labels)
    # model.classifier = Classifier(config.hidden_size, config.num_labels)

    # Get datasets
    train_dataset = (
        GlueDataset(
            data_args,
            tokenizer=tokenizer)  # local_rank=training_args.local_rank)
        if training_args.do_train else None)
    eval_dataset = (
        GlueDataset(
            data_args,
            tokenizer=tokenizer,
            # local_rank=training_args.local_rank,
            mode='dev') if training_args.do_eval else None)

    def compute_metrics(p: EvalPrediction) -> Dict:
        logger.info(f"The shape of p.predictions: {p.predictions.shape}")
        logger.info(f"The shape of p.label_ids: {p.label_ids.shape}")
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

    def set_parameter_requires_grad(model):
        for param in model.parameters():
            param.requires_grad = False

    def reset_parameters(model):
        """Implement reset_parameters
        when parameters dimension of two, reset them by xavier uniform, or glorot uniform.
        when parameters has dimension of one, reset them by uniform distribution.
        """
        for name, param in model.named_parameters():
            # if param.ndim == 2:
            logger.info(f"before: {name} {param.flatten()[0]}")
            if name[-6:] == "weight":
                _initialize_truncated_normal(param, stddev=0.02)
            else:  # For bias, initialize parameters with zero.
                nn.init.zeros_(param)
            logger.info(f"after: {name} {param.flatten()[0]}")

    logger.info(f"Fix encoder :  {model_args.fix_encoder}")
    if model_args.fix_encoder == 1:
        set_parameter_requires_grad(model.albert)
    if model_args.init_model == 1:
        logger.info(f"Initializing parameters in AlbertModel")
        reset_parameters(model.albert)
    logger.info(f"Initializing parameters in Classifier")
    reset_parameters(model.classifier)
    for param in model.parameters():
        print("name: ", param.shape, "Requires_grad: ", param.requires_grad)
    # Initialize our Trainer
    if model_args.fix_encoder == 1 or model_args.init_model == 1:
        trainer = FinetuneTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
    # Training
    if training_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path if os.path.
                      isdir(model_args.model_name_or_path) else None)
        trainer.save_model()
        # model_args.model_name_or_path = training_args.output_dir
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args,
                                                    task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args,
                            tokenizer=tokenizer,
                            local_rank=training_args.local_rank,
                            mode='dev'))

        for eval_dataset in eval_datasets:
            result = trainer.evaluate(eval_dataset=eval_dataset)
            preds = trainer.predict(test_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir,
                f"eval_results_{eval_dataset.args.task_name}.txt")
            output_preds = os.path.join(training_args.output_dir, f"preds.csv")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(
                    eval_dataset.args.task_name))
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)
            results["num_layer"] = model_args.num_layer
            result_file = os.path.join(
                training_args.output_dir,
                f"eval_results_{model_args.num_layer}.json")
            with open(result_file, "w") as f:
                string = json.dumps(results, indent=4)
                f.write(string)

            with open(output_preds, "w") as f:
                f.write("idx, pred, label_id\n")
                for idx, (pred, label_id) in enumerate(
                        zip(preds.predictions.flatten(),
                            preds.label_ids.flatten())):
                    f.write(f"{idx}, {pred} {label_id}\n")
    return results


if __name__ == "__main__":
    main()
