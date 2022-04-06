"""Evaluating the performances of MLM task.
This implementation is based on huggingface's sample available at 
https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py
Please prepare the dataset in advance by running the `setup/script/make_mlm_data.py`.
"""
import json
import logging
import math
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.model import albert_utils
from src.library.utils.data_collator import DataCollatorForAlbertLanguageModeling
from src.library.glue.trainer import FinetuneTrainer

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer that we are going to fine-tune, or train from randomly initialized model including way of model initialization or type of model.
    """
    init_model: bool = field(
        default=False,
        metadata={"help": "Specifying whether or not to initialize layer parameters."})
    init_readout: bool = field(
        default=False,
        metadata={
            "help":
            "This option will initialize the prediction layer of ALBERT."
        })
    fix_encoder: bool = field(
        default=False,
        metadata={
            "help":
            "Specifying whether or not to fix the parameters of the encoder during training."
        })
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch. The default value initializes the model parameter."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Please choose the type from " +
            ", ".join(MODEL_TYPES) + ", otherwise, set model_name_or_path."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path. This argument will be used if config_name is not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path. This argument will be used if tokenizer_name is not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Specifying the path storing the pretrained models downloaded from s3"
        })
    num_layer: Optional[int] = field(
        default=24,
        metadata={"help": "Specifying the number of layers."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."})
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help":
            "Specifying whether distinct lines of text in the dataset are to be handled as distinct sequences"
        },
    )

    mlm: bool = field(
        default=False,
        metadata={
            "help":
            "Specifying whether to use masked-language modeling loss during training instead of language modeling one."
        })
    mlm_probability: float = field(
        default=0.10,
        metadata={
            "help": "Ratio of tokens to mask for masked language modeling loss"
        })

    block_size: int = field(
        default=-1,
        metadata={
            "help":
            "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "The default setup uses the maximum length for single sentence inputs (including special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"})


def get_dataset(args: DataTrainingArguments,
                tokenizer: PreTrainedTokenizer,
                evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer,
                                     file_path=file_path,
                                     block_size=args.block_size)
    else:
        return TextDataset(tokenizer=tokenizer,
                           file_path=file_path,
                           block_size=args.block_size,
                           overwrite_cache=args.overwrite_cache)


def acc_metrics(output):
    preds = output.predictions
    preds = np.argmax(preds, axis=2)
    labels = output.label_ids
    logger.info(f"shape of preds:{preds.shape}, labels:{labels.shape}")
    acc_list = []
    for (pred, label) in zip(preds, labels):
        pred = np.where(label == -100, -1, pred)
        num_correct = np.where(pred == label, 1., 0.).sum()
        num_mask = np.where(label == -100, 0., 1.).sum()
        _acc = num_correct / num_mask
        acc_list.append(_acc)
    acc = np.mean(acc_list)
    return {"acc": acc}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument.")

    if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir) and training_args.do_train
            and not training_args.overwrite_output_dir):
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

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    logger.info(f"Specified num hidden layers : {model_args.num_layer}")
    config.update({"num_hidden_layers": model_args.num_layer})
    logger.info(f"model configuration : \n {config}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir)

    logger.info("Loading Pretrained models...")
    model = AutoModelWithLMHead.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir)
    if model_args.init_model:
        logger.info("Initializing ALBERT encoder layers...")
        albert_utils.reset_parameters(model.albert)

    # initialize prediction layer
    if model_args.init_readout:
        albert_utils.reset_parameters(model.predictions)
    else:
        pass

    if model_args.fix_encoder:
        albert_utils.set_parameter_requires_grad(model.albert)
    model.resize_token_embeddings(len(tokenizer))

    for param in model.parameters():
        logger.info(
            f"name: {param.shape}, Requires_grad: {param.requires_grad}")

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Here, the block size will be set to the maximum length for the model.
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    train_dataset = get_dataset(
        data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(
        data_args, tokenizer=tokenizer,
        evaluate=True) if training_args.do_eval else None
    data_collator = DataCollatorForAlbertLanguageModeling(
        tokenizer=tokenizer,
        mlm=data_args.mlm,
        mlm_probability=data_args.mlm_probability)

    # Initialize the Trainer
    trainer = FinetuneTrainer(model=model,
                              args=training_args,
                              data_collator=data_collator,
                              train_dataset=train_dataset,
                              eval_dataset=eval_dataset,
                              prediction_loss_only=False,
                              compute_metrics=acc_metrics,
                              train_encoder=~model_args.fix_encoder)

    # Training
    if training_args.do_train:
        model_path = (model_args.model_name_or_path
                      if model_args.model_name_or_path is not None
                      and os.path.isdir(model_args.model_name_or_path) else
                      None)
        trainer.train(model_path=model_path)
        # trainer.save_model()
        # if trainer.is_world_master():
        #     tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate(prediction_loss_only=False)
        logger.info(f"The eval_output {eval_output}, type:{type(eval_output)}")
        logger.info(f"The keys of metrics {eval_output.keys()}")
        perplexity = math.exp(eval_output["eval_loss"])
        result = {
            "perplexity": perplexity,
            "acc": eval_output["eval_acc"],
            "eval_loss": eval_output["eval_loss"]
        }

        output_eval_file = os.path.join(training_args.output_dir,
                                        "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)
        results["num_layer"] = model_args.num_layer
        result_file = os.path.join(
            training_args.output_dir,
            f"eval_results_{model_args.num_layer}.json")
        with open(result_file, "w") as f:
            string = json.dumps(results, indent=4)
            f.write(string)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
