"""The wrapper trainer class from transformers v2.10.0
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union
import logging
from tqdm.auto import tqdm, trange
import os
import gc
from transformers import Trainer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import PreTrainedTokenizer

_has_apex = False
_has_wandb = False


def is_wandb_available():
    return _has_wandb


def is_apex_available():
    return _has_apex


logger = logging.getLogger(__name__)


class PredictionOutput(NamedTuple):
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used
    to compute metrics.
    """

    predictions: np.ndarray
    label_ids: np.ndarray


class FinetuneTrainer(Trainer):
    def __init__(self, train_encoder=False, **kwargs):
        super().__init__(**kwargs)
        self.optimizers = None
        self.train_encoder = train_encoder

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        if self.train_encoder:
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=self.args.learning_rate,
                              eps=self.args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps)
        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda x: 0.9995**x)

        return optimizer, scheduler

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        prediction_loss_only: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.
        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        output = self._prediction_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=prediction_loss_only)

        self._log(output.metrics)

        return output.metrics

    def _prediction_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(
                inputs.get(k) is not None
                for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                outputs = tuple(output.to("cpu") for output in outputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.cpu().detach()
                else:
                    preds = torch.cat((preds, logits.cpu().detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].cpu().detach()
                    else:
                        label_ids = torch.cat(
                            (label_ids, inputs["labels"].cpu().detach()),
                            dim=0)
                elif inputs.get("masked_lm_labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["masked_lm_labels"].cpu().detach()
                    else:
                        label_ids = torch.cat(
                            (label_ids,
                             inputs["masked_lm_labels"].cpu().detach()),
                            dim=0)
            del outputs
            gc.collect()

            torch.cuda.empty_cache()

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(
                    preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(
                    label_ids,
                    num_total_examples=self.num_examples(dataloader))

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()
        logger.info(f"preds: {preds.shape}, label_ids: {label_ids.shape}")
        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(
                EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        logger.info(f"The length of eval_loss {len(eval_losses)}")
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds,
                                label_ids=label_ids,
                                metrics=metrics)

    def get_x0s(self,
                tokenizer: PreTrainedTokenizer,
                save_dir: str,
                dataset: Optional[Dataset] = None,
                mode: Optional[str] = "training") -> Dict[str, float]:
        """
        Getting x0 vectors.
        Args:
            tokenizer: Pass a tokenizer to reconstruct the input sentences.
            dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the potential metrics computed from the predictions
        """
        eval_dataloader = self.get_eval_dataloader(dataset)
        self.model.eval()
        self._get_x0_loop(eval_dataloader,
                          tokenizer=tokenizer,
                          save_dir=save_dir,
                          description="Getting state vectors",
                          mode=mode)

    def _get_x0_loop(self,
                     dataloader: DataLoader,
                     description: str,
                     save_dir: str,
                     tokenizer: PreTrainedTokenizer,
                     mode: Optional[str] = "train") -> PredictionOutput:
        """
        Getting x0 states loop, shared by `evaluate()` and `predict()`.
        """
        os.makedirs(save_dir, exist_ok=True)
        model = self.model

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        label_ids: torch.Tensor = None
        model.eval()

        sample_number = 0
        input_texts = pd.DataFrame()
        scores = pd.DataFrame()
        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(
                inputs.get(k) is not None
                for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                label_ids = inputs.pop(
                    "labels").cpu().detach().numpy().tolist()
                outputs = model.albert(**inputs)
                # (sequence_output, pooled_output, state_outputs)
                input_ids = inputs["input_ids"]
                input_ids = input_ids.to("cpu").detach().numpy()
                tokens = []
                for idx in range(input_ids.shape[0]):
                    tokens.append(
                        tokenizer.convert_ids_to_tokens(
                            input_ids[idx].tolist()))
                states = tuple(
                    state.to("cpu").detach().numpy() for state in outputs[2])
                for i in range(batch_size):
                    state_save_dir = os.path.join(save_dir, mode,
                                                  f"{sample_number+i:04d}")
                    os.makedirs(state_save_dir, exist_ok=True)
                for idx, state in enumerate(states):
                    for i in range(batch_size):
                        pd.DataFrame(state[i]).to_csv(os.path.join(
                            save_dir, mode, f"{sample_number+i:04d}",
                            f"step{idx:02d}.csv"),
                                                      index=None)
                for i in range(batch_size):
                    input_texts[f"{sample_number+i:04d}"] = tokens[i]
                    scores[f"{sample_number+i:04d}"] = [label_ids[i]]
                sample_number += batch_size
            del outputs
            gc.collect()

            torch.cuda.empty_cache()
            input_texts.to_csv(os.path.join(save_dir, mode, "input_texts.csv"),
                               index=None)
            scores.T.to_csv(os.path.join(save_dir, mode, "scores.csv"))
