import torch
import torch.nn.functional as F
from collections.abc import Callable
from typing import Optional

from datasets import Dataset
from torch import nn
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from data_collator import SelfDistillationDataCollator
from opsd_trainer import OPSDTrainer
from trl.experimental.gold.gold_config import GOLDConfig


if is_peft_available():
    from peft import PeftConfig


class AnchoredSelfDistillationDataCollator(SelfDistillationDataCollator):
    """Extends OPSD collation with a reference CE branch on solution tokens."""

    def __call__(self, features):
        result = super().__call__(features)

        reference_texts = []
        for feature in features:
            problem = feature["problem"]
            solution = feature["solution"]

            student_user_message = (
                f"Problem: {problem}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
            student_messages = [{"role": "user", "content": student_user_message}]
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            reference_texts.append(student_prompt + solution)

        reference_encoded = self.tokenizer(
            reference_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        reference_labels = reference_encoded["input_ids"].clone()
        prompt_lengths = result["student_prompt_lengths_per_example"].tolist()
        for i, prompt_len in enumerate(prompt_lengths):
            reference_labels[i, : min(prompt_len, reference_labels.shape[1])] = -100

        if self.tokenizer.pad_token_id is not None:
            reference_labels[reference_labels == self.tokenizer.pad_token_id] = -100

        result.update(
            {
                "reference_input_ids": reference_encoded["input_ids"],
                "reference_attention_mask": reference_encoded["attention_mask"],
                "reference_labels": reference_labels,
            }
        )
        return result


class AnchoredOPSDTrainer(OPSDTrainer):
    """
    OPSD with an additional teacher-forced reference CE term on `solution`.

    Total loss:
        loss = opsd_loss + reference_ce_weight * reference_ce_loss
    """

    def __init__(
        self,
        model: nn.Module | str | None = None,
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
        use_thinking_machines_loss: bool = False,
        fixed_teacher: bool = False,
        reason_first: bool = False,
        top_k_loss: int | None = None,
        jsd_token_clip: float | None = None,
        use_ema_teacher: bool = False,
        ema_decay: float = 0.999,
        reference_ce_weight: float = 1.0,
    ):
        if reference_ce_weight < 0:
            raise ValueError("reference_ce_weight must be non-negative.")

        self.reference_ce_weight = reference_ce_weight

        if data_collator is None:
            data_collator = AnchoredSelfDistillationDataCollator(
                tokenizer=processing_class,
                max_length=getattr(args, "max_length", 2048),
                reason_first=reason_first,
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            use_thinking_machines_loss=use_thinking_machines_loss,
            fixed_teacher=fixed_teacher,
            reason_first=reason_first,
            top_k_loss=top_k_loss,
            jsd_token_clip=jsd_token_clip,
            use_ema_teacher=use_ema_teacher,
            ema_decay=ema_decay,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base_result = super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )
        if return_outputs:
            opsd_loss, outputs = base_result
        else:
            opsd_loss = base_result
            outputs = None

        if self.reference_ce_weight == 0:
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["opsd_loss"].append(float(opsd_loss.detach()))
            self._metrics[mode]["reference_ce_loss"].append(0.0)
            self._metrics[mode]["reference_ce_weight"].append(self.reference_ce_weight)
            if return_outputs:
                outputs.loss = opsd_loss
                return opsd_loss, outputs
            return opsd_loss

        reference_outputs = model(
            input_ids=inputs["reference_input_ids"],
            attention_mask=inputs["reference_attention_mask"],
        )
        reference_logits = reference_outputs.logits[:, :-1, :].contiguous()
        shifted_reference_labels = inputs["reference_labels"][:, 1:].contiguous()

        reference_ce_loss = F.cross_entropy(
            reference_logits.view(-1, reference_logits.size(-1)),
            shifted_reference_labels.view(-1),
            ignore_index=-100,
        )

        loss = opsd_loss + self.reference_ce_weight * reference_ce_loss

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["opsd_loss"].append(float(opsd_loss.detach()))
        self._metrics[mode]["reference_ce_loss"].append(float(reference_ce_loss.detach()))
        self._metrics[mode]["reference_ce_weight"].append(self.reference_ce_weight)

        if return_outputs:
            outputs.loss = loss
            return loss, outputs
        return loss
