# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionWithSTIntraConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    alpha: float = field(
        default=0.0,
        metadata={"help": "alpha for intra-modal consistency regularization"},
    )
    src_id: int = field(
        default=10000,
        metadata={"help": "source language id"},
    )
    tgt_id: int = field(
        default=10001,
        metadata={"help": "target language id"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion(
    "label_smoothed_cross_entropy_with_st_intra", dataclass=LabelSmoothedCrossEntropyCriterionWithSTIntraConfig
)
class LabelSmoothedCrossEntropyWithSTIntraCriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            alpha=0.0,
            src_id=10000,
            tgt_id=10001,
            ignore_prefix_size=0,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.alpha = alpha
        self.src_id = src_id
        self.tgt_id = tgt_id

    def intraconst(self, model, sample, reduce):
        valid_indices = (sample["target"] != self.padding_idx)

        # ST 1st Pass
        encoder_out = model.encoder(
            src_tokens=sample["net_input"]["audio"],
            src_lengths=sample["net_input"]["audio_lengths"],
            job_type='st')
        decoder_out_1 = model.decoder(
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            encoder_out=encoder_out)

        st_loss, st_nll_loss = self.compute_loss(model, decoder_out_1, sample, reduce=reduce)

        if self.alpha == 0.0:
            return st_loss, st_nll_loss

        st_prob_1 = torch.nn.functional.softmax(decoder_out_1[0], dim=-1)

        # ST 2nd Pass
        encoder_out = model.encoder(
            src_tokens=sample["net_input"]["audio"],
            src_lengths=sample["net_input"]["audio_lengths"],
            job_type='st')
        decoder_out_2 = model.decoder(
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            encoder_out=encoder_out)

        st_prob_2 = torch.nn.functional.softmax(decoder_out_2[0], dim=-1)

        st_kl_loss_1 = torch.nn.functional.kl_div(
            input=torch.nn.functional.log_softmax(decoder_out_2[0], dim=-1),
            target=st_prob_1, reduction='none')
        st_kl_loss_1 = st_kl_loss_1.sum(dim=-1)
        st_kl_loss_1 = st_kl_loss_1 * valid_indices.float()
        if reduce:
            st_kl_loss_1 = st_kl_loss_1.sum()

        st_kl_loss_2 = torch.nn.functional.kl_div(
            input=torch.nn.functional.log_softmax(decoder_out_1[0], dim=-1),
            target=st_prob_2, reduction='none')
        st_kl_loss_2 = st_kl_loss_2.sum(dim=-1)
        st_kl_loss_2 = st_kl_loss_2 * valid_indices.float()
        if reduce:
            st_kl_loss_2 = st_kl_loss_2.sum()

        if model.training:
            return st_loss + self.alpha * (st_kl_loss_1 + st_kl_loss_2) / 2.0, st_nll_loss
        else:
            return st_loss, st_nll_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample["net_input"]["prev_output_tokens"][:, 0] = self.tgt_id

        loss, nll_loss = self.intraconst(model, sample, reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
