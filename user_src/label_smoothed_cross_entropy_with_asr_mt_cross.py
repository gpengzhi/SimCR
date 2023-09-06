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
class LabelSmoothedCrossEntropyCriterionWithASRMTCrossConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    beta: float = field(
        default=0.0,
        metadata={"help": "beta for cross-modal consistency regularization"},
    )
    src_id: int = field(
        default=10000,
        metadata={"help": "source language id"},
    )
    tgt_id: int = field(
        default=10001,
        metadata={"help": "target language id"},
    )
    eval_type: str = field(
        default='mt',
        metadata={"help": "which job to evaluate"}
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion(
    "label_smoothed_cross_entropy_with_asr-mt_cross", dataclass=LabelSmoothedCrossEntropyCriterionWithASRMTCrossConfig
)
class LabelSmoothedCrossEntropyWithASRMTCrossCriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            beta=0.0,
            src_id=10000,
            tgt_id=10001,
            eval_type='mt',
            ignore_prefix_size=0,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.beta = beta
        self.src_id = src_id
        self.tgt_id = tgt_id
        self.eval_type = eval_type

    def crossconst(self, model, sample, reduce):
        tgt_valid_indices = (sample["target"] != self.padding_idx)
        src_valid_indices = (sample["target_src"] != self.padding_idx)

        # MT 1st Pass
        encoder_out = model.encoder(
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            job_type='mt')
        mt_decoder_out_1 = model.decoder(
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            encoder_out=encoder_out)

        mt_loss, mt_nll_loss = self.compute_loss(model, mt_decoder_out_1, sample, reduce=reduce)

        # ASR 1st Pass
        encoder_out = model.encoder(
            src_tokens=sample["net_input"]["audio"],
            src_lengths=sample["net_input"]["audio_lengths"],
            job_type='st')
        asr_decoder_out_1 = model.decoder(
            prev_output_tokens=sample["net_input"]["prev_output_src_tokens"],
            encoder_out=encoder_out)

        asr_loss, asr_nll_loss = self.compute_loss(model, asr_decoder_out_1, sample, reduce=reduce, job_type='asr')
        asr_prob_1 = torch.nn.functional.softmax(asr_decoder_out_1[0], dim=-1)

        asr_ae_kl_loss = 0.0
        if self.beta != 0:
            # ASR AE 1st Pass
            encoder_out = model.encoder(
                src_tokens=sample["net_input"]["src_tokens"],
                src_lengths=sample["net_input"]["src_lengths"],
                job_type='mt')
            decoder_out = model.decoder(
                prev_output_tokens=sample["net_input"]["prev_output_src_tokens"],
                encoder_out=encoder_out)

            asr_ae_kl_loss = torch.nn.functional.kl_div(
                input=torch.nn.functional.log_softmax(decoder_out[0], dim=-1),
                target=asr_prob_1, reduction='none')
            asr_ae_kl_loss = asr_ae_kl_loss.sum(dim=-1)
            asr_ae_kl_loss = asr_ae_kl_loss * src_valid_indices.float()
            if reduce:
                asr_ae_kl_loss = asr_ae_kl_loss.sum()

        return mt_loss + asr_loss + self.beta * asr_ae_kl_loss, asr_nll_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample["net_input"]["prev_output_tokens"][:, 0] = self.tgt_id
        sample["net_input"]["prev_output_src_tokens"][:, 0] = self.src_id

        if model.training:
            loss, nll_loss = self.crossconst(model, sample, reduce)
        else:
            if self.eval_type == 'mt':
                encoder_out = model.encoder(
                    src_tokens=sample["net_input"]["src_tokens"],
                    src_lengths=sample["net_input"]["src_lengths"],
                    job_type='mt')
                decoder_out = model.decoder(
                    prev_output_tokens=sample["net_input"]["prev_output_tokens"],
                    encoder_out=encoder_out)
                loss, nll_loss = self.compute_loss(model, decoder_out, sample, reduce=reduce)
            if self.eval_type == 'st':
                encoder_out = model.encoder(
                    src_tokens=sample["net_input"]["audio"],
                    src_lengths=sample["net_input"]["audio_lengths"],
                    job_type='st')
                decoder_out = model.decoder(
                    prev_output_tokens=sample["net_input"]["prev_output_tokens"],
                    encoder_out=encoder_out)
                loss, nll_loss = self.compute_loss(model, decoder_out, sample, reduce=reduce)
            if self.eval_type == 'asr':
                encoder_out = model.encoder(
                    src_tokens=sample["net_input"]["audio"],
                    src_lengths=sample["net_input"]["audio_lengths"],
                    job_type='st')
                decoder_out = model.decoder(
                    prev_output_tokens=sample["net_input"]["prev_output_src_tokens"],
                    encoder_out=encoder_out)
                loss, nll_loss = self.compute_loss(model, decoder_out, sample, reduce=reduce, job_type='asr')

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

    def get_lprobs_and_target(self, model, net_output, sample, job_type=None):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # target = model.get_targets(sample, net_output)
        if job_type == 'asr':
            target = sample['target_src']
        else:
            target = sample['target']
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True, job_type=None):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample, job_type)
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
