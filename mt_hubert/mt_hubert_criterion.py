# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional
from torch import nn
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class HubertCriterionConfig(FairseqDataclass):
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "weight for predictive loss for masked frames"},
    )
    pred_nomask_weight: float = field(
        default=0.0,
        metadata={"help": "weight for predictive loss for unmasked frames"},
    )
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )


@register_criterion("hubert", dataclass=HubertCriterionConfig)
class HubertCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        pred_masked_weight,
        pred_nomask_weight,
        loss_weights=None,
        log_keys=None,
    ):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys
        self.prob_factor = nn.Sigmoid()

    def reshape_logit_list(self, logp_list, tar_label_list, nontar_label_list):
        if nontar_label_list == "None":
            for i, tar_label in enumerate(tar_label_list):
                logp = logp_list[i]
                batch_indices = torch.arange(tar_label.size(0), device=tar_label.device)
                logp[batch_indices, tar_label + 1] = logp[batch_indices, 0]
                logp_list[i] = logp[:, 1:]
            #     for j in range(tar_label.size(0)):
            #         logp_list[i][j, tar_label[j] + 1] = logp_list[i][j, 0]
            # logp_list = [logp[:, 1:] for logp in logp_list]
        else:
            for i, (tar_label, nontar_label) in enumerate(zip(tar_label_list, nontar_label_list)):
                assert tar_label.size(0) == nontar_label.size(0)
                logp = logp_list[i]
                batch_indices = torch.arange(tar_label.size(0), device=tar_label.device)
                same_label = nontar_label == tar_label
                diff_label = ~same_label
                logp[batch_indices[diff_label], tar_label[diff_label] + 2] = logp[batch_indices[diff_label], 0]
                logp[batch_indices[diff_label], nontar_label[diff_label] + 2] = logp[batch_indices[diff_label], 1]
                logp[batch_indices[same_label], tar_label[same_label] + 2] = (
                        logp[batch_indices[same_label], 0] + logp[batch_indices[same_label], 1]
                )
                logp_list[i] = logp[:, 2:]
            #     for j in range(tar_label.size(0)):
            #         logp_list[i][j, tar_label[j] + 2] = logp_list[i][j, 0]
            #         logp_list[i][j, nontar_label[j] + 2] = logp_list[i][j, 1]
            # logp_list = [logp[:, 2:] for logp in logp_list]

        return logp_list

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(target_list=sample["target_list"], **sample["net_input"])
        loss = 0.0
        c_sample_size, m_sample_size, sample_size = 0, 0, 0
        logging_output = {}
        reduction = "sum" if reduce else "none"

        c_loss_m_list, m_loss_m_list = [], []
        c_logp_m_list, m_logp_m_list = model.get_logits(net_output, True)
        c_targ_m_list, m_targ_m_list, c_m_label_list, tar_m_label_list, nontar_m_label_list = model.get_targets(net_output, True)
        assert self.pred_masked_weight == 0 or len(c_logp_m_list) > 0 or len(m_logp_m_list) > 0
        c_logp_m_list = self.reshape_logit_list(c_logp_m_list, c_m_label_list, "None")
        m_logp_m_list = self.reshape_logit_list(m_logp_m_list, tar_m_label_list, nontar_m_label_list)
        for i, (c_logp_m, m_logp_m, c_targ_m, m_targ_m) in enumerate(zip(c_logp_m_list, m_logp_m_list, c_targ_m_list, m_targ_m_list)):
            c_loss_m = F.binary_cross_entropy_with_logits(c_logp_m, c_targ_m, reduction=reduction)
            m_loss_m = F.binary_cross_entropy_with_logits(m_logp_m, m_targ_m, reduction=reduction)
            c_loss_m_list.append(c_loss_m)
            m_loss_m_list.append(m_loss_m)
            logging_output[f"clean_loss_m_{i}"] = c_loss_m.detach().item()
            logging_output[f"mixed_loss_m_{i}"] = m_loss_m.detach().item()
        if self.pred_masked_weight > 0:
            loss += self.pred_masked_weight * (sum(c_loss_m_list) + sum(m_loss_m_list))
            c_sample_size += c_targ_m_list[0].size(0)
            m_sample_size += m_targ_m_list[0].size(0)
            sample_size += (c_targ_m_list[0].size(0) + m_targ_m_list[0].size(0))

        c_loss_u_list, m_loss_u_list = [], []
        c_logp_u_list, m_logp_u_list = model.get_logits(net_output, False)
        c_targ_u_list, m_targ_u_list, c_u_label_list, tar_u_label_list, nontar_u_label_list = model.get_targets(net_output, False)
        assert self.pred_nomask_weight == 0 or len(c_logp_u_list) > 0 or len(m_logp_u_list) > 0
        c_logp_u_list = self.reshape_logit_list(c_logp_u_list, c_u_label_list, "None")
        m_logp_u_list = self.reshape_logit_list(m_logp_u_list, tar_u_label_list, nontar_u_label_list)
        for i, (c_logp_u, m_logp_u, c_targ_u, m_targ_u) in enumerate(zip(c_logp_u_list, m_logp_u_list, c_targ_u_list, m_targ_u_list)):
            c_loss_u = F.binary_cross_entropy_with_logits(c_logp_u, c_targ_u, reduction=reduction)
            m_loss_u = F.binary_cross_entropy_with_logits(m_logp_u, m_targ_u, reduction=reduction)
            c_loss_u_list.append(c_loss_u)
            m_loss_u_list.append(m_loss_u)
            logging_output[f"clean_loss_u_{i}"] = c_loss_u.detach().item()
            logging_output[f"mixed_loss_u_{i}"] = m_loss_u.detach().item()
        if self.pred_nomask_weight > 0:
            loss += self.pred_nomask_weight * (sum(c_loss_u_list) + sum(m_loss_u_list))
            c_sample_size += c_targ_u_list[0].size(0)
            m_sample_size += m_targ_u_list[0].size(0)
            sample_size += (c_targ_u_list[0].size(0) + m_targ_u_list[0].size(0))

        # if self.loss_weights is not None:
        #     assert hasattr(model, "get_extra_losses")
        #     extra_losses, names = model.get_extra_losses(net_output)
        #     if torch.is_tensor(extra_losses):
        #         extra_losses = [extra_losses]
        #         names = [names]
        #     if len(self.loss_weights) == 1 and len(extra_losses) != 1:
        #         self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
        #     assert len(extra_losses) == len(
        #         self.loss_weights
        #     ), f"{len(extra_losses)}, {len(self.loss_weights)}"
        #     for p, n, coef in zip(extra_losses, names, self.loss_weights):
        #         if coef != 0 and p is not None:
        #             p = coef * p.float() * sample_size
        #             loss += p
        #             logging_output[f"loss_{n}"] = p.item()

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "clean_sample_size": c_sample_size,
            "mixed_sample_size": m_sample_size,
            "sample_size": sample_size,
            **logging_output,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        def compute_correct(logits, targ, nontarg):
            if logits.numel() == 0:
                return 0, 0
            else:
                assert logits.dim() > 1, logits.shape
                logits = self.prob_factor(logits)
                if nontarg == "None":
                    _, top1_indices = torch.topk(logits, k=1, dim=-1)
                    corr, count = 0, top1_indices.numel()
                    for idx in range(targ.size(0)):
                        tar_label = targ[idx].item()
                        log_label = top1_indices[idx].item()
                        if tar_label == log_label:
                            corr += 1
                else:
                    _, top2_indices = torch.topk(logits, k=2, dim=-1)
                    corr, count = 0, top2_indices.numel()
                    for idx in range(targ.size(0)):
                        tar_label = targ[idx].item()
                        nontar_label = nontarg[idx].item()
                        log_labels = top2_indices[idx, :].tolist()
                        if tar_label in log_labels:
                            corr += 1
                        if nontar_label in log_labels:
                            corr += 1

                return corr, count

        with torch.no_grad():
            for i, (c_logp_m, m_logp_m, c_m, t_m, nt_m) in enumerate(zip(c_logp_m_list, m_logp_m_list, c_m_label_list,
                                                                         tar_m_label_list, nontar_m_label_list)):
                c_corr_m, c_count_m = compute_correct(c_logp_m, c_m, "None")
                m_corr_m, m_count_m = compute_correct(m_logp_m, t_m, nt_m)
                logging_output[f"clean_correct_m_{i}"] = c_corr_m
                logging_output[f"clean_count_m_{i}"] = c_count_m
                logging_output[f"mixed_correct_m_{i}"] = m_corr_m
                logging_output[f"mixed_count_m_{i}"] = m_count_m

            for i, (c_logp_u, m_logp_u, c_u, t_u, nt_u) in enumerate(zip(c_logp_u_list, m_logp_u_list, c_u_label_list,
                                                                         tar_u_label_list, nontar_u_label_list)):
                c_corr_u, c_count_u = compute_correct(c_logp_u, c_u, "None")
                m_corr_u, m_count_u = compute_correct(m_logp_u, t_u, nt_u)
                logging_output[f"clean_correct_u_{i}"] = c_corr_u
                logging_output[f"clean_count_u_{i}"] = c_count_u
                logging_output[f"mixed_correct_u_{i}"] = m_corr_u
                logging_output[f"mixed_count_u_{i}"] = m_count_u

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        c_sample_size = sum(log.get("clean_sample_size", 0) for log in logging_outputs)
        m_sample_size = sum(log.get("mixed_sample_size", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=6
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("clean_count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val
            if lk.startswith("mixed_count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("clean_loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / c_sample_size / math.log(2), round=6)
            elif lk.startswith("mixed_loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / m_sample_size / math.log(2), round=6)
            elif lk.startswith("clean_correct_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])
            elif lk.startswith("mixed_correct_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
