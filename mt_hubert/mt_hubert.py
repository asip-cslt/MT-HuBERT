# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
from omegaconf import II
from torch.nn import functional as F
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    LAYER_TYPE_CHOICES,
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingConfig,
    HubertPretrainingTask,
)

logger = logging.getLogger(__name__)


@dataclass
class HubertConfig(FairseqDataclass):
    label_rate: float = II("task.label_rate")

    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})


@register_model("hubert", dataclass=HubertConfig)
class HubertModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: HubertConfig,
        task_cfg: HubertPretrainingConfig,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__()
        logger.info(f"HubertModel Config: {cfg}")

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info("cannot find dictionary. assume will be used for fine-tuning")
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertConfig, task: HubertPretrainingTask):
        """Build a new model instance."""

        model = HubertModel(cfg, task.cfg, task.dictionaries)
        return model

    def apply_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def compute_nce(self, x, negs, tar_pos, nontar_pos):
        neg_is_tar_pos = (tar_pos == negs).all(-1)
        tar_pos = tar_pos.unsqueeze(0)
        if nontar_pos == "None":
            targets = torch.cat([tar_pos, negs], dim=0)
            logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)  # (num_cls+1, num_x)
            logits /= self.logit_temp
            if neg_is_tar_pos.any():
                logits[1:][neg_is_tar_pos] = float("-inf")
        else:
            neg_is_nontar_pos = (nontar_pos == negs).all(-1)
            nontar_pos = nontar_pos.unsqueeze(0)
            targets = torch.cat([tar_pos, nontar_pos, negs], dim=0)
            logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)  # (num_cls+2, num_x)
            logits /= self.logit_temp
            if neg_is_tar_pos.any():
                logits[2:][neg_is_tar_pos] = float("-inf")
            if neg_is_nontar_pos.any():
                logits[2:][neg_is_nontar_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+2) or (num_x, num_cls+1)

        return logits

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def prepare_nontar_speech(self, source: torch.Tensor):
        b, t = source.size()
        nontar_source = torch.empty_like(source)
        nontar_idx_list = []
        for i in range(b):
            # available_indices = [idx for idx in range(b) if idx != i and idx not in nontar_idx_list]
            available_indices = [idx for idx in range(b) if idx != i]
            sampled_idx = random.choice(available_indices)
            nontar_source[i] = source[sampled_idx]
            nontar_idx_list.append(sampled_idx)
        # mixed_source = source + nontar_source

        return nontar_source, nontar_idx_list

    def prepare_mixed_speech(self, source: torch.Tensor, nontar_source: torch.Tensor):
        assert source.size() == nontar_source.size()
        rms_s1 = torch.sqrt(torch.mean(source**2, dim=-1))
        rms_s2 = torch.sqrt(torch.mean(nontar_source**2, dim=-1))
        max_rms = torch.max(torch.stack([rms_s1, rms_s2], dim=-1), dim=-1)[0]
        max_rms = torch.where(max_rms <= 0.01, 1, max_rms)
        s1_norm = source * (max_rms / rms_s1).unsqueeze(-1)
        s2_norm = nontar_source * (max_rms / rms_s2).unsqueeze(-1)
        ratio_s1 = torch.rand(source.size(0), 1, device=source.device) * 0.8 + 0.1
        ratio_s2 = torch.rand(source.size(0), 1, device=source.device) * 0.8 + 0.1
        s1_scaled = s1_norm * ratio_s1
        s2_scaled = s2_norm * ratio_s2
        mixed_speech = s1_scaled + s2_scaled
        mixed_speech = mixed_speech.to(source.dtype)

        return mixed_speech

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        """prepare nontarget speech"""
        nontar_source, nontar_idx_list = self.prepare_nontar_speech(source)
        nontarget_list, nontar_padding_mask = [target_list[0][nontar_idx_list]], padding_mask[nontar_idx_list]
        """prepare mixed speech"""
        mixed_speech = self.prepare_mixed_speech(source, nontar_source)
        # mixed_speech = source + nontar_source
        """extract clean & mixed speech features"""
        clean_features = self.forward_features(source)  # (B, C, t) C=512
        mixed_features = self.forward_features(mixed_speech)  # (B, C, t) C=512
        """prepare tar & nontar target_list"""
        clean_features, target_list = self.forward_targets(clean_features, target_list)
        mixed_features, nontarget_list = self.forward_targets(mixed_features, nontarget_list)
        assert clean_features.size() == mixed_features.size()

        features_pen = clean_features.float().pow(2).mean()

        clean_features, mixed_features = clean_features.transpose(1, 2), mixed_features.transpose(1, 2)  # (B, t, C)
        clean_features, mixed_features = self.layer_norm(clean_features), self.layer_norm(mixed_features)
        # unmasked_features = features.clone()

        """prepare tar & nontar padding mask"""
        tar_padding_mask = self.forward_padding_mask(clean_features, padding_mask)
        nontar_padding_mask = self.forward_padding_mask(mixed_features, nontar_padding_mask)
        mixed_padding_mask = torch.logical_or(tar_padding_mask, nontar_padding_mask)

        if self.post_extract_proj is not None:
            clean_features = self.post_extract_proj(clean_features)
            mixed_features = self.post_extract_proj(mixed_features)  # (B, t, D) D=768

        clean_features, mixed_features = self.dropout_input(clean_features), self.dropout_input(mixed_features) #(B,t,D)
        # unmasked_features = self.dropout_features(unmasked_features)
        """apply feat-mask"""
        if mask:
            clean_x, clean_mask_indices = self.apply_mask(clean_features, tar_padding_mask, target_list)
            mixed_x, mixed_mask_indices = self.apply_mask(mixed_features, mixed_padding_mask, target_list)
        else:
            clean_x, mixed_x = clean_features, mixed_features
            clean_mask_indices, mixed_mask_indices = None, None

        clean_x, _ = self.encoder(clean_x,
                                  padding_mask=tar_padding_mask,
                                  layer=None if output_layer is None else output_layer - 1, )
        mixed_x, _ = self.encoder(mixed_x,
                                  padding_mask=mixed_padding_mask,
                                  layer=None if output_layer is None else output_layer - 1, )

        # if features_only:
        #     return {"x": x, "padding_mask": padding_mask, "features": features}

        def compute_pred(proj_x, label_embs, target, nontarget):
            # compute logits for the i-th label set
            if nontarget == "None":
                tar_y = torch.index_select(label_embs, 0, target.long())
                negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
                if self.target_glu:
                    tar_y = self.target_glu(tar_y)
                    negs = self.target_glu(negs)
                return self.compute_nce(proj_x, negs, tar_y, "None")
            else:
                tar_y = torch.index_select(label_embs, 0, target.long())
                nontar_y = torch.index_select(label_embs, 0, nontarget.long())
                negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
                if self.target_glu:
                    tar_y = self.target_glu(tar_y)
                    nontar_y = self.target_glu(nontar_y)
                    negs = self.target_glu(negs)
                return self.compute_nce(proj_x, negs, tar_y, nontar_y)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

        if not self.skip_masked:
            clean_masked_indices = torch.logical_and(~tar_padding_mask, clean_mask_indices)
            mixed_masked_indices = torch.logical_and(~mixed_padding_mask, mixed_mask_indices)
            clean_proj_x_m = self.final_proj(clean_x[clean_masked_indices])  # (masked_unit_nums, D) D=256
            mixed_proj_x_m = self.final_proj(mixed_x[mixed_masked_indices])  # (masked_unit_nums, D) D=256
            clean_m_label_list = [t[clean_masked_indices] for t in target_list]
            tar_m_label_list = [t[mixed_masked_indices] for t in target_list]
            nontar_m_label_list = [nt[mixed_masked_indices] for nt in nontarget_list]
            if self.untie_final_proj:
                clean_proj_x_m_list = clean_proj_x_m.chunk(len(target_list), dim=-1)  # tuple--->(proj_x_m, )
                mixed_proj_x_m_list = mixed_proj_x_m.chunk(len(target_list), dim=-1)  # tuple--->(proj_x_m, )
            else:
                clean_proj_x_m_list = [clean_proj_x_m for _ in range(len(target_list))]  # list--->[proj_x_m]
                mixed_proj_x_m_list = [mixed_proj_x_m for _ in range(len(target_list))]  # list--->[proj_x_m]
            clean_logit_m_list = [
                compute_pred(clean_proj_x_m, label_embs_list[i], tar[clean_masked_indices], "None")
                for i, (clean_proj_x_m, tar) in enumerate(zip(clean_proj_x_m_list, target_list))
            ]
            mixed_logit_m_list = [
                compute_pred(mixed_proj_x_m, label_embs_list[i], tar[mixed_masked_indices], ntar[mixed_masked_indices])
                for i, (mixed_proj_x_m, tar, ntar) in enumerate(zip(mixed_proj_x_m_list, target_list, nontarget_list))
            ]
        else:
            clean_logit_m_list = [None for _ in target_list]
            mixed_logit_m_list = [None for _ in target_list]
            clean_m_label_list = [None for _ in target_list]
            tar_m_label_list = [None for _ in target_list]
            nontar_m_label_list = [None for _ in nontarget_list]

        if not self.skip_nomask:
            clean_nomask_indices = torch.logical_and(~tar_padding_mask, ~clean_mask_indices)
            mixed_nomask_indices = torch.logical_and(~mixed_padding_mask, ~mixed_mask_indices)
            clean_proj_x_u = self.final_proj(clean_x[clean_nomask_indices])  # (unmasked_unit_nums, D) D=256
            mixed_proj_x_u = self.final_proj(mixed_x[mixed_nomask_indices])  # (unmasked_unit_nums, D) D=256
            clean_u_label_list = [t[clean_nomask_indices] for t in target_list]
            tar_u_label_list = [t[mixed_nomask_indices] for t in target_list]
            nontar_u_label_list = [nt[mixed_nomask_indices] for nt in nontarget_list]
            if self.untie_final_proj:
                clean_proj_x_u_list = clean_proj_x_u.chunk(len(target_list), dim=-1)  # tuple--->(proj_x_u, )
                mixed_proj_x_u_list = mixed_proj_x_u.chunk(len(target_list), dim=-1)  # tuple--->(proj_x_u, )
            else:
                clean_proj_x_u_list = [clean_proj_x_u for _ in range(len(target_list))]  # list--->[proj_x_u]
                mixed_proj_x_u_list = [mixed_proj_x_u for _ in range(len(target_list))]  # list--->[proj_x_u]
            clean_logit_u_list = [
                compute_pred(clean_proj_x_u, label_embs_list[i], tar[clean_nomask_indices], "None")
                for i, (clean_proj_x_u, tar) in enumerate(zip(clean_proj_x_u_list, target_list))
            ]
            mixed_logit_u_list = [
                compute_pred(mixed_proj_x_u, label_embs_list[i], tar[mixed_nomask_indices], ntar[mixed_nomask_indices])
                for i, (mixed_proj_x_u, tar, ntar) in enumerate(zip(mixed_proj_x_u_list, target_list, nontarget_list))
            ]
        else:
            clean_logit_u_list = [None for _ in target_list]
            mixed_logit_u_list = [None for _ in target_list]
            clean_u_label_list = [None for _ in target_list]
            tar_u_label_list = [None for _ in target_list]
            nontar_u_label_list = [None for _ in nontarget_list]

        result = {
            "clean_logit_m_list": clean_logit_m_list,
            "mixed_logit_m_list": mixed_logit_m_list,
            "clean_m_label_list": clean_m_label_list,
            "tar_m_label_list": tar_m_label_list,
            "nontar_m_label_list": nontar_m_label_list,
            "clean_logit_u_list": clean_logit_u_list,
            "mixed_logit_u_list": mixed_logit_u_list,
            "clean_u_label_list": clean_u_label_list,
            "tar_u_label_list": tar_u_label_list,
            "nontar_u_label_list": nontar_u_label_list,
            "clean_padding_mask": tar_padding_mask,
            "mixed_padding_mask": mixed_padding_mask,
            "features_pen": features_pen,
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            clean_logits_list = net_output["clean_logit_m_list"]
            mixed_logits_list = net_output["mixed_logit_m_list"]
        else:
            clean_logits_list = net_output["clean_logit_u_list"]
            mixed_logits_list = net_output["mixed_logit_u_list"]
        clean_logits_list = [x.float() for x in clean_logits_list if x is not None]
        mixed_logits_list = [y.float() for y in mixed_logits_list if y is not None]

        return clean_logits_list, mixed_logits_list

    def get_targets(self, net_output, is_masked=True):
        clean_logits_list, mixed_logits_list = self.get_logits(net_output, is_masked)
        if is_masked:
            clean_label_list = net_output["clean_m_label_list"]
            tar_label_list = net_output["tar_m_label_list"]
            nontar_label_list = net_output["nontar_m_label_list"]
        else:
            clean_label_list = net_output["clean_u_label_list"]
            tar_label_list = net_output["tar_u_label_list"]
            nontar_label_list = net_output["nontar_u_label_list"]
        num_classes_list = [logit.size(1)-1 for logit in clean_logits_list]
        c_targets_list = [F.one_hot(labels.long(), num_classes=num_classes_list[i]).to(torch.float)
                          for i, labels in enumerate(clean_label_list)]
        tar_1hot_list = [F.one_hot(labels.long(), num_classes=num_classes_list[i]).to(torch.float)
                         for i, labels in enumerate(tar_label_list)]
        nontar_1hot_list = [F.one_hot(labels.long(), num_classes=num_classes_list[i]).to(torch.float)
                            for i, labels in enumerate(nontar_label_list)]
        m_targets_list = [tar_label + nontar_label for (tar_label, nontar_label) in
                          zip(tar_1hot_list, nontar_1hot_list)]
        m_targets_list = [torch.where(target > 1, 1, target) for target in m_targets_list]

        return c_targets_list, m_targets_list, clean_label_list, tar_label_list, nontar_label_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
