import copy
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

import model.NetModules as NM
from transformers import HubertModel


def pad_list(xs, pad_value):
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = torch.zeros(
            batchs,
            max_len,
            dtype=xs[0].dtype,
            device=xs[0].device,
        )
    elif ndim == 2:
        pad_res = torch.zeros(
            batchs,
            max_len,
            xs[0].shape[1],
            dtype=xs[0].dtype,
            device=xs[0].device,
        )
    elif ndim == 3:
        pad_res = torch.zeros(
            batchs,
            max_len,
            xs[0].shape[1],
            xs[0].shape[2],
            dtype=xs[0].dtype,
            device=xs[0].device,
        )
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, :len(xs[i])] = xs[i]
    return pad_res


att_dict = {
    "MultiHeadCrossAtt": NM.MultiHeadCrossAtt,
    "MultiHeadAtt": NM.MultiHeadAtt,
}


class FrozenHubert(nn.Module):
    def __init__(self, model_name: str, device=None):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(model_name, use_safetensors=False)
        self.hubert.eval()
        for p in self.hubert.parameters():
            p.requires_grad = False

        self.device = device
        if device is not None:
            self.hubert.to(device)

    @torch.no_grad()
    def forward(self, wav_16k: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attn_mask = None
        if mask is not None:
            attn_mask = mask.long()
        out = self.hubert(input_values=wav_16k, attention_mask=attn_mask)
        return out.last_hidden_state


def _default_hubert_model_name() -> str:
    return "TencentGameMate/chinese-hubert-large"


def _parse_loss_weights(loss_weights=None, loss_weight=None) -> Tuple[float, float]:
    if isinstance(loss_weights, dict):
        return float(loss_weights.get("ctc", 0.0)), float(loss_weights.get("bce", 0.0))
    if isinstance(loss_weights, (list, tuple)) and len(loss_weights) >= 2:
        return float(loss_weights[0]), float(loss_weights[1])
    if isinstance(loss_weight, (list, tuple)) and len(loss_weight) >= 2:
        return float(loss_weight[0]), float(loss_weight[1])
    return 0.3, 0.6


class TransformerKWSPhone_hubert_wenet(nn.Module):
    def __init__(
        self,
        audio_net_config,
        kw_net_config,
        num_audio_block=8,
        num_kw_block=4,
        sok=1,
        eok=1,
        batch_padding_idx=-1,
        loss_weight=None,
        loss_weights=None,
        input_mode="auto",
        md_num_heads=1,
        hubert_model_name=None,
        hubert_input_dim=1024,
        **kwargs,
    ):
        super().__init__()
        self.sok = sok
        self.eok = eok
        self.batch_padding_idx = batch_padding_idx
        self.input_mode = input_mode
        self.md_num_heads = int(md_num_heads)
        if self.md_num_heads < 1:
            raise ValueError(f"md_num_heads must be >= 1, got {self.md_num_heads}")

        au_input_trans_config = audio_net_config["input_trans"]
        au_transformer_config = audio_net_config["transformer_config"]
        au_self_att = att_dict[au_transformer_config["self_att"]]
        au_self_att_config = au_transformer_config["self_att_config"]
        au_cross_att = att_dict[au_transformer_config["cross_att"]]
        au_cross_att_config = au_transformer_config["corss_att_config"]
        au_feed_forward_config = au_transformer_config["feed_forward_config"]
        au_hidden_dim = au_transformer_config["size"]

        kw_input_trans_config = kw_net_config["input_trans"]
        num_phn_token = kw_net_config["num_phn_token"]
        kw_transformer_config = kw_net_config["transformer_config"]
        kw_self_att = att_dict[kw_transformer_config["self_att"]]
        kw_self_att_config = kw_transformer_config["self_att_config"]
        kw_feed_forward_config = kw_transformer_config["feed_forward_config"]
        kw_hidden_dim = kw_transformer_config["size"]

        self.hubert = None
        if self.input_mode == "waveform":
            hubert_model_name = hubert_model_name or _default_hubert_model_name()
            self.hubert = FrozenHubert(hubert_model_name)
        self.hubert_trans = nn.Linear(hubert_input_dim, au_hidden_dim)

        self.au_pos_emb = NM.PositionalEncoding(au_hidden_dim)
        self.au_transformer = nn.ModuleList(
            [
                NM.TransformerLayer(
                    size=au_hidden_dim,
                    self_att=au_self_att(**au_self_att_config),
                    cross_att=au_cross_att(**au_cross_att_config),
                    feed_forward=NM.FNNBlock(**au_feed_forward_config),
                )
                for _ in range(num_audio_block)
            ]
        )

        self.md_transformer = nn.ModuleList(
            [
                NM.TransformerLayer(
                    size=au_hidden_dim,
                    self_att=au_self_att(**au_self_att_config),
                    cross_att=au_cross_att(**au_cross_att_config),
                    feed_forward=NM.FNNBlock(**au_feed_forward_config),
                )
                for _ in range(4)
            ]
        )

        self.phn_emb = NM.WordEmbedding(
            num_tokens=num_phn_token,
            dim=kw_transformer_config["size"],
        )
        self.kw_pos_emb = NM.PositionalEncoding(kw_hidden_dim)
        self.kw_trans = NM.FNNBlock(**kw_input_trans_config)
        self.kw_transformer = nn.ModuleList(
            [
                NM.TransformerLayer(
                    size=kw_hidden_dim,
                    self_att=kw_self_att(**kw_self_att_config),
                    feed_forward=NM.FNNBlock(**kw_feed_forward_config),
                )
                for _ in range(num_kw_block)
            ]
        )
        if kw_hidden_dim != au_hidden_dim:
            self.kw_au_link = nn.Linear(kw_hidden_dim, au_hidden_dim)
        else:
            self.kw_au_link = nn.Identity()

        if self.md_num_heads == 1:
            self.det_net = nn.Sequential(
                NM.FNNBlock(**au_feed_forward_config),
                nn.Linear(au_hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.det_nets = None
        else:
            self.det_net = None
            self.det_nets = nn.ModuleList(
                [
                    nn.Sequential(
                        NM.FNNBlock(**au_feed_forward_config),
                        nn.Linear(au_hidden_dim, 1),
                        nn.Sigmoid(),
                    )
                    for _ in range(self.md_num_heads)
                ]
            )

        phn_ctc_conf = {
            "num_tokens": num_phn_token,
            "front_output_size": au_hidden_dim,
        }
        self.lambda_ctc, self.lambda_bce = _parse_loss_weights(loss_weights, loss_weight)
        self.det_crit = nn.BCELoss(reduction="none")
        self.phn_asr_crit = NM.CTC(**phn_ctc_conf)

    def forward_transformer(
        self,
        transformer_module,
        input,
        mask=None,
        cross_embedding=None,
        analyse=False,
        print_mask=False,
    ):
        if analyse:
            batch_size = input.size(0)
            att_scores = {i: [] for i in range(batch_size)}
            embeddings = {i: [] for i in range(batch_size)}
        for _, tf_layer in enumerate(transformer_module):
            input, att_score = tf_layer(
                input,
                mask,
                cross_input=cross_embedding,
                print_mask=print_mask,
            )
            if not analyse:
                continue
            for batch_idx, att in enumerate(att_score):
                att_scores[batch_idx].append(copy.deepcopy(att))
                embeddings[batch_idx].append(copy.deepcopy(input))
        if analyse:
            return input, (att_scores, embeddings)
        return input

    def forward_audio_transformer(self, input, mask=None, cross_embedding=None):
        for tf_layer in self.au_transformer:
            input, _ = tf_layer(input, mask, cross_input=cross_embedding)
        return input

    def forward_md_transformer(self, input, mask=None, cross_embedding=None):
        for tf_layer in self.md_transformer:
            input, _ = tf_layer(input, mask, cross_input=cross_embedding)
        return input

    def _prepare_audio_inputs(self, sph_input, sph_len):
        if self.input_mode == "waveform":
            sph_mask = ~NM.make_mask(sph_len)
            sph_emb = self.hubert(sph_input, mask=sph_mask)
            sph_len = self.hubert.hubert._get_feat_extract_output_lengths(sph_len)
            sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
            sph_emb = self.hubert_trans(sph_emb)
            return sph_emb, sph_len, sph_mask

        if self.input_mode == "embedding":
            sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
            sph_emb = self.hubert_trans(sph_input)
            return sph_emb, sph_len, sph_mask

        raise ValueError(f"Unsupported input_mode: {self.input_mode}")

    def _prepare_keyword_inputs(self, kw_label, kw_len):
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
        kw_emb = self.phn_emb(kw_label.to(torch.long))
        kw_emb = self.kw_trans(kw_emb)
        return kw_emb, kw_mask

    def _forward_backbone(self, sph_input, sph_len, kw_label, kw_len):
        sph_emb, sph_len, sph_mask = self._prepare_audio_inputs(sph_input, sph_len)
        kw_emb, kw_mask = self._prepare_keyword_inputs(kw_label, kw_len)

        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)

        sph_emb = self.au_pos_emb(sph_emb)
        kw_emb = self.kw_pos_emb(kw_emb)

        kw_emb = self.forward_transformer(
            self.kw_transformer,
            kw_emb,
            mask=kw_mask,
        )
        sph_emb = self.forward_audio_transformer(
            sph_emb,
            mask=sph_mask,
            cross_embedding=(kw_emb, kw_emb, cross_mask),
        )
        kw_emb = self.forward_md_transformer(
            kw_emb,
            mask=kw_mask,
            cross_embedding=(sph_emb, sph_emb, cross_mask.transpose(-2, -1)),
        )
        return sph_emb, sph_len, kw_emb

    def _get_detection_outputs(self, kw_emb) -> List[torch.Tensor]:
        if self.det_nets is not None:
            return [net(kw_emb).squeeze(-1) for net in self.det_nets]
        return [self.det_net(kw_emb).squeeze(-1)]

    def _normalize_md_labels(self, data: Sequence[torch.Tensor]):
        if len(data) == 9:
            sph_input, sph_len, phn_label, phn_len, kw_label, kw_len, md_label, md_label_len, target = data
            md_labels = [md_label for _ in range(self.md_num_heads)]
        elif len(data) >= 11:
            sph_input, sph_len, phn_label, phn_len, kw_label, kw_len, *rest = data
            md_label_len = rest[-2]
            target = rest[-1]
            raw_md_labels = list(rest[:-2])
            if len(raw_md_labels) == 0:
                raise ValueError("Missing md labels in input data")
            if len(raw_md_labels) < self.md_num_heads:
                raw_md_labels.extend([raw_md_labels[-1]] * (self.md_num_heads - len(raw_md_labels)))
            md_labels = raw_md_labels[: self.md_num_heads]
        else:
            raise ValueError(
                f"Unexpected number of input tensors: {len(data)}. "
                "Expected 9 or at least 11 tensors."
            )
        return (
            sph_input,
            sph_len,
            phn_label,
            phn_len,
            kw_label,
            kw_len,
            md_labels,
            md_label_len,
            target,
        )

    def forward(self, input_data):
        data = list(input_data)
        (
            sph_input,
            sph_len,
            phn_label,
            phn_len,
            kw_label,
            kw_len,
            md_labels,
            md_label_len,
            _target,
        ) = self._normalize_md_labels(data)

        md_mask = ~NM.make_mask(md_label_len)
        sph_emb, sph_len, kw_emb = self._forward_backbone(sph_input, sph_len, kw_label, kw_len)

        phn_ctc_loss, _ = self.phn_asr_crit(
            sph_emb,
            phn_label,
            sph_len,
            phn_len,
            return_hyp=True,
        )

        det_results = self._get_detection_outputs(kw_emb)
        det_losses = []
        for det_result, md_label in zip(det_results, md_labels):
            loss_one = self.det_crit(det_result, md_label.to(torch.float32))
            loss_one = loss_one.masked_fill(~md_mask, 0)
            loss_one = loss_one.sum(dim=-1).mean()
            det_losses.append(loss_one)

        det_loss = torch.stack(det_losses).mean()
        total_loss = (self.lambda_ctc * phn_ctc_loss) + (self.lambda_bce * det_loss)

        detail_loss = {
            "phn_ctc_loss": phn_ctc_loss.detach().clone(),
            "det_loss": det_loss.detach().clone(),
        }
        if len(det_losses) > 1:
            for idx, one_loss in enumerate(det_losses, start=1):
                detail_loss[f"det_loss{idx}"] = one_loss.detach().clone()
        return total_loss, detail_loss

    @torch.no_grad()
    def evaluate(self, input_data):
        sph_input, sph_len, kw_label, kw_len, _md_label = input_data
        sph_emb, _, kw_emb = self._forward_backbone(sph_input, sph_len, kw_label, kw_len)

        hyp_phn = self.phn_asr_crit.get_hyp(sph_emb)
        hyp_phn = hyp_phn.log_softmax(dim=-1)
        _, hyp = hyp_phn.sort(descending=True)
        hyp = hyp[:, :, 0]

        det_results = self._get_detection_outputs(kw_emb)
        if len(det_results) == 1:
            return det_results[0], hyp
        return (*det_results, hyp)

    @torch.no_grad()
    def evaluate_mean(self, input_data):
        eval_outputs = self.evaluate(input_data)
        if self.md_num_heads == 1:
            return eval_outputs
        det_results = list(eval_outputs[:-1])
        hyp = eval_outputs[-1]
        det_mean = torch.stack(det_results, dim=0).mean(dim=0)
        return det_mean, hyp

    @torch.no_grad()
    def evaluate_multi(self, input_data):
        return self.evaluate(input_data)
