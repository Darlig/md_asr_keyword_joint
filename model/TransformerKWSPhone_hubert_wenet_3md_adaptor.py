import math
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import model.NetModules as NM
from transformers import HubertModel
from typing import Optional



def subsequent_mask( size, device):
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask

def pad_list(xs, pad_value):
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = torch.zeros(batchs,
                              max_len,
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 2:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 3:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              xs[0].shape[2],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, :len(xs[i])] = xs[i]
    return pad_res

def add_sos_eos(ys_pad, sos, eos, ignore_id):
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)

att_dict = {
    'MultiHeadCrossAtt': NM.MultiHeadCrossAtt, 
    'MultiHeadAtt': NM.MultiHeadAtt
}


class FrozenHubert(nn.Module):
    """
    输入: wav (B, L) float32/float16, 16kHz
    输出: feats (B, T, D), attention_mask (B, T)  (T 是 HuBERT 帧数)
    """
    def __init__(self, model_name="facebook/hubert-base-ls960", device=None):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(model_name, use_safetensors=False)
        #self.hubert = HubertModel.from_pretrained(model_name, use_safetensors=True)
        self.hubert.eval()
        for p in self.hubert.parameters():
            p.requires_grad = False

        self.device = device
        if device is not None:
            self.hubert.to(device)

    @torch.no_grad()
    def forward(self, wav_16k: torch.Tensor, mask: Optional[torch.Tensor]=None):
        """
        wav_16k: (B, L) 16kHz
        mask: (B, L) 0/1, 用于构造 attention mask（可选但建议给）
        # """
        B, L = wav_16k.shape
        attn_mask = None
        if mask is not None:
            attn_mask = mask.long()
        out = self.hubert(input_values=wav_16k, attention_mask=attn_mask)
        feats = out.last_hidden_state  # (B, T, D)
        return feats

class TransformerKWSPhone_hubert_wenet_3md_adaptor(nn.Module):
    def __init__(
        self,
        audio_net_config,
        kw_net_config,
        num_audio_block=8,
        num_kw_block=4,
        sok=1,
        eok=1,
        batch_padding_idx=-1,
        loss_weight=[0.3,0.6,0.1],
        **kwargs,
    ):
        super(TransformerKWSPhone_hubert_wenet_3md_adaptor, self).__init__()
        self.sok = sok
        self.eok = eok
        self.batch_padding_idx = batch_padding_idx
        # detach_config 
        # TODO: looks stupid ....   >_<
        # audio config
        au_input_trans_config = audio_net_config['input_trans']
        au_transformer_config = audio_net_config['transformer_config']
        au_self_att = att_dict[au_transformer_config['self_att']]
        au_self_att_cofing = au_transformer_config['self_att_config']
        au_cross_att = att_dict[au_transformer_config['cross_att']]
        au_cross_att_config = au_transformer_config['corss_att_config']
        au_feed_forward_config = au_transformer_config['feed_forward_config']
        au_hidden_dim = au_transformer_config['size']

        # vocab config
        kw_input_trans_config = kw_net_config['input_trans']
        num_phn_token = kw_net_config['num_phn_token']
        kw_transformer_config = kw_net_config['transformer_config']
        kw_self_att = att_dict[kw_transformer_config['self_att']]
        kw_self_att_cofing = kw_transformer_config['self_att_config']
        kw_feed_forward_config = kw_transformer_config['feed_forward_config']
        kw_hidden_dim = kw_transformer_config['size']

        # hubert frontend
        self.hubert = FrozenHubert('/work107/weiyang/project/maolidan_thesis/experiment/md_asr_keyword_joint_hubert_aishell2/pretrained_models/chinese-hubert-large')
        #self.hubert = FrozenHubert()
        self.hubert_trans = nn.Linear(1024, au_hidden_dim)  # hubert wenet large output dim is 1024

        ## audio net
        #self.au_conv = nn.Sequential(
        #    torch.nn.Conv2d(1, au_hidden_dim, 3, 2),
        #    torch.nn.ReLU(),
        #    torch.nn.Conv2d(au_hidden_dim, au_hidden_dim, 3, 2),
        #    torch.nn.ReLU(),
        #)
        #self.au_conv_trans = nn.Linear(au_hidden_dim * (((40 - 1) // 2 - 1) // 2), au_hidden_dim)

        #self.au_trans = NM.FNNBlock(
        #    **au_input_trans_config
        #)
        self.au_pos_emb = NM.PositionalEncoding(au_hidden_dim)

        self.au_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=au_hidden_dim,
                self_att=au_self_att(**au_self_att_cofing),
                cross_att=au_cross_att(**au_cross_att_config),
                feed_forward=NM.FNNBlock(**au_feed_forward_config),
            ) for _ in range(num_audio_block )
        ])

        self.md_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=au_hidden_dim,
                self_att=au_self_att(**au_self_att_cofing),
                cross_att=au_cross_att(**au_cross_att_config),
                feed_forward=NM.FNNBlock(**au_feed_forward_config),
            ) for _ in range(4)
        ])
        #self.skip_trans = nn.ModuleList([nn.Linear(au_hidden_dim, au_hidden_dim) for _ in range(num_audio_block // 2 - 1)])

        # kw net
        self.phn_emb = NM.WordEmbedding(
            num_tokens=num_phn_token, dim=kw_transformer_config['size']
        )
        self.kw_pos_emb = NM.PositionalEncoding(kw_hidden_dim)
        self.kw_trans = NM.FNNBlock(**kw_input_trans_config)
        self.kw_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=kw_hidden_dim,
                self_att=kw_self_att(**kw_self_att_cofing),
                feed_forward=NM.FNNBlock(**kw_feed_forward_config)
            ) for _ in range(num_kw_block)
        ])
        if kw_hidden_dim != au_hidden_dim:
            self.kw_au_link = nn.Linear(kw_hidden_dim, au_hidden_dim)
        else:
            self.kw_au_link = nn.Identity()

        # === MD heads (3-annotator) ===
        # Keep a single shared MD transformer, but use 3 independent
        # (FNN + Linear + Sigmoid) heads to model different annotators.
        self.det_nets = nn.ModuleList([
            nn.Sequential(
                NM.FNNBlock(**au_feed_forward_config),
                nn.Linear(au_hidden_dim, 1),
                nn.Sigmoid(),
            ) for _ in range(3)
        ])

        # decoder net
        
        phn_ctc_conf = {
            'num_tokens': num_phn_token,
            'front_output_size': au_hidden_dim 
        }
        self.l1, self.l2, self.l3 = loss_weight
        self.det_crit = nn.BCELoss(reduction='none')
        self.phn_asr_crit = NM.CTC(**phn_ctc_conf)

    def forward_transformer(
        self,
        transformer_module,
        input,
        mask=None,
        cross_embedding=None,
        analyse=False,
        print_mask=False
    ):
        if analyse:
            b = input.size(0)
            att_scores = {i:[] for i in range(b)}
            embeddings = {i:[] for i in range(b)}
        for i, tf_layer in enumerate(transformer_module):
            input, att_score = tf_layer(input, mask, cross_input=cross_embedding, print_mask=print_mask)
            if not analyse:
                continue
            for i, att in enumerate(att_score):
                att_scores[i].append(copy.deepcopy(att))
                embeddings[i].append(copy.deepcopy(input))
        if analyse:
            return input, (att_scores, embeddings)
        else:
            return input
    

    def forward_audio_transformer(self, input, mask=None, cross_embedding=None):
        for i, tf_layer in enumerate(self.au_transformer):
            input, att_score = tf_layer(input, mask, cross_input=cross_embedding)
        return input

    def forward_md_transformer(self, input, mask=None, cross_embedding=None):
        for i, tf_layer in enumerate(self.md_transformer):
            input, att_score = tf_layer(input, mask, cross_input=cross_embedding)
        return input


    def forward(self, input_data):
        # mixspeech,mixspeech_len,phn_label,phn_label_len,keyword,keyword_len,
        # md_label1,md_label2,md_label3,md_label_len,target
        # Backward compatible: if only one md_label is provided, reuse it for all 3 heads.
        data = list(input_data)
        if len(data) == 9:
            sph_input, sph_len, phn_label, phn_len, kw_label, kw_len, md_label, md_label_len, target = data
            md_label1 = md_label2 = md_label3 = md_label
        elif len(data) == 11:
            sph_input, sph_len, phn_label, phn_len, kw_label, kw_len, md_label1, md_label2, md_label3, md_label_len, target = data
        else:
            raise ValueError(f"Unexpected number of input tensors: {len(data)}. \
Expected 9 (single md_label) or 11 (3 md_labels).")
        sph_mask = ~NM.make_mask(sph_len)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
        md_mask = ~NM.make_mask(md_label_len)
        
        # hubert frontend
        sph_emb = self.hubert(sph_input, mask=sph_mask)
        org_sph_len = sph_len
        sph_len = self.hubert.hubert._get_feat_extract_output_lengths(sph_len)
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)
        sph_emb = self.hubert_trans(sph_emb)


        # phone embedding
        kw_emb = self.phn_emb(kw_label.to(torch.long))
        kw_emb = self.kw_trans(kw_emb)

        # add position embedding
        sph_emb = self.au_pos_emb(sph_emb)
        kw_emb = self.kw_pos_emb(kw_emb)

        kw_emb = self.forward_transformer(
            self.kw_transformer,
            kw_emb,
            mask=kw_mask,
        )
        sph_emb = self.forward_audio_transformer(sph_emb, mask=sph_mask, cross_embedding=(kw_emb, kw_emb, cross_mask))
        kw_emb = self.forward_md_transformer(kw_emb, mask=kw_mask, cross_embedding=(sph_emb, sph_emb, cross_mask.transpose(-2,-1)))
        # asr loss
        phn_ctc_loss, phn_asr_hyp = self.phn_asr_crit(
            sph_emb, phn_label, sph_len, phn_len, return_hyp=True
        )

        # pos loss
        # cross_context, cross_att = self.location_cross_att(
            # kw_emb, sph_emb, sph_emb, mask=cross_mask.transpose(-2,-1),
        # )
        # detection loss (3 heads)
        # input: md_label1/2/3 are per-annotator md labels, padded by 0
        det_results = [net(kw_emb).squeeze(-1) for net in self.det_nets]

        det_losses = []
        for det_result_k, md_label_k in zip(det_results, [md_label1, md_label2, md_label3]):
            loss_k = self.det_crit(det_result_k, md_label_k.to(torch.float32))
            loss_k = loss_k.masked_fill(~md_mask, 0)
            loss_k = loss_k.sum(dim=-1).mean()
            det_losses.append(loss_k)

        det_loss = sum(det_losses) / len(det_losses)

        # decoder output
        total_loss = (self.l1 * phn_ctc_loss) + (self.l2 * det_loss)
        detail_loss = {
            'phn_ctc_loss': phn_ctc_loss.clone().detach(),
            'det_loss': det_loss.clone().detach(),
            'det_loss1': det_losses[0].clone().detach(),
            'det_loss2': det_losses[1].clone().detach(),
            'det_loss3': det_losses[2].clone().detach(),
        }
        return total_loss, detail_loss
    

    @torch.no_grad()
    def evaluate(self, input_data):
        sph_input, sph_len,  kw_label, kw_len, md_label = input_data # target here used to debug lol ^v^
        sph_mask = ~NM.make_mask(sph_len)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)

        # hubert frontend
        sph_emb = self.hubert(sph_input, mask=sph_mask)
        sph_len = self.hubert.hubert._get_feat_extract_output_lengths(sph_len)
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)
        sph_emb = self.hubert_trans(sph_emb)
        
        # phone embedding
        kw_emb = self.phn_emb(kw_label.to(torch.long))
        kw_emb = self.kw_trans(kw_emb)

        # add position embedding
        sph_emb = self.au_pos_emb(sph_emb)
        kw_emb = self.kw_pos_emb(kw_emb)

        kw_emb = self.forward_transformer(
            self.kw_transformer,
            kw_emb,
            mask=kw_mask,
        )
        #sph_emb = self.forward_audio_transformer(sph_emb, mask=sph_mask)# cross_embedding=(kw_emb, kw_emb, cross_mask))
        sph_emb = self.forward_audio_transformer(sph_emb, mask=sph_mask, cross_embedding=(kw_emb, kw_emb, cross_mask))
        #sph_emb =  torch.ones_like(sph_emb)
        kw_emb = self.forward_md_transformer(kw_emb, mask=kw_mask, cross_embedding=(sph_emb, sph_emb, cross_mask.transpose(-2,-1)))
        # asr loss
        hyp_phn = self.phn_asr_crit.get_hyp(sph_emb)
        hyp_phn = hyp_phn.log_softmax(dim=-1)
        scores, hyp = hyp_phn.sort(descending=True)
        hyp = hyp[:,:,0]
        # detection output (3 heads)
        det_results = [net(kw_emb).squeeze(-1) for net in self.det_nets]
        det_mean = sum(det_results) / len(det_results)

        # keep backward-compatible return: (mean_posterior, hyp)
        return det_mean, hyp


    @torch.no_grad()
    def evaluate_multi(self, input_data):
        """
        Return per-head posteriors for 3 annotators.
        This is the recommended API for multi-annotator evaluation/inference.

        Returns:
            det1, det2, det3, hyp
        """
        sph_input, sph_len, kw_label, kw_len, _md_label = input_data
        sph_mask = ~NM.make_mask(sph_len)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)

        sph_emb = self.hubert(sph_input, mask=sph_mask)
        sph_len = self.hubert.hubert._get_feat_extract_output_lengths(sph_len)
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)
        sph_emb = self.hubert_trans(sph_emb)

        kw_emb = self.phn_emb(kw_label.to(torch.long))
        kw_emb = self.kw_trans(kw_emb)

        sph_emb = self.au_pos_emb(sph_emb)
        kw_emb = self.kw_pos_emb(kw_emb)

        kw_emb = self.forward_transformer(
            self.kw_transformer,
            kw_emb,
            mask=kw_mask,
        )
        sph_emb = self.forward_audio_transformer(sph_emb, mask=sph_mask, cross_embedding=(kw_emb, kw_emb, cross_mask))
        kw_emb = self.forward_md_transformer(kw_emb, mask=kw_mask, cross_embedding=(sph_emb, sph_emb, cross_mask.transpose(-2,-1)))

        hyp_phn = self.phn_asr_crit.get_hyp(sph_emb)
        hyp_phn = hyp_phn.log_softmax(dim=-1)
        scores, hyp = hyp_phn.sort(descending=True)
        hyp = hyp[:,:,0]

        det_results = [net(kw_emb).squeeze(-1) for net in self.det_nets]
        return det_results[0], det_results[1], det_results[2], hyp
