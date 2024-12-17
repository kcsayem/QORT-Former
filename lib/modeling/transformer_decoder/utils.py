# Modified from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class QueryProposal(nn.Module):

    def __init__(self, num_features, num_queries, num_classes):
        super().__init__()
        self.topk = num_queries
        self.num_classes = num_classes
        self.count = 0

        self.conv_proposal_left_cls_logits = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 2, kernel_size=1, stride=1, padding=0),
        )

        self.conv_proposal_right_cls_logits = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 2, kernel_size=1, stride=1, padding=0),
        )

        self.conv_proposal_obj_cls_logits = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_classes - 2 + 1, kernel_size=1, stride=1, padding=0),
        )

        self.contactmap_estimation = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((778, 1)),
            nn.Flatten(),
            nn.Linear(778 * 32, 778 * 2),
        )

        self.contactmap_conv = ContactMapConvolution(
            in_channels=2,
            out_channels=num_features,
            kernel_size=3,
            num_queries=self.topk,
            num_features=num_features,
        )

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(0, 1, h, device=x.device)
        x_loc = torch.linspace(0, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        locations = torch.stack([x_loc, y_loc], 0).unsqueeze(0)
        return locations

    def seek_local_maximum(self, x, epsilon=1e-6):
        x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
        # top, bottom, left, right, top-left, top-right, bottom-left, bottom-right
        maximum = (x >= x_pad[:, :, :-2, 1:-1]) & \
                  (x >= x_pad[:, :, 2:, 1:-1]) & \
                  (x >= x_pad[:, :, 1:-1, :-2]) & \
                  (x >= x_pad[:, :, 1:-1, 2:]) & \
                  (x >= x_pad[:, :, :-2, :-2]) & \
                  (x >= x_pad[:, :, :-2, 2:]) & \
                  (x >= x_pad[:, :, 2:, :-2]) & \
                  (x >= x_pad[:, :, 2:, 2:]) & \
                  (x >= epsilon)
        return maximum.to(x)

    def get_proposal_cls_probs(self, proposal_cls_probs, left_hand = None):
        if left_hand is None:
            proposal_cls_one_hot = F.one_hot(proposal_cls_probs[:, :-1, :, :].max(1)[1],
                                             num_classes=self.num_classes - 2 + 1).permute(0, 3, 1, 2)  # b, c, h, w
            proposal_cls_probs = proposal_cls_probs[:, :, :, :].mul(proposal_cls_one_hot)
        elif left_hand == True:
            proposal_cls_one_hot = F.one_hot(proposal_cls_probs[:, :-1, :, :].max(1)[1],
                                             num_classes=2).permute(0, 3, 1, 2)  # b, c, h, w
            proposal_cls_probs = proposal_cls_probs[:, :, :, :].mul(proposal_cls_one_hot)
        elif left_hand == False:
            proposal_cls_one_hot = F.one_hot(proposal_cls_probs[:, :-1, :, :].max(1)[1],
                                             num_classes=2).permute(0, 3, 1, 2)  # b, c, h, w
            proposal_cls_probs = proposal_cls_probs[:, :, :, :].mul(proposal_cls_one_hot)
        proposal_local_maximum_map = self.seek_local_maximum(proposal_cls_probs)  # b, c, h, w
        proposal_cls_probs = proposal_cls_probs + proposal_local_maximum_map
        return proposal_cls_probs
    def forward(self, x, y, pos_embeddings, targets=None):
        proposal_left_cls_logits = self.conv_proposal_left_cls_logits(x)
        proposal_right_cls_logits = self.conv_proposal_right_cls_logits(x)
        proposal_obj_cls_logits = self.conv_proposal_obj_cls_logits(x)
        proposal_left_cls_probs = proposal_left_cls_logits.softmax(dim=1)
        proposal_right_cls_probs = proposal_right_cls_logits.softmax(dim=1)
        proposal_obj_cls_probs = proposal_obj_cls_logits.softmax(dim=1)
        proposal_left_cls_probs = self.get_proposal_cls_probs(proposal_left_cls_probs, left_hand=True)
        proposal_right_cls_probs = self.get_proposal_cls_probs(proposal_right_cls_probs, left_hand=False)
        proposal_obj_cls_probs = self.get_proposal_cls_probs(proposal_obj_cls_probs, left_hand=None)

        # top-k indices
        topk_left_hand_indices = \
        torch.topk(proposal_left_cls_probs[:, :-1, :, :][:, None, :, :].flatten(2).max(1)[0],
                   40,
                   dim=1)[1]  # b, q
        topk_right_hand_indices = \
        torch.topk(proposal_right_cls_probs[:, :-1, :, :][:, None, :, :].flatten(2).max(1)[0],
                   40,
                   dim=1)[1]  # b, q
        topk_obj_indices = torch.topk(proposal_obj_cls_probs[:, :-1, :, :].flatten(2).max(1)[0],
                                      20,
                                      dim=1)[
            1]  # b, q

        topk_indices = torch.concat((topk_left_hand_indices, topk_right_hand_indices, topk_obj_indices), dim=1)
        topk_indices = topk_indices.unsqueeze(1)  # b, 1, q

        # topk queries
        topk_proposals = torch.gather(x.flatten(2), dim=2, index=topk_indices.repeat(1, x.shape[1], 1))  # b, c, q
        pos_embeddings = pos_embeddings.repeat(x.shape[0], 1, 1, 1).flatten(2)
        topk_pos_embeddings = torch.gather(
            pos_embeddings, dim=2, index=topk_indices.repeat(1, pos_embeddings.shape[1], 1)
        )  # b, c, q

        # contact map
        contact_map = self.contactmap_estimation(y).view(-1, 2, 778, 1)
        contact_map_features = self.contactmap_conv(contact_map)
        topk_proposals += contact_map_features

        if self.training:
            locations = self.compute_coordinates(x).repeat(x.shape[0], 1, 1, 1)
            topk_locations = torch.gather(
                locations.flatten(2), dim=2, index=topk_indices.repeat(1, locations.shape[1], 1)
            )
            topk_locations = topk_locations.transpose(-1, -2)  # b, q, 2
            #torch.save(topk_locations.detach().cpu(), f"queries/points_all_{self.count}.pt")
            self.count += 1
        else:
            topk_locations = None
        proposal_cls_logits = [proposal_left_cls_logits, proposal_right_cls_logits, proposal_obj_cls_logits]
        return topk_proposals, topk_pos_embeddings, topk_locations, proposal_cls_logits, contact_map


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class ContactMapConvolution(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, num_queries, num_features
    ):
        super(ContactMapConvolution, self).__init__()
        self.num_queries = num_queries
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.linear = nn.Linear(out_channels, num_queries * num_features)

    def forward(self, x):
        batch_size, _, num_vertices, _ = x.size()
        x = x.view(batch_size, -1, num_vertices)
        x = self.conv1d(x)
        x = torch.mean(x, dim=2)
        output = self.linear(x)
        output = output.view(batch_size, -1, self.num_queries)
        return output