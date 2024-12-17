import math
import torch
from torch import nn
from torch.nn import functional as F
from .utils import  QueryProposal, \
    CrossAttentionLayer, SelfAttentionLayer, FFNLayer, MLP
from ... import GraFormer

class QORT_Decoder(nn.Module):
    def __init__(
            self,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            num_aux_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,

    ):
        super().__init__()
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.iam_queries = num_queries
        self.num_queries = self.iam_queries
        self.num_aux_queries = num_aux_queries
        self.criterion = None

        meta_pos_size = int(round(math.sqrt(self.iam_queries)))
        self.meta_pos_embed = nn.Parameter(torch.empty(1, hidden_dim, meta_pos_size, meta_pos_size))
        if num_aux_queries > 0:
            self.empty_query_features = nn.Embedding(num_aux_queries, hidden_dim)
            self.empty_query_pos_embed = nn.Embedding(num_aux_queries, hidden_dim)


        self.query_proposal = QueryProposal(hidden_dim, num_queries, num_classes)

        self.transformer_query_cross_attention_layers = nn.ModuleList()
        self.transformer_query_self_attention_layers = nn.ModuleList()
        self.transformer_query_ffn_layers = nn.ModuleList()
        self.transformer_mask_cross_attention_layers = nn.ModuleList()
        self.transformer_mask_ffn_layers = nn.ModuleList()

        for idx in range(self.num_layers):
            self.transformer_query_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_query_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_query_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_mask_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm
                )
            )
            self.transformer_mask_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm
                )
            )

        self.decoder_query_norm_layers = nn.ModuleList()
        self.class_embed_layers = nn.ModuleList()
        self.keypoint_embed_left = MLP(hidden_dim, hidden_dim, 42, 3)
        self.keypoint_embed_right = MLP(hidden_dim, hidden_dim, 42, 3)
        self.obj_keypoint_embed = MLP(hidden_dim, hidden_dim, 63, 3)
        self.class_embed_layers = nn.Linear(hidden_dim, num_classes)

        nn.init.constant_(self.keypoint_embed_left.layers[-1].weight.data, 0)
        nn.init.constant_(self.keypoint_embed_left.layers[-1].bias.data, 0)
        nn.init.constant_(self.keypoint_embed_right.layers[-1].weight.data, 0)
        nn.init.constant_(self.keypoint_embed_right.layers[-1].bias.data, 0)

        nn.init.constant_(self.obj_keypoint_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.obj_keypoint_embed.layers[-1].bias.data, 0)
        for idx in range(self.num_layers + 1):
            self.decoder_query_norm_layers.append(nn.LayerNorm(hidden_dim))
        self.keypoint_embed_left = nn.ModuleList([self.keypoint_embed_left for _ in range(self.num_layers+1)])
        self.keypoint_embed_right = nn.ModuleList([self.keypoint_embed_right for _ in range(self.num_layers+1)])
        self.obj_keypoint_embed = nn.ModuleList([self.obj_keypoint_embed for _ in range(self.num_layers+1)])
        self.class_embed_layers = nn.ModuleList([self.class_embed_layers for _ in range(self.num_layers+1)])
        edges = GraFormer.create_edges(num_nodes=21)
        adj = GraFormer.adj_mx_from_edges(num_pts=21, edges=edges, sparse=False)
        self.graformer_left = GraFormer.GraFormer(adj=adj.cuda(), hid_dim=64, coords_dim=(2, 3),
                                        n_pts=21, num_layers=2, n_head=4, dropout=0.1)
        self.graformer_right = GraFormer.GraFormer(adj=adj.cuda(), hid_dim=64, coords_dim=(2, 3),
                                                  n_pts=21, num_layers=2, n_head=4, dropout=0.1)
        self.graformer_left = nn.ModuleList([self.graformer_left for _ in range(self.num_layers+1)])
        self.graformer_right = nn.ModuleList([self.graformer_right for _ in range(self.num_layers+1)])

    def forward(self, x, mask_features, targets=None):
        bs = x[0].shape[0]
        proposal_size = x[1].shape[-2:]
        pixel_feature_size = x[2].shape[-2:]

        pixel_pos_embeds = F.interpolate(self.meta_pos_embed, size=pixel_feature_size,
                                         mode="bilinear", align_corners=False)
        proposal_pos_embeds = F.interpolate(self.meta_pos_embed, size=proposal_size,
                                            mode="bilinear", align_corners=False)

        pixel_features = x[2].flatten(2).permute(2, 0, 1)
        pixel_pos_embeds = pixel_pos_embeds.flatten(2).permute(2, 0, 1)

        query_features, query_pos_embeds, query_locations, proposal_cls_logits, contact_map = self.query_proposal(
            x[1], x[1], proposal_pos_embeds, targets
        )
        query_features = query_features.permute(2, 0, 1)
        query_pos_embeds = query_pos_embeds.permute(2, 0, 1)
        if self.num_aux_queries > 0:
            aux_query_features = self.empty_query_features.weight.unsqueeze(1).repeat(1, bs, 1)
            aux_query_pos_embed = self.empty_query_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            query_features = torch.cat([query_features, aux_query_features], dim=0)
            query_pos_embeds = torch.cat([query_pos_embeds, aux_query_pos_embed], dim=0)

        outputs_class, key_preds = self.forward_prediction_heads(
            query_features, -1
        )
        predictions_class = [outputs_class]
        query_feature_memory = [query_features]
        pixel_feature_memory = [pixel_features]
        key, obj_key = key_preds
        outputs_keypoints = [key.sigmoid()]
        outputs_obj_keypoints = [obj_key.sigmoid()]

        for i in range(self.num_layers):
            query_features, pixel_features = self.forward_one_layer(
                query_features, pixel_features, query_pos_embeds, pixel_pos_embeds, i
            )
            if i < self.num_layers - 1:
                outputs_class, key_preds = self.forward_prediction_heads(
                    query_features, i
                )
            else:
                outputs_class, key_preds = self.forward_prediction_heads(
                    query_features, i
                )
            key, obj_key = key_preds

            outputs_keypoint = key.sigmoid()
            outputs_obj_keypoint = obj_key.sigmoid()
            outputs_keypoints.append(outputs_keypoint)
            outputs_obj_keypoints.append(outputs_obj_keypoint)
            predictions_class.append(outputs_class)
            query_feature_memory.append(query_features)
            pixel_feature_memory.append(pixel_features)

        predictions_class = predictions_class
        outputs_keypoints = torch.stack(outputs_keypoints)
        outputs_obj_keypoints = torch.stack(outputs_obj_keypoints)
        out = {
            'proposal_cls_logits': proposal_cls_logits,
            'pred_logits': predictions_class[-1],
            'pred_keypoints': outputs_keypoints[-1], 'pred_obj_keypoints': outputs_obj_keypoints[-1],
            'query_locations': query_locations if query_locations is not None else query_locations,
            "contact_map": contact_map,
        }
        if self.training:
            out['aux_outputs'] = self._set_aux_loss(
                predictions_class, query_locations, outputs_keypoints, outputs_obj_keypoints)
        return out
    def make_patches(self, feature_map, feat_pos_emb, keypoints):
        B, C, H, W = feature_map.size()
        _, _, num_keypoints, _ = keypoints.size()
        patch_size = 3
        half_patch = patch_size // 2

        patches = []
        pos_patches = []

        for b in range(B):
            kps = keypoints[b, 0]  # Shape: (21, 2)

            for kp in kps:
                x, y = kp
                x, y = int(x.item() * W), int(y.item() * H)

                x = max(half_patch, min(x, W - half_patch - 1))
                y = max(half_patch, min(y, H - half_patch - 1))

                patch = feature_map[b, :, y - half_patch:y + half_patch + 1,
                        x - half_patch:x + half_patch + 1]  # Shape: (C, 3, 3)
                if b == 0:
                    pos_patch = feat_pos_emb[b, :, y - half_patch:y + half_patch + 1,
                            x - half_patch:x + half_patch + 1]  # Shape: (C, 3, 3)
                    pos_patches.append(pos_patch)

                patches.append(patch)

        patches = torch.stack(patches)  # Shape: (B * 21, C, 3, 3)
        patches = patches.view(B, num_keypoints,C, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)

        pos_patches = torch.stack(pos_patches)  # Shape: (B * 21, C, 3, 3)
        pos_patches = pos_patches.view(1, num_keypoints, C, patch_size, patch_size)
        pos_patches = pos_patches.permute(0, 2, 1, 3, 4)
        return patches.flatten(3).flatten(2).permute(2, 0, 1), pos_patches.flatten(3).flatten(2).permute(2, 0, 1)
    def forward_one_layer(self, query_features, pixel_features, query_pos_embeds, pixel_pos_embeds, i):
        pixel_features = self.transformer_mask_cross_attention_layers[i](
            pixel_features, query_features, query_pos=pixel_pos_embeds, pos=query_pos_embeds
        )
        pixel_features = self.transformer_mask_ffn_layers[i](pixel_features)

        init_class = self.class_embed_layers[i](query_features.permute(1, 0, 2)[:, :self.num_queries])
        init_left_kp = self.keypoint_embed_left[i](query_features.permute(1, 0, 2)[:, :self.num_queries])[
                           torch.arange(init_class.size()[0]), torch.argmax(init_class[:, :, 9], dim=-1)][:, None,
                       :].sigmoid()
        init_right_kp = self.keypoint_embed_right[i](query_features.permute(1, 0, 2)[:, :self.num_queries])[
                            torch.arange(init_class.size()[0]), torch.argmax(init_class[:, :, 10], dim=-1)][:, None,
                        :].sigmoid()
        init_obj_kp = self.obj_keypoint_embed[i](query_features.permute(1, 0, 2)[:, :self.num_queries])[
                          torch.arange(init_class.size()[0]),
                          torch.max(torch.max(init_class[:, :, 1:9], dim=2).values, dim=1)[1]][:, None, :].sigmoid()

        patched_l, pos_patched_l = self.make_patches(
            pixel_features.permute(1, 2, 0).view(pixel_features.size()[1], pixel_features.size()[2], 68, 120),
            pixel_pos_embeds.permute(1, 2, 0).view(pixel_pos_embeds.size()[1],
                                                   pixel_pos_embeds.size()[2], 68, 120),
            init_left_kp.view(init_left_kp.size()[0], init_left_kp.size()[1], 21, 2))
        patched_r, pos_patched_r = self.make_patches(
            pixel_features.permute(1, 2, 0).view(pixel_features.size()[1], pixel_features.size()[2], 68, 120),
            pixel_pos_embeds.permute(1, 2, 0).view(pixel_pos_embeds.size()[1],
                                                   pixel_pos_embeds.size()[2], 68, 120),
            init_right_kp.view(init_right_kp.size()[0], init_right_kp.size()[1], 21, 2))

        patched_obj, pos_patched_obj = self.make_patches(
            pixel_features.permute(1, 2, 0).view(pixel_features.size()[1], pixel_features.size()[2], 68, 120),
            pixel_pos_embeds.permute(1, 2, 0).view(pixel_pos_embeds.size()[1],
                                                   pixel_pos_embeds.size()[2], 68, 120),
            init_obj_kp.view(init_obj_kp.size()[0], init_obj_kp.size()[1], 21, 3)[..., :2])

        new_pixel_features = torch.concat([pixel_features, patched_l, patched_r, patched_obj], dim=0)
        new_pixel_pos_embeds = torch.concat([pixel_pos_embeds, pos_patched_l, pos_patched_r, pos_patched_obj], dim=0)
        query_features = self.transformer_query_cross_attention_layers[i](
            query_features, new_pixel_features, memory_mask=None, query_pos=query_pos_embeds,
            pos=new_pixel_pos_embeds
        )
        query_features = self.transformer_query_self_attention_layers[i](
            query_features, query_pos=query_pos_embeds
        )
        query_features = self.transformer_query_ffn_layers[i](query_features)
        return query_features, pixel_features

    def forward_prediction_heads(self, query_features, idx_layer):
        decoder_query_features = self.decoder_query_norm_layers[idx_layer + 1](query_features[:self.num_queries])
        decoder_query_features = decoder_query_features.transpose(0, 1)
        if self.training or idx_layer + 1 == self.num_layers:
            outputs_class = self.class_embed_layers[idx_layer + 1](decoder_query_features)
        else:
            outputs_class = None
        obj_key = self.obj_keypoint_embed[idx_layer + 1](decoder_query_features)
        key_left = self.keypoint_embed_left[idx_layer + 1](decoder_query_features)
        key_right = self.keypoint_embed_right[idx_layer + 1](decoder_query_features)
        key_left_graformer = self.graformer_left[idx_layer + 1](key_left.view(key_left.shape[0] * key_left.shape[1], 21, 2)[..., :2])
        key_right_graformer = self.graformer_right[idx_layer + 1](key_right.view(key_right.shape[0] * key_right.shape[1], 21, 2)[..., :2])
        key_left = key_left.view(key_left.shape[0], key_left.shape[1], 21, 2)
        key_right = key_right.view(key_right.shape[0], key_right.shape[1], 21, 2)
        key_left = torch.concat((key_left, torch.ones((key_left.shape[0], key_left.shape[1], 21, 1)).cuda()), dim=3)
        key_right = torch.concat((key_right, torch.ones((key_right.shape[0], key_right.shape[1], 21, 1)).cuda()), dim=3)
        key = torch.stack([
                           key_left_graformer.view(key_left.shape[0], key_left.shape[1], 63),
                           key_right_graformer.view(key_right.shape[0], key_right.shape[1], 63),
                           key_left.view(key_left.shape[0], key_left.shape[1], 63), key_right.view(key_right.shape[0], key_right.shape[1], 63)])
        return outputs_class, (key, obj_key)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, output_query_locations, output_keyp, output_obj_keyp):
        return [
            {
                "query_locations": output_query_locations,
                "pred_logits": a,
                "pred_keypoints": d,
                "pred_obj_keypoints": e}
            for a, d, e in zip(outputs_class[:-1], output_keyp[:-1], output_obj_keyp[:-1])
        ]
