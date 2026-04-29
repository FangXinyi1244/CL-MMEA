#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

try:
    from layers import *
except:
    from src.layers import *


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag):
        super(GAT, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                MultiHeadGraphAttention(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False))

    def forward(self, x, adj):
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, adj)
            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        return x


""" vanilla GCN """


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # change to leaky relu
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = F.relu(x)
        return x


""" loss """


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    a = norm.expand_as(X) + 1e-8
    X = torch.div(X, a)
    return X


# ========== 添加 FeatureMasking 类 ==========
class FeatureMasking(nn.Module):
    """
    对输入特征进行随机掩码（Feature Masking）
    每个 epoch 随机将特征向量的 mask_ratio 比例维度置零
    注意：此类仅在 training=True 且 mask_ratio > 0 时生效
    """
    
    def __init__(self, mask_ratio=0.15, mask_method="random"):
        super(FeatureMasking, self).__init__()
        self.mask_ratio = mask_ratio
        self.mask_method = mask_method
        
    def forward(self, x, training=True):
        """
        安全的前向传播：
        - 非训练模式：返回原始特征
        - mask_ratio=0：返回原始特征
        - 否则：返回掩码后的特征
        """
        # 关键：如果不训练或掩码比例为 0，直接返回输入
        if not training or self.mask_ratio == 0.0:
            return x
        
        batch_size, feature_dim = x.shape
        
        if self.mask_method == "random":
            # 方法1: 每个样本独立随机掩码
            mask = torch.rand_like(x) < self.mask_ratio
            x_masked = x.clone()
            x_masked[mask] = 0.0
            return x_masked
            
        elif self.mask_method == "structured":
            # 方法2: 结构化掩码（同一 batch 掩码相同维度）
            mask_indices = torch.randperm(feature_dim)[:int(feature_dim * self.mask_ratio)]
            mask = torch.zeros_like(x)
            mask[:, mask_indices] = 1.0
            return x * (1 - mask)
            
        else:
            # 默认返回原始特征
            return x


class MultiModalFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(torch.ones((self.modal_num, 1)),
                                   requires_grad=self.requires_grad)

    def forward(self, embs):
        assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [weight_norm[idx] * F.normalize(embs[idx]) for idx in range(self.modal_num) if embs[idx] is not None]
        joint_emb = torch.cat(embs, dim=1)
        # joint_emb = torch.sum(torch.stack(embs, dim=1), dim=1)
        return joint_emb


class MultiModalEncoder(nn.Module):
    """
    entity embedding: (ent_num, input_dim)
    gcn layer: n_units

    """

    def __init__(self, args,
                 ent_num,
                 img_feature_dim,
                 char_feature_dim=None,
                 use_project_head=False):
        super(MultiModalEncoder, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        name_dim = self.args.name_dim
        char_dim = self.args.char_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        #########################
        ######## Entity Embedding
        #########################
        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True

        #########################
        ######## Modal Encoder
        #########################

        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(1000, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)
        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)

        # structure encoder
        if self.args.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.args.dropout)
        elif self.args.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=args.dropout,
                                         attn_dropout=args.attn_dropout,
                                         instance_normalization=self.args.instance_normalization, diag=True)

        #########################
        ##### Projection Head
        #########################
        if self.use_project_head:
            self.img_pro = ProjectionHead(img_dim, img_dim, img_dim, dropout)
            self.att_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.rel_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.gph_pro = ProjectionHead(self.n_units[2], self.n_units[2], self.n_units[2], dropout)

        #########################
        ######## Fusion Encoder
        #########################

        self.fusion = MultiModalFusion(modal_num=self.args.inner_view_num,
                                       with_weight=self.args.with_weight)

         # ========== 新增：特征掩码模块（可选） ==========
        self.mask_ratio = getattr(args, 'mask_ratio', 0.0)  # 默认 0（关闭）
        self.mask_method = getattr(args, 'mask_method', 'random')
        # 缓存掩码特征（用于计算掩码对比损失）
        self.masked_features_cache = None
        
        if self.mask_ratio > 0.0:
            self.feature_masking = FeatureMasking(
                mask_ratio=self.mask_ratio,
                mask_method=self.mask_method
            )
        else:
            self.feature_masking = None

    def forward(self,
                input_idx,
                adj,
                img_features=None,
                rel_features=None,
                att_features=None,
                name_features=None,
                char_features=None):

        if self.args.w_gcn:
            gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
        else:
            gph_emb = None
        if self.args.w_img:
            img_emb = self.img_fc(img_features)
        else:
            img_emb = None
        if self.args.w_rel:
            rel_emb = self.rel_fc(rel_features)
        else:
            rel_emb = None
        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
        else:
            att_emb = None
        if self.args.w_name:
            name_emb = self.name_fc(name_features)
        else:
            name_emb = None
        if self.args.w_char:
            char_emb = self.char_fc(char_features)
        else:
            char_emb = None

        if self.use_project_head:
            gph_emb = self.gph_pro(gph_emb)
            img_emb = self.img_pro(img_emb)
            rel_emb = self.rel_pro(rel_emb)
            att_emb = self.att_pro(att_emb)
            pass

        joint_emb = self.fusion([img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb])

        # ========== 新增：生成并缓存掩码特征 ==========
        joint_emb_masked = None  # 初始化
        if self.feature_masking is not None and self.training:
            # 对每种模态特征进行掩码
            masked_img = self.feature_masking(img_emb) if img_emb is not None else None
            masked_rel = self.feature_masking(rel_emb) if rel_emb is not None else None
            masked_att = self.feature_masking(att_emb) if att_emb is not None else None
            masked_gph = self.feature_masking(gph_emb) if gph_emb is not None else None
            masked_name = self.feature_masking(name_emb) if name_emb is not None else None
            masked_char = self.feature_masking(char_emb) if char_emb is not None else None
            
            # 缓存掩码特征（用于计算掩码对比损失）
            self.masked_features_cache = {
                'img': masked_img,
                'rel': masked_rel,
                'att': masked_att,
                'gph': masked_gph,
                'name': masked_name,
                'char': masked_char,
            }
            
            # 计算掩码后的 joint_emb
            joint_emb_masked = self.fusion([masked_img, masked_att, masked_rel, masked_gph, masked_name, masked_char])
        else:
            self.masked_features_cache = None

        return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, joint_emb_masked

    def get_masked_features(self):
        """获取掩码后的特征（用于计算掩码对比损失）"""
        return self.masked_features_cache


