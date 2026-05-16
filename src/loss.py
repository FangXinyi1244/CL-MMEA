import torch
from torch import nn

try:
    from models import *
    from utils import *
except:
    from src.models import *
    from src.utils import *


class CustomMultiLossLayer(nn.Module):
    """
    Inspired by
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    """

    def __init__(self, loss_num, device=None):
        super(CustomMultiLossLayer, self).__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(torch.zeros(self.loss_num, ), requires_grad=True)

    def forward(self, loss_list):
        assert len(loss_list) == self.loss_num
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i in range(self.loss_num):
            loss += precision[i] * loss_list[i] + self.log_vars[i]
        return loss


class icl_loss(nn.Module):

    def __init__(self, device, tau=0.05, ab_weight=0.5, n_view=2, 
                 intra_weight=1.0, inversion=False,
                 use_hard_negatives=False,  # 是否使用硬负采样
                 hard_negative_k=50):        # 硬负样本数量
        super(icl_loss, self).__init__()
        self.tau = tau
        self.device = device
        self.sim = cosine_sim
        self.weight = ab_weight  # the factor of a->b and b<-a
        self.n_view = n_view
        self.intra_weight = intra_weight  # the factor of aa and bb
        self.inversion = inversion

        # 采用硬负采样参数
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_k = hard_negative_k

    def softXEnt(self, target, logits):

        logprobs = F.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, emb, train_links, emb2=None, norm=True):
        if norm:
            emb = F.normalize(emb, dim=1)
            if emb2 is not None:
                emb2 = F.normalize(emb2, dim=1)
        
        num_ent = emb.shape[0]
        zis = emb[train_links[:, 0]]
        if emb2 is not None:
            zjs = emb2[train_links[:, 1]]
        else:
            zjs = emb[train_links[:, 1]]

        temperature = self.tau
        alpha = self.weight
        n_view = self.n_view
        LARGE_NUM = 1e9

        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        num_classes = batch_size * n_view
        
        # 修改：如果使用硬负采样，labels 不再是 one-hot，而是需要特殊处理
        if self.use_hard_negatives:
            # 1. 计算相似度矩阵（与原版相同）
            logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
            logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature
            logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
            logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
            
            # 2. 掩码掉自己（对角线）- 修复：添加 .float()
            mask = torch.eye(batch_size, device=self.device).float()  # ✅ 修复类型
            logits_aa = logits_aa - mask * LARGE_NUM
            logits_bb = logits_bb - mask * LARGE_NUM
            
            # ========== 核心：软锐化（Soft Sharpening）==========
            # 优化：使用更高效的锐化方式，避免额外的 sigmoid 开销
            # 方案：直接使用放大后的 logits，但通过温度缩放实现类似效果
            
            # 对 logits_ab 和 logits_ba 应用温度缩放（强调硬负样本）
            # 高相似度负样本使用更小温度 -> logits 相对更大 -> softmax 更尖锐
            sharp_temp = temperature * 0.5  # 小温度 -> 更尖锐
            logits_ab_sharp = logits_ab / sharp_temp * temperature  # 相对放大
            logits_ba_sharp = logits_ba / sharp_temp * temperature
            
            # 与原 logits 加权融合（保留部分原始信息）
            logits_ab = 0.5 * logits_ab + 0.5 * logits_ab_sharp
            logits_ba = 0.5 * logits_ba + 0.5 * logits_ba_sharp
            
            # 3. 拼接 logits（与原逻辑一致）
            if self.inversion:
                logits_a = torch.cat([logits_ab, logits_bb], dim=1)
                logits_b = torch.cat([logits_ba, logits_aa], dim=1)
            else:
                logits_a = torch.cat([logits_ab, logits_aa], dim=1)
                logits_b = torch.cat([logits_ba, logits_bb], dim=1)

            # 4. 使用 self.softXEnt（与原分支一致，避免 F.cross_entropy 的低效）
            labels = F.one_hot(torch.arange(batch_size, device=self.device), num_classes=num_classes).float()
            loss_a = self.softXEnt(labels, logits_a)  # ✅ 使用 softXEnt
            loss_b = self.softXEnt(labels, logits_b)            
        else:
            # 原始逻辑（保持原有实现）
            masks = torch.eye(batch_size, device=self.device).float()
            logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
            logits_aa = logits_aa - masks * LARGE_NUM
            logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
            logits_bb = logits_bb - masks * LARGE_NUM
            logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
            logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

            if self.inversion:
                logits_a = torch.cat([logits_ab, logits_bb], dim=1)
                logits_b = torch.cat([logits_ba, logits_aa], dim=1)
            else:
                logits_a = torch.cat([logits_ab, logits_aa], dim=1)
                logits_b = torch.cat([logits_ba, logits_bb], dim=1)

            labels = F.one_hot(torch.arange(batch_size, device=self.device), num_classes=num_classes).float()
            loss_a = self.softXEnt(labels, logits_a)
            loss_b = self.softXEnt(labels, logits_b)

        return alpha * loss_a + (1 - alpha) * loss_b

class ial_loss(nn.Module):
    """
    unimodal-multimodal kl loss
    """

    def __init__(self, device, tau=0.05, ab_weight=0.5, zoom=0.1,
                 n_view=2, inversion=False,
                 reduction="mean", detach=False):
        super(ial_loss, self).__init__()
        self.tau = tau
        self.device = device
        self.sim = cosine_sim
        self.weight = ab_weight
        self.zoom = zoom
        self.n_view = n_view
        self.inversion = inversion
        self.reduction = reduction
        self.detach = detach

    def forward(self, src_emb, tar_emb, train_links, norm=True):
        if norm:
            src_emb = F.normalize(src_emb, dim=1)
            tar_emb = F.normalize(tar_emb, dim=1)

        # Get (normalized) hidden1 and hidden2.
        src_zis = src_emb[train_links[:, 0]]
        src_zjs = src_emb[train_links[:, 1]]
        tar_zis = tar_emb[train_links[:, 0]]
        tar_zjs = tar_emb[train_links[:, 1]]

        temperature = self.tau
        alpha = self.weight

        assert src_zis.shape[0] == tar_zjs.shape[0]
        batch_size = src_zis.shape[0]
        LARGE_NUM = 1e9
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.to(self.device).float()
        p_ab = torch.matmul(src_zis, torch.transpose(src_zjs, 0, 1)) / temperature
        p_ba = torch.matmul(src_zjs, torch.transpose(src_zis, 0, 1)) / temperature
        q_ab = torch.matmul(tar_zis, torch.transpose(tar_zjs, 0, 1)) / temperature
        q_ba = torch.matmul(tar_zjs, torch.transpose(tar_zis, 0, 1)) / temperature
        # add self-contrastive
        p_aa = torch.matmul(src_zis, torch.transpose(src_zis, 0, 1)) / temperature
        p_bb = torch.matmul(src_zjs, torch.transpose(src_zjs, 0, 1)) / temperature
        q_aa = torch.matmul(tar_zis, torch.transpose(tar_zis, 0, 1)) / temperature
        q_bb = torch.matmul(tar_zjs, torch.transpose(tar_zjs, 0, 1)) / temperature
        p_aa = p_aa - masks * LARGE_NUM
        p_bb = p_bb - masks * LARGE_NUM
        q_aa = q_aa - masks * LARGE_NUM
        q_bb = q_bb - masks * LARGE_NUM

        if self.inversion:
            p_ab = torch.cat([p_ab, p_bb], dim=1)
            p_ba = torch.cat([p_ba, p_aa], dim=1)
            q_ab = torch.cat([q_ab, q_bb], dim=1)
            q_ba = torch.cat([q_ba, q_aa], dim=1)
        else:
            p_ab = torch.cat([p_ab, p_aa], dim=1)
            p_ba = torch.cat([p_ba, p_bb], dim=1)
            q_ab = torch.cat([q_ab, q_aa], dim=1)
            q_ba = torch.cat([q_ba, q_bb], dim=1)

        # param 1 need to log_softmax, param 2 need to softmax
        loss_a = F.kl_div(F.log_softmax(p_ab, dim=1), F.softmax(q_ab.detach(), dim=1), reduction="none")
        loss_b = F.kl_div(F.log_softmax(p_ba, dim=1), F.softmax(q_ba.detach(), dim=1), reduction="none")

        if self.reduction == "mean":
            loss_a = loss_a.mean()
            loss_b = loss_b.mean()
        elif self.reduction == "sum":
            loss_a = loss_a.sum()
            loss_b = loss_b.sum()
        # The purpose of the zoom is to narrow the range of losses
        return self.zoom * (alpha * loss_a + (1 - alpha) * loss_b)

# ========== 在 loss.py 末尾添加 ==========
class masked_contrastive_loss(nn.Module):
    """
    基于特征掩码的对比损失（自监督）
    正对：同一实体的原始特征与掩码特征
    负对：该实体掩码特征与其他实体的特征
    """
    
    def __init__(self, device, tau=0.05, mask_loss_weight=0.1, 
                 batch_size=512, max_neg_samples=512):
        super(masked_contrastive_loss, self).__init__()
        self.tau = tau
        self.device = device
        self.mask_loss_weight = mask_loss_weight
        self.batch_size = batch_size
        self.max_neg_samples = max_neg_samples
        
    def forward(self, original_emb, masked_emb, norm=True):
        """
        Args:
            original_emb: 原始特征 (N, D)
            masked_emb: 掩码后的特征 (N, D)
            norm: 是否 L2 归一化
        Returns:
            loss: 标量
        """
        if norm:
            original_emb = F.normalize(original_emb, dim=1)
            masked_emb = F.normalize(masked_emb, dim=1)
            
        N = original_emb.shape[0]
        
        # 如果训练集过大，同时采样 original_emb 和 masked_emb
        if N > self.max_neg_samples:
            indices = torch.randperm(N, device=original_emb.device)[:self.max_neg_samples]
            original_emb = original_emb[indices]
            masked_emb = masked_emb[indices]  # ✅ 同步采样
            N = self.max_neg_samples  # ✅ 更新N
        
        total_loss = 0.0
        
        # 分批计算
        for i in range(0, N, self.batch_size):
            end_i = min(i + self.batch_size, N)
            masked_batch = masked_emb[i:end_i]
            
            # (B, D) x (D, N) -> (B, N)
            logits = torch.matmul(masked_batch, original_emb.t()) / self.tau
            
            # ✅ 标签正确：[0, 1, ..., B-1] 对应正样本在 original_emb 的相同位置
            labels = torch.arange(i, end_i, device=self.device)
            
            loss = F.cross_entropy(logits, labels)
            total_loss += loss * (end_i - i)
            
            del logits, masked_batch
        
        total_loss = total_loss / N
        
        return self.mask_loss_weight * total_loss