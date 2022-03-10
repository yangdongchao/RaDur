# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 16:33
# @Author  : dongchao yang
# @File    : train.py
from builtins import print
import ignite.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
class BCELoss(nn.Module):
    """docstring for BCELoss"""
    def __init__(self):
        super().__init__()
    def forward(self,frame_prob, tar,frame_level_time=None):
        return nn.functional.binary_cross_entropy(input=frame_prob, target=tar)

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,logit,target):
        _,label = torch.max(target,dim=-1)
        label = label.long()
        return nn.CrossEntropyLoss()(logit, label)
class BCELossWithLabelSmoothing(nn.Module):
    """docstring for BCELoss"""
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
    def forward(self, clip_prob, frame_prob, tar):
        n_classes = clip_prob.shape[-1]
        with torch.no_grad():
            tar = tar * (1 - self.label_smoothing) + (
                1 - tar) * self.label_smoothing / (n_classes - 1)
        return nn.functional.binary_cross_entropy(clip_prob, tar)

class Loss_join(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_join, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
        self.CE = CELoss()

    def update(self, output):
        decision,decision_up, frame_level_target, time,embed_label,logit = output
        #average_loss = self._loss_fn(decision,frame_level_target) + self.CE(logit,embed_label)
        average_loss = self._loss_fn(decision,frame_level_target)
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N

class Loss_join_clr(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_join_clr, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
        self.CLR = SniCoLoss()

    def update(self, output):
        decision,decision_up, frame_level_target, time,embed_label,logit,contrast_pairs = output
        average_loss = self._loss_fn(decision,frame_level_target) + 0.01*self.CLR(contrast_pairs)
        #average_loss = self._loss_fn(decision,frame_level_target)
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N

class Loss(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)

    def update(self, output):
        decision,decision_up, frame_level_target, time,frame_level_time = output
        # print('decision ',decision[0,:5])
        # print('frame_level_target ',frame_level_target[0,:5])
        # print('decision.shape ',decision.shape)
        # print('frame_level_target.shape ',frame_level_target.shape)
        average_loss = self._loss_fn(decision,frame_level_target,frame_level_time)
        # print('average_loss ',average_loss)
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N

class Loss_dc(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_dc, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)

    def update(self, output):
        decision,decision_up, frame_level_target, time,frame_level_time,inputs,recon_spec= output
        average_loss = self._loss_fn(decision,frame_level_target,recon_spec,inputs, frame_level_time)
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N

class Loss_dc2(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss_dc2, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)
        self.bce = BCELoss()

    def update(self, output):
        decision,decision_up,decision2, decision_up2, frame_level_target, time,frame_level_time,inputs,recon_spec= output
        average_loss = self._loss_fn(decision,frame_level_target,recon_spec,inputs, frame_level_time)
        neg_frame_level_target = torch.zeros((frame_level_target.shape[0],frame_level_target.shape[1])).cuda()
        neg_loss = self.bce(decision2,neg_frame_level_target)
        loss = average_loss + neg_loss
        if len(loss.shape) != 0:
            raise ValueError('loss_fn did not return loss.')
        N = self._batch_size(output[0])
        self._sum += loss.item() * N
        self._num_examples += N

class DCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = FocalLoss()
    def forward(self,frame_prob, tar,predict_spec,true_spec,frame_level_time=None):
        # print('frame_prob ',frame_prob.shape)
        # print('tar ',tar.shape)
        # print('predict_spec ',predict_spec.shape)
        # print('true_spec ',true_spec.shape)
        loss_res = F.mse_loss(true_spec, predict_spec, reduction='mean')
        loss_sed = self.bce(frame_prob, tar)
        # print('loss_res ',loss_res)
        # print('loss_sed ',loss_sed)
        # assert 1==2
        loss = loss_sed + 0.05*loss_res
        return loss 
class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()

    def forward(self, video_scores, label):
        label = label / torch.sum(label, dim=1, keepdim=True)
        loss = self.bce_criterion(video_scores, label)
        return loss

class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.25):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)
        return loss

    def consin(self,q,k,neg):
        neg = torch.mean(neg,1)
        loss_w = 0.6*torch.cosine_similarity(q,k,dim=1).mean() - 0.4*torch.cosine_similarity(q,neg,dim=1).mean()
        # print('loss_w ',loss_w)
        return loss_w
    
    def forward(self, contrast_pairs):
        HA_refinement = self.consin(
            torch.mean(contrast_pairs['HA'], 1), 
            torch.mean(contrast_pairs['EA'], 1), 
            contrast_pairs['EB']
        )

        HB_refinement = self.consin(
            torch.mean(contrast_pairs['HB'], 1), 
            torch.mean(contrast_pairs['EB'], 1), 
            contrast_pairs['EA']
        )
        # HA_refinement_consin = self.consin(
        #     torch.mean(contrast_pairs['HA'], 1), 
        #     torch.mean(contrast_pairs['EA'], 1), 
        #     contrast_pairs['EB']
        # )

        # HB_refinement_consin = self.consin(
        #     torch.mean(contrast_pairs['HB'], 1), 
        #     torch.mean(contrast_pairs['EB'], 1), 
        #     contrast_pairs['EA']
        # )
        # loss = HA_refinement + HB_refinement + HA_refinement_consin + HB_refinement_consin
        loss = HA_refinement + HB_refinement
        return loss
        
class SniCoLoss_consin(nn.Module):
    def __init__(self):
        super(SniCoLoss_consin, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.5):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)
        return loss

    def consin(self,q,k,neg):
        neg = torch.mean(neg,1)
        loss_w = 0.6*torch.cosine_similarity(q,k,dim=1).mean() - 0.4*torch.cosine_similarity(q,neg,dim=1).mean()
        # print('loss_w ',loss_w)
        return loss_w
    
    def forward(self, contrast_pairs):
        # HA_refinement = self.NCE(
        #     torch.mean(contrast_pairs['HA'], 1), 
        #     torch.mean(contrast_pairs['EA'], 1), 
        #     contrast_pairs['EB']
        # )

        # HB_refinement = self.NCE(
        #     torch.mean(contrast_pairs['HB'], 1), 
        #     torch.mean(contrast_pairs['EB'], 1), 
        #     contrast_pairs['EA']
        # )
        HA_refinement = self.consin(
            torch.mean(contrast_pairs['HA'], 1), 
            torch.mean(contrast_pairs['EA'], 1), 
            contrast_pairs['EB']
        )

        HB_refinement = self.consin(
            torch.mean(contrast_pairs['HB'], 1), 
            torch.mean(contrast_pairs['EB'], 1), 
            contrast_pairs['EA']
        )
        loss = HA_refinement + HB_refinement
        return loss
class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()

    def forward(self, video_scores, label, contrast_pairs):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_total = loss_cls + 0.01 * loss_snico

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.65, gamma=2):
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        # print(torch.log(predict).shape)
        # print('target ',target.shape)
        # assert 1==2
        loss = (-target*tmp*torch.log(predict+self.eps)).mean() + (-(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        #loss_bce = (-target*torch.log(predict)).mean() + (-(1-target)*torch.log(1-predict)).mean()
        # print('loss_bce ',loss_bce)
        # print(self.bce(predict,target))
        # print('loss ',loss)
        # assert 1==2
        return loss

class FocalLoss_gamma1(nn.Module):
    def __init__(self,alpha=0.65, gamma=2):
        super(FocalLoss_gamma1,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        # print(torch.log(predict).shape)
        # print('target ',target.shape)
        # assert 1==2
        loss = (-target*tmp*torch.log(predict+self.eps)).mean() + (-(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        #loss_bce = (-target*torch.log(predict)).mean() + (-(1-target)*torch.log(1-predict)).mean()
        # print('loss_bce ',loss_bce)
        # print(self.bce(predict,target))
        # print('loss ',loss)
        # assert 1==2
        return loss

class Du_FocalLoss(nn.Module):
    def __init__(self,alpha=0.65, gamma=2, w=1):
        super(Du_FocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss05(nn.Module):
    def __init__(self,alpha=0.5, gamma=2, w=1):
        super(Du_FocalLoss05,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss05_a02(nn.Module):
    def __init__(self,alpha=0.5, gamma=2, w=1):
        super(Du_FocalLoss05_a02, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.2
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss05_a05(nn.Module):
    def __init__(self,alpha=0.5, gamma=2.0, w=1):
        super(Du_FocalLoss05_a05, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss05_a08(nn.Module):
    def __init__(self,alpha=0.5, gamma=2.0, w=1):
        super(Du_FocalLoss05_a08, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.8
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss05_a12(nn.Module):
    def __init__(self,alpha=0.5, gamma=2.0, w=1):
        super(Du_FocalLoss05_a12, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 1.2
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss05_a15(nn.Module):
    def __init__(self,alpha=0.5, gamma=2.0, w=1):
        super(Du_FocalLoss05_a15, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 1.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss65_a02(nn.Module):
    def __init__(self,alpha=0.65, gamma=2, w=1):
        super(Du_FocalLoss65_a02, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.2
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss65_a05(nn.Module):
    def __init__(self,alpha=0.65, gamma=2.0, w=1):
        super(Du_FocalLoss65_a05, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss65_a08(nn.Module):
    def __init__(self,alpha=0.65, gamma=2.0, w=1):
        super(Du_FocalLoss65_a08, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.8
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss65_a12(nn.Module):
    def __init__(self,alpha=0.65, gamma=2.0, w=1):
        super(Du_FocalLoss65_a12, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 1.2
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss65_a15(nn.Module):
    def __init__(self,alpha=0.65, gamma=2.0, w=1):
        super(Du_FocalLoss65_a15, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 1.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss75_a02(nn.Module):
    def __init__(self,alpha=0.75, gamma=2, w=1):
        super(Du_FocalLoss75_a02, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.2
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss75_a05(nn.Module):
    def __init__(self,alpha=0.75, gamma=2.0, w=1):
        super(Du_FocalLoss75_a05, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss75_a08(nn.Module):
    def __init__(self,alpha=0.75, gamma=2.0, w=1):
        super(Du_FocalLoss75_a08, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.8
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss75_a12(nn.Module):
    def __init__(self,alpha=0.75, gamma=2.0, w=1):
        super(Du_FocalLoss75_a12, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 1.2
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss75_a15(nn.Module):
    def __init__(self,alpha=0.75, gamma=2.0, w=1):
        super(Du_FocalLoss75_a15, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 1.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss85_a02(nn.Module):
    def __init__(self,alpha=0.85, gamma=2, w=1):
        super(Du_FocalLoss85_a02, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.2
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss85_a05(nn.Module):
    def __init__(self,alpha=0.85, gamma=2.0, w=1):
        super(Du_FocalLoss85_a05, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss85_a08(nn.Module):
    def __init__(self,alpha=0.85, gamma=2.0, w=1):
        super(Du_FocalLoss85_a08, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.8
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss85_a12(nn.Module):
    def __init__(self,alpha=0.85, gamma=2.0, w=1):
        super(Du_FocalLoss85_a12, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 1.2
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss85_a15(nn.Module):
    def __init__(self,alpha=0.85, gamma=2.0, w=1):
        super(Du_FocalLoss85_a15, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 1.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss



























class Du_FocalLoss065_a05(nn.Module):
    def __init__(self,alpha=0.65, gamma=2.0, w=1):
        super(Du_FocalLoss065_gamma05, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss065_a10(nn.Module):
    def __init__(self,alpha=0.65, gamma=2.0, w=1):
        super(Du_FocalLoss065_gamma10, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 1.0
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss065_a15(nn.Module):
    def __init__(self,alpha=0.65, gamma=2.0, w=1):
        super(Du_FocalLoss065_gamma15, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 1.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss065_gamma20(nn.Module):
    def __init__(self,alpha=0.65, gamma=2.0, w=1):
        super(Du_FocalLoss065_gamma20, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss


class Du_FocalLoss075_gamma05(nn.Module):
    def __init__(self,alpha=0.75, gamma=0.5, w=1):
        super(Du_FocalLoss075_gamma05, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss075_gamma10(nn.Module):
    def __init__(self,alpha=0.75, gamma=1.0, w=1):
        super(Du_FocalLoss075_gamma10, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss075_gamma15(nn.Module):
    def __init__(self,alpha=0.75, gamma=1.5, w=1):
        super(Du_FocalLoss075_gamma15, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss075_gamma20(nn.Module):
    def __init__(self,alpha=0.75, gamma=2.0, w=1):
        super(Du_FocalLoss075_gamma20, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss


class Du_FocalLoss085_gamma05(nn.Module):
    def __init__(self,alpha=0.85, gamma=0.5, w=1):
        super(Du_FocalLoss085_gamma05, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss085_gamma10(nn.Module):
    def __init__(self,alpha=0.85, gamma=1.0, w=1):
        super(Du_FocalLoss085_gamma10, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss085_gamma15(nn.Module):
    def __init__(self,alpha=0.85, gamma=1.5, w=1):
        super(Du_FocalLoss085_gamma15, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss085_gamma20(nn.Module):
    def __init__(self,alpha=0.85, gamma=2.0, w=1):
        super(Du_FocalLoss085_gamma20, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-the_a*(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class Du_FocalLoss_pos(nn.Module):
    def __init__(self,alpha=0.65, gamma=2, w=1):
        super(Du_FocalLoss_pos,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        self.a = 0.5
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        w_max = 4.5399929762484854e-5
        up_ = torch.exp(-self.w*frame_level_time)-w_max # e(-w)-e(-10)
        down_ = 1-w_max
        the_a = (1+self.a*(up_/down_))
        loss = (-the_a*target*tmp*torch.log(predict+self.eps)).mean() + (-(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        return loss

class FocalLoss_no(nn.Module):
    def __init__(self,alpha=0.65, gamma=2):
        super(FocalLoss_no,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha) # t=0
        # print('tmp2 ',tmp2.shape)
        # print(torch.log(predict).shape)
        # print('target ',target.shape)
        # assert 1==2
        loss = (-target*tmp*torch.log(predict+self.eps)).mean() + (-(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        #loss_bce = (-target*torch.log(predict)).mean() + (-(1-target)*torch.log(1-predict)).mean()
        # print('loss_bce ',loss_bce)
        # print(self.bce(predict,target))
        # print('loss ',loss)
        # assert 1==2
        return loss

class FocalLoss05(nn.Module):
    def __init__(self,alpha=0.5, gamma=2):
        super(FocalLoss05,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        # print(torch.log(predict).shape)
        # print('target ',target.shape)
        # assert 1==2
        loss = (-target*tmp*torch.log(predict)).mean() + (-(1-target)*tmp2*torch.log(1-predict)).mean()
        #loss_bce = (-target*torch.log(predict)).mean() + (-(1-target)*torch.log(1-predict)).mean()
        # print('loss_bce ',loss_bce)
        # print(self.bce(predict,target))
        # print('loss ',loss)
        # assert 1==2
        return loss
class FocalLoss_slack_time(nn.Module):
    def __init__(self,alpha=0.65, gamma=2,w=0.25):
        super(FocalLoss_slack_time,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        slack_time = torch.exp(-self.w*frame_level_time)
        tmp = slack_time*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-slack_time)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        # print(torch.log(predict).shape)
        # print('target ',target.shape)
        # assert 1==2
        loss = (-target*tmp*torch.log(predict)).mean() + (-(1-target)*tmp2*torch.log(1-predict)).mean()
        # loss_bce = (-target*torch.log(predict)).mean() + (-(1-target)*torch.log(1-predict)).mean()
        # print('loss_bce ',loss_bce)
        # print(self.bce(predict,target))
        # print('loss ',loss)
        # assert 1==2
        return loss

class FocalLoss_plus(nn.Module):
    def __init__(self,alpha=0.75, gamma=2,w=0.25):
        super(FocalLoss_plus,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        # print('frame_level_time ',frame_level_time[0:5])
        slack_time = torch.exp(-self.w*frame_level_time)
        # print('slack_time ',slack_time[0:5])
        # assert 1==2
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        # print(torch.log(predict).shape)
        # print('target ',target.shape)
        # assert 1==2
        loss = (-target*tmp*slack_time*torch.log(predict)).mean() + (-(1-target)*tmp2*torch.log(1-predict)).mean()
        #loss = (-target*slack_time*torch.log(predict)).mean() + (-(1-target)*torch.log(1-predict)).mean()
        #loss_bce = (-target*torch.log(predict)).mean() + (-(1-target)*torch.log(1-predict)).mean()
        # print('loss_bce ',loss_bce)
        # print(self.bce(predict,target))
        # print('loss ',loss)
        # assert 1==2
        return loss

class FocalLoss_plus_total_time(nn.Module):
    def __init__(self,alpha=0.6, gamma=2,w=0.25):
        super(FocalLoss_plus_total_time,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.w = w
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        # print('frame_level_time ',frame_level_time[0:5])
        slack_time = 1.0 + frame_level_time*2.0
        # print('slack_time ',slack_time[:,0])
        # assert 1==2
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # print('tmp2 ',tmp2.shape)
        loss = (-target*tmp*slack_time*torch.log(predict)).mean() + (-(1-target)*tmp2*torch.log(1-predict)).mean()
        #loss = (-target*slack_time*torch.log(predict)).mean() + (-(1-target)*torch.log(1-predict)).mean()
        return loss
