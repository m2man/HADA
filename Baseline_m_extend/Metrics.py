import numpy as np
import torch
import torch.nn as nn
device = torch.device('cuda:0')
from torch.autograd import Variable

# ===== FROM SGM =====
def xattn_score_t2i(images, captions, obj_nums=None, cap_lens=None):
    """
    Images: (n_image, max_n_objs, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    """
    similarities = []
    n_image = images.size(0)
    max_obj = images.size(1)
    n_caption = captions.size(0)
    max_n_word = captions.size(1)
    
    if obj_nums is None:
        obj_nums = [max_obj for x in range(n_image)]
    if cap_lens is None:
        cap_lens = [max_n_word for x in range(n_caption)]
        cap_lens = torch.tensor(cap_lens, dtype=captions.dtype)
    cap_lens = Variable(cap_lens).to(device)
    captions = torch.transpose(captions, 1, 2)
    
    for i in range(n_image):
        n_obj = obj_nums[i]
        if n_obj == 0:
            img_i = images[i, :, :].unsqueeze(0).contiguous()
        else:
            img_i = images[i, : n_obj, :].unsqueeze(0).contiguous()
        # --> (n_caption , n_region ,d)
        img_i_expand = img_i.repeat(n_caption, 1, 1)
        # --> (n_caption, d, max_n_word)
        dot = torch.bmm(img_i_expand, captions)
        # if opt.clamp:
        #     dot = torch.clamp(dot, min=0)
        dot = dot.max(dim=1, keepdim=True)[0].squeeze()
        dot = dot.view(n_caption, -1).contiguous()
        dot = dot.sum(dim=1, keepdim=True)
        cap_lens = cap_lens.contiguous().view(-1,1)
        dot = dot/(cap_lens+1e-6)
        # dot = dot.mean(dim=1, keepdim=True)
        dot = torch.transpose(dot, 0, 1)
        similarities.append(dot)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 0)
    
    return similarities


def xattn_score_i2t(images, captions, obj_nums=None, cap_lens=None):
    """
    Images: (batch_size, max_n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    """
    similarities = []
    n_image = images.size(0)
    max_obj = images.size(1)
    n_caption = captions.size(0)
    max_n_word = captions.size(1)

    if cap_lens is None:
        cap_lens = [max_n_word for x in range(n_caption)]
    if obj_nums is None:
        obj_nums = [max_obj for x in range(n_image)]
        obj_nums = torch.tensor(obj_nums, dtype=images.dtype)
    obj_nums = Variable(obj_nums).to(device)
    
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        if n_word == 0:
            cap_i = captions[i, :, :].unsqueeze(0).contiguous()
        else:
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        cap_i_expand = cap_i_expand.contiguous()
        cap_i_expand = torch.transpose(cap_i_expand, 1,2)
        dot = torch.bmm(images, cap_i_expand)
        # if opt.clamp:
        #     dot = torch.clamp(dot, min=0)
        dot = dot.max(dim=2, keepdim=True)[0].squeeze()
        dot = dot.sum(dim=1, keepdim=True)
        obj_nums = obj_nums.contiguous().view(-1,1)
        dot = dot/(obj_nums+1e-6)
        similarities.append(dot)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities
# ===== END SGM =====




# ===== FROM LGSGM =====
def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def CosineSimilarity(images_geb, captions_geb):
    similarities = sim_matrix(images_geb, captions_geb) # n_img, n_caption
    return similarities

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        # self.cross_attn = cross_attn

    def forward(self, im, s, im_l=None, s_l=None, cross_attn='i2t'):
        # compute image-sentence score matrix
        if cross_attn == 't2i':
            scores = xattn_score_t2i(im, s, im_l, s_l)
        elif cross_attn == 'i2t':
            scores = xattn_score_i2t(im, s, im_l, s_l)
        else:
            raise ValueError("unknown first norm type")

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.mean() + cost_im.mean()
    
class ContrastiveLoss_CosineSimilarity(nn.Module):
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss_CosineSimilarity, self).__init__()
        self.max_violation = max_violation
        self.margin = margin
        
    def forward(self, images_geb, captions_geb):
        scores = CosineSimilarity(images_geb, captions_geb)
        diagonal = scores.diag().view(len(images_geb), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.mean() + cost_im.mean()
# ===== END LGSGM =====



# ===== LIGHTNINGDOT =====
from torch import Tensor as T
import torch.nn.functional as F

def dot_product_scores(q_vectors: T, ctx_vectors: T, cosine=False) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    if cosine:
        n1 = torch.norm(q_vectors, dim=-1)
        n2 = torch.norm(ctx_vectors, dim=-1)
        n_out = torch.ger(n1, n2)
        return r / n_out
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)

class Dot_NLLLoss(object):
    def calc_1(self, imgs, caps, positive_idx_per_question=None):
        with torch.no_grad():
            n_imgs, n_caps = imgs.shape[0], caps.shape[0]
            if positive_idx_per_question is None:
                positive_idx_per_question = [x for x in range(n_imgs)]
        scores = imgs @ caps.T      
        # scores = dot_product_scores(imgs, caps)        
        if len(imgs.size()) > 1:
            q_num = imgs.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(softmax_scores, torch.tensor(positive_idx_per_question).to(softmax_scores.device),
                          reduction='mean')
        with torch.no_grad():
            max_score, max_idxs = torch.max(softmax_scores, 1)
            correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()
        return loss, correct_predictions_count, scores
    
    def calc_multi(self, imgs, caps, list_img_cap):
        # list_img_cap (imgid_txtorder: 1201312_0, 1201312_4)
        # same imgid --> captions for same image
        # imgs (B,F), caps (B,F)
        # positive_idx_per_question is list of list indicate index of caps that match imgs
        with torch.no_grad():
            bs = len(list_img_id)
            img_id_only = [x.split('_')[0] for x in list_img_cap]
            weight = []
            for idx in range(bs):
                weight.append([1 if x == img_id_only[idx] else 0 for x in img_id_only])
        weight = torch.tensor(weight).to(imgs.device)
        scores = imgs @ caps.T 
        loss = -torch.sum(F.log_softmax(scores, dim=1)*weight,dim=1).mean()
        return loss, None, scores