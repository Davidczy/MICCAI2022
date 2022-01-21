import torch
import torch.nn as nn
import torch.nn.functional as func
# from mmcv.cnn import normal_init


class MedianTripletHead(nn.Module):
    def __init__(self, predictor, gamma, size_average=True):
        super(MedianTripletHead, self).__init__()
        self.predictor = predictor#没有builder类
        self.size_average = size_average
        self.ranking_loss = nn.MarginRankingLoss(margin=100.)
        self.gamma = gamma

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        pred = self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        n = input.size(0)
        dist = -2. * torch.matmul(pred_norm, target_norm.t()) 
        idx = torch.arange(n)
        mask = idx.expand(n, n).eq(idx.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].median().unsqueeze(0))
            #down_k = torch.topk(dist[i][mask[i]==0], 5, dim=-1, largest=False)
            #down_k = down_k[0][-1].unsqueeze(0)
            #dist_an.append(down_k)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss_triplet = self.ranking_loss(dist_an, self.gamma * dist_ap, y)
        return loss_triplet


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class unifyloss(nn.Module):
    def __init__(self, tt=1, ts=1):
        super(unifyloss, self).__init__()
        self.tt = tt
        self.ts = ts

    def forward(self, feature1, feature2):
        f1 = func.softmax(feature1.div_(self.tt))
        f2 = func.softmax(feature2.div_(self.ts)).exp_()
        loss = torch.mm(f1, f2.T)
        loss = loss.mean()
        print(loss)
        return loss

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def HCLloss(out_1,out_2,tau_plus,batch_size,beta, estimator):
        # neg score
        temperature = 7000
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        old_neg = neg.clone()
        mask = get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)
        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)
        # negative samples similarity scoring
        if estimator=='hard':
            N = batch_size * 2 - 2
            imp = (beta* neg.log()).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        elif estimator=='easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
            
        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()

        return loss