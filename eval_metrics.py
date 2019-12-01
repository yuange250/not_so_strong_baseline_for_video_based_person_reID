from __future__ import print_function, absolute_import
import numpy as np
import copy
import torch

from re_ranking import re_ranking


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP




def evaluate_reranking(qf, q_pids, q_camids, gf, g_pids, g_camids, ranks):
    m, n = qf.size(0), gf.size(0)

    q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    q_g_dist.addmm_(1, -2, qf, gf.t())
    q_q_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
               torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
    q_q_dist.addmm_(1, -2, qf, qf.t())
    g_g_dist = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
    g_g_dist.addmm_(1, -2, gf, gf.t())

    q_g_dist = q_g_dist.cpu().numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(q_g_dist, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    q_q_dist = q_q_dist.cpu().numpy()
    g_g_dist = g_g_dist.cpu().numpy()
    rerank_dis = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    print("Computing rerank CMC and mAP")
    rerank_cmc, rerank_mAP = evaluate(rerank_dis, q_pids, g_pids, q_camids, g_camids)

    print("rerank Results ----------")
    print("mAP: {:.1%}".format(rerank_mAP))
    print("CMC curve")
    for r in ranks:
        print("rerank Rank-{:<3}: {:.1%}".format(r, rerank_cmc[r - 1]))
    print("------------------")
    return cmc + [mAP]

if __name__ == "__main__":
    a = np.random.rand(3, 2)
    b = np.random.rand(4, 2)
    q_g_dist = np.power(a, 2).sum(1, keepdims=True).repeat(4, axis=1) + \
               np.power(b, 2).sum(1, keepdims=True).repeat(3, axis=1).t()
    q_g_dist = q_g_dist - 2 * a.matmul(b.t())

    a = torch.Tensor(a)
    b = torch.Tensor(b)
    q_g_dist2 = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(3, 4) + \
               torch.pow(b, 2).sum(dim=1, keepdim=True).expand(4, 3).t()
    q_g_dist2.addmm_(1, -2, a, b.t())
