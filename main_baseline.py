from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

from torch.utils.data import DataLoader

import data_manager
from samplers import RandomIdentitySampler
from video_loader import VideoDataset

try:
    import apex
except:
    pass
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from lr_schedulers import WarmupMultiStepLR
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, TripletLoss, TripletLossAttrWeightes, CosineTripletLoss
from utils import AverageMeter, Logger, AttributesMeter, EMA, make_optimizer
from eval_metrics import evaluate_reranking
from config import cfg

parser = argparse.ArgumentParser(description="ReID Baseline Training")
parser.add_argument(
    "--config_file", default="./configs/softmax_triplet.yml", help="path to config file", type=str
)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)

args_ = parser.parse_args()

if args_.config_file != "":
    cfg.merge_from_file(args_.config_file)
cfg.merge_from_list(args_.opts)

tqdm_enable = False

def main():
    runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, runId)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    print(cfg.OUTPUT_DIR)
    torch.manual_seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    use_gpu = torch.cuda.is_available() and cfg.MODEL.DEVICE == "cuda"
    if not cfg.EVALUATE_ONLY:
        sys.stdout = Logger(osp.join(cfg.OUTPUT_DIR, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(cfg.OUTPUT_DIR, 'log_test.txt'))

    print("==========\nConfigs:{}\n==========".format(cfg))

    if use_gpu:
        print("Currently using GPU {}".format(cfg.MODEL.DEVICE_ID))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(cfg.DATASETS.NAME))


    dataset = data_manager.init_dataset(root=cfg.DATASETS.ROOT_DIR, name=cfg.DATASETS.NAME)
    print("Initializing model: {}".format(cfg.MODEL.NAME))

    if cfg.MODEL.ARCH == 'video_baseline':
        torch.backends.cudnn.benchmark = False
        model = models.init_model(name=cfg.MODEL.ARCH, num_classes=625, pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE,
                                  last_stride=cfg.MODEL.LAST_STRIDE,
                                  neck=cfg.MODEL.NECK, model_name=cfg.MODEL.NAME, neck_feat=cfg.TEST.NECK_FEAT,
                                  model_path=cfg.MODEL.PRETRAIN_PATH)

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))



    transform_train = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])
    transform_test = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pin_memory = True if use_gpu else False

    cfg.DATALOADER.NUM_WORKERS = 0

    trainloader = DataLoader(
        VideoDataset(dataset.train, seq_len=cfg.DATASETS.SEQ_LEN, sample=cfg.DATASETS.TRAIN_SAMPLE_METHOD, transform=transform_train,
                     dataset_name=cfg.DATASETS.NAME),
        sampler=RandomIdentitySampler(dataset.train, num_instances=cfg.DATALOADER.NUM_INSTANCE),
        batch_size=cfg.SOLVER.SEQS_PER_BATCH, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=True
    )

    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=cfg.DATASETS.SEQ_LEN, sample=cfg.DATASETS.TEST_SAMPLE_METHOD, transform=transform_test,
                     max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME),
        batch_size=cfg.TEST.SEQS_PER_BATCH , shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=False
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=cfg.DATASETS.SEQ_LEN, sample=cfg.DATASETS.TEST_SAMPLE_METHOD, transform=transform_test,
                     max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME),
        batch_size=cfg.TEST.SEQS_PER_BATCH , shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=False,
    )

    if cfg.MODEL.SYN_BN:
        if use_gpu:
            model = nn.DataParallel(model)
        if cfg.SOLVER.FP_16:
            model = apex.parallel.convert_syncbn_model(model)
        model.cuda()


    start_time = time.time()
    xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids)
    tent = TripletLoss(cfg.SOLVER.MARGIN)

    optimizer = make_optimizer(cfg, model)

    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    # metrics = test(model, queryloader, galleryloader, cfg.TEST.TEMPORAL_POOL_METHOD, use_gpu)
    no_rise = 0
    best_rank1 = 0
    start_epoch = 0
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        # if no_rise == 10:
        #     break
        scheduler.step()
        print("noriase:", no_rise)
        print("==> Epoch {}/{}".format(epoch + 1, cfg.SOLVER.MAX_EPOCHS))
        print("current lr:", scheduler.get_lr()[0])

        train(model, trainloader, xent, tent, optimizer, use_gpu)
        if cfg.SOLVER.EVAL_PERIOD > 0 and ((epoch + 1) % cfg.SOLVER.EVAL_PERIOD == 0 or (epoch + 1) == cfg.SOLVER.MAX_EPOCHS):
            print("==> Test")

            metrics = test(model, queryloader, galleryloader, cfg.TEST.TEMPORAL_POOL_METHOD, use_gpu)
            rank1 = metrics[0]
            if rank1 > best_rank1:
                best_rank1 = rank1
                no_rise = 0
            else:
                no_rise += 1
                continue

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, osp.join(cfg.OUTPUT_DIR, "rank1_" + str(rank1) + '_checkpoint_ep' + str(epoch + 1) + '.pth'))
            # best_p = osp.join(cfg.OUTPUT_DIR, "rank1_" + str(rank1) + '_checkpoint_ep' + str(epoch + 1) + '.pth')

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, trainloader, xent, tent, optimizer, use_gpu):
    model.train()
    xent_losses = AverageMeter()
    tent_losses = AverageMeter()
    losses = AverageMeter()


    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        outputs, features = model(imgs)
        # combine hard triplet loss with cross entropy loss
        xent_loss = xent(outputs, pids)
        tent_loss, _, _ = tent(features, pids)
        xent_losses.update(xent_loss.item(), 1)
        tent_losses.update(tent_loss.item(), 1)
        loss = xent_loss + tent_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), 1)

        # attr_losses.update(attr_loss.item(), pids.size(0))
    print("Batch {}/{}\t Loss {:.6f} ({:.6f}) xent Loss {:.6f} ({:.6f}), tent Loss {:.6f} ({:.6f})".format(
        batch_idx + 1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg, tent_losses.val, tent_losses.avg))

    return losses.avg


def test(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):

    with torch.no_grad():
        model.eval()
        # ema.apply_shadow()
        qf, q_pids, q_camids = [], [], []
        query_pathes = []
        for batch_idx, (imgs, pids, camids, img_path) in enumerate(tqdm(queryloader)):
            query_pathes.append(img_path[0])
            if use_gpu:
                imgs = imgs.cuda()
            b, n, s, c, h, w = imgs.size()
            assert (b == 1)
            imgs = imgs.view(b * n, s, c, h, w)
            features = model(imgs)
            q_pids.extend(pids)
            q_camids.extend(camids)

            features = features.view(n, -1)

            features = torch.mean(features, 0)
            qf.append(features)
            del imgs
            del features
        qf = torch.stack(qf)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        np.save("query_pathes", query_pathes)


        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        gallery_pathes = []
        for batch_idx, (imgs, pids, camids, img_path) in enumerate(tqdm(galleryloader)):
            gallery_pathes.append(img_path[0])
            if use_gpu:
                imgs = imgs.cuda()
            b, n, s, c, h, w = imgs.size()
            imgs = imgs.view(b * n, s, c, h, w)
            assert (b == 1)
            features = model(imgs)

            features = features.view(n, -1)
            if pool == 'avg':
                features = torch.mean(features, 0)
            else:
                features, _ = torch.max(features, 0)
            g_pids.extend(pids)
            g_camids.extend(camids)
            gf.append(features)
            del imgs
            del features

        gf = torch.stack(gf)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        np.save("gallery_pathes", gallery_pathes)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("Computing distance matrix")

        if cfg.DATASETS.NAME == "duke":
            print("gallary with query result:")
            gf = torch.cat([gf, qf], 0)
            g_pids = np.concatenate([g_pids, q_pids], 0)
            g_camids = np.concatenate([g_camids, q_camids], 0)
            metrics = evaluate_reranking(qf, q_pids, q_camids, gf, g_pids, g_camids, ranks)
        else:
            metrics = evaluate_reranking(qf, q_pids, q_camids, gf, g_pids, g_camids, ranks)
        return metrics


if __name__ == '__main__':

    main()




