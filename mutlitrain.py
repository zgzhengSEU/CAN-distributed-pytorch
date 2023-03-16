import numpy as np
import time
import torch
import torch.nn as nn
import os
import random
from tqdm import tqdm as tqdm

from cannet import CANNet
from my_dataset import CrowdDataset
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate

import argparse
from torch.utils.tensorboard import SummaryWriter
import tempfile
import math
import torch.optim.lr_scheduler as lr_scheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-7)
    parser.add_argument('--lrf', type=float, default=0.1)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/home/wz/data_set/flower_data/flower_photos")

    # resnet34 官方权重下载地址
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='cvpr2019_CAN_SHHA_353.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda',
                        help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    args = parser.parse_args()

    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    init_distributed_mode(args=args)

    rank = args.rank
    # device = torch.device(args.device)
    # batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    # configuration
    train_image_root = 'data/VisDrone/images/train'
    train_dmap_root = '/home/zzg/zgzheng/DMNet/train/dens'
    test_image_root = 'data/VisDrone/images/val'
    test_dmap_root = '/home/zzg/zgzheng/DMNet/val/dens'
    gpu_or_cpu = args.device  # use cuda or cpu
    lr = args.lr
    batch_size = args.batch_size
    momentum = 0.95
    epochs = args.epochs
    steps = [-1, 1, 100, 150]
    scales = [1, 1, 1, 1]
    num_workers = 4
    seed = time.time()
    print_freq = 30

    # ==================================== dataload ==================
    train_dataset = CrowdDataset(
        train_image_root, train_dmap_root, gt_downsample=8, phase='train')
    test_dataset = CrowdDataset(
        test_image_root, test_dmap_root, gt_downsample=8, phase='test')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset)

    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=False)

    if rank == 0:
        print('Using {} dataloader workers every process'.format(num_workers))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=False,
                                               num_workers=num_workers,
                                               batch_size=1,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              sampler=test_sampler,
                                              pin_memory=False,
                                              num_workers=num_workers,
                                              batch_size=1,
                                              shuffle=False)
    # ============================================================================

    device = torch.device(gpu_or_cpu)
    torch.cuda.manual_seed(seed)
    criterion = nn.MSELoss(size_average=False).to(device)

    # ========================================= model ===========================
    model = CANNet().to(device)

    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(
            tempfile.gettempdir(), "initial_weights.pth")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if args.syncBN:
        # 使用SyncBatchNorm后训练会更耗时
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    # ========================================================================================
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr,
                                momentum=momentum,
                                weight_decay=0)

    def lf(x): return ((1 + math.cos(x * math.pi / args.epochs)) / 2) * \
        (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # optimizer=torch.optim.Adam(model.parameters(),lr)

    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    # ========================================= train and eval ============================================
    min_mae = 10000
    min_epoch = 0
    for epoch in range(0, epochs):
        train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epoch=epoch
        )

        scheduler.step()

        # testing phase
        mae_sum = evaluate(
            model=model,
            test_loader=test_loader,
            device=device,
            epoch=epoch
        )

        if rank == 0:
            mean_mae = mae_sum / len(test_loader)
            if mean_mae < min_mae:
                min_mae = mean_mae
                min_epoch = epoch
                torch.save(model.state_dict(),
                           './checkpoints/epoch_{}.pth'.format(epoch))
            
            print("[epoch {}] mae: {}, min_mae: {}, min_epoch: {}".format(
                epoch, mean_mae, min_mae, min_epoch))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], mean_mae, epoch)
            tb_writer.add_scalar(
                tags[2], optimizer.param_groups[0]["lr"], epoch)

        # show an image
        # index = random.randint(0, len(test_loader)-1)
        # img, gt_dmap = test_dataset[index]
        # img = img.unsqueeze(0).to(device)
        # gt_dmap = gt_dmap.unsqueeze(0)
        # et_dmap = model(img)
        # et_dmap = et_dmap.squeeze(0).detach().cpu().numpy()

    import time
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
