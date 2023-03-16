import time
import torch
import os
from tqdm import tqdm as tqdm
import time
import random

from model.CANNet import CANNet
from model.CrowdDatase import CrowdDataset
from utils.distributed_utils import init_distributed_mode, dist, cleanup
from utils.train_eval_utils import train_one_epoch, evaluate

import argparse

import tempfile
import math
import torch.optim.lr_scheduler as lr_scheduler

import wandb


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    init_distributed_mode(args=args)

    rank = args.rank
    init_checkpoint = args.init_checkpoint
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    temp_init_checkpoint_path = "checkpoints"
    use_wandb = args.wandb
    show_images = args.show
    
    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print("train start!", time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        print(args)

        if os.path.exists(temp_init_checkpoint_path) is False:
            os.makedirs(temp_init_checkpoint_path)
        if use_wandb:    
            wandb.init(project="VisDrone", group="CAN", mode="offline")

    # DataPath Shanghai_part_A
    # train_image_root = args.data_root + 'train_data/images'
    # train_dmap_root = args.data_root + 'train_data/ground_truth'
    # test_image_root = args.data_root + '.test_data/images'
    # test_dmap_root = args.data_root + '.test_data/ground_truth'

    train_image_root = 'data/VisDrone/images/train'
    train_dmap_root = 'data/Density-VisDrone/DMNdata/train/dens'
    test_image_root = 'data/VisDrone/images/val'
    test_dmap_root = 'data/Density-VisDrone/DMNdata/val/dens'

    # configuration
    gpu_or_cpu = args.device  # use cuda or cpu
    lr = args.lr
    batch_size = args.batch_size
    momentum = 0.95
    epochs = args.epochs
    num_workers = 1
    seed = time.time()
    
    # ======================== cuda ====================================
    
    device = torch.device(gpu_or_cpu)
    torch.cuda.manual_seed(seed)
    
    # ==================================== dataloader ==================
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
                                               batch_sampler=train_batch_sampler)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              sampler=test_sampler,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              shuffle=False)
    
    # ========================================= model ===========================
    model = CANNet().to(device)

    if os.path.exists(init_checkpoint):
        if rank == 0:
            print('load checkpoint from {}'.format(init_checkpoint))
        weights_dict = torch.load(init_checkpoint, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        temp_init_checkpoint_path = os.path.join(
            tempfile.gettempdir(), "initial_weights.pth")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), temp_init_checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(
            temp_init_checkpoint_path, map_location=device))

    if args.syncBN:
        # 使用SyncBatchNorm后训练会更耗时
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])

    # ===================================== optimizer ===========================================
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr,
                                momentum=momentum,
                                weight_decay=0)

    def lf(x): return ((1 + math.cos(x * math.pi / args.epochs)) / 2) * \
        (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # optimizer=torch.optim.Adam(model.parameters(),lr)

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
            if use_wandb: 
                wandb.log({'loss': mean_loss})
                wandb.log({'mae': mean_mae})
                wandb.log({'lr': optimizer.param_groups[0]["lr"]})
            
            # show an image
            if show_images:
                images = []
                index=random.randint(0,len(test_loader)-1)
                img, gt_dmap=test_dataset[index]

                images.append(wandb.Image(img, caption=f"image {epoch}"))
                images.append(wandb.Image(gt_dmap/(gt_dmap.max())*255, caption=f"gt_density {epoch}"))

                img=img.unsqueeze(0).to(device)
                gt_dmap=gt_dmap.unsqueeze(0)
                et_dmap=model(img)
                et_dmap=et_dmap.squeeze(0).detach().cpu().numpy()
                images.append(wandb.Image(et_dmap/(et_dmap.max())*255, caption=f"gt_density {epoch}"))
                wandb.log({"examples" : [wandb.Image(im) for im in images]})
                
    if rank == 0:
        print("train done!", time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
   
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-7)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--syncBN', type=bool, default=True)
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--show', type=bool, default=True)
    parser.add_argument('--data_root', type=str,
                        default="./data/Shanghai_part_A/")
    parser.add_argument('--init_checkpoint', type=str, default='./checkpoints/cvpr2019_CAN_SHHA_353.pth',
                        help='initial weights path')
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda',
                        help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    args = parser.parse_args()

    main(args)