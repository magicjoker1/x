
import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader
from data.dataset import NoisyDataset, PSF_RealDataset,  AberrationStrehl, AberrationDataset
from models.resnet import *
from utils import *
from tqdm import trange
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from my_parser import parse_args_train
import numpy as np
import os
from collections import OrderedDict
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision import utils
from matplotlib.lines import Line2D
import sys
import pandas as pd
import time


def _log_stats(writer,num_iter,phase, loss_avg):
    tag_value = {
        f'{phase}_loss_avg': loss_avg,
    }

    for tag, value in tag_value.items():
        writer.add_scalar(tag, value, num_iter)


def train(model, data_loader, criterion, optimizer, scaler, scheduler, epoch_start=0,num_epochs=5, saver=None, output_dir='',save_every=10):
# 新增scheduler
    train_metrics = dict()
    val_metrics = dict()
    test_metrics = dict()
    best_metric = None
    best_epoch = None
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))


    # 新增：初始化早停器
    early_stopper = EarlyStopping(patience=10, min_delta=1e-5)
    for epoch in trange(epoch_start,num_epochs+epoch_start, desc="Epochs"):
        result = []
        # 记录当前 epoch 的验证损失
        current_val_loss = None

        for phase in ['train', 'val','test']:
            if phase == "train":  # training mode
                model.train()
            else:     #  validation or test mode
                model.eval()

            # keep track of  loss
            running_loss = 0.0

            for data, target in data_loader[phase]:
                # load the data and target to respective device
                data, target = data.cuda(), target.cuda()

                with torch.set_grad_enabled(phase == "train"):

                    output = model(data)
                    loss = criterion(output, target)

                    if phase == "train":
                        scaler.scale(loss).backward()

                        scaler.step(optimizer)
                        scaler.update()
                         # zero the grad to stop it from accumulating
                        optimizer.zero_grad()

                running_loss += loss.item() * data.size(0)

            epoch_loss = running_loss / len(data_loader[phase].sampler)
            print(f"Epoch {epoch + 1}/{num_epochs + epoch_start}, {phase}_loss: {epoch_loss:.6f}")
            # monitor learning rate
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            _log_stats(writer,epoch,phase, epoch_loss)

            result.append('{} LR: {:.4f} Loss: {:.4f} '.format(phase, lr, epoch_loss))

            if phase == "train":
                train_metrics = OrderedDict([('loss', epoch_loss)])
            elif phase == "test":
                test_metrics = OrderedDict([('loss', epoch_loss)])
            else:
                val_metrics = OrderedDict([('loss', epoch_loss)])
                if saver is not None:
                    # save proper checkpoint with val metric
                    best_metric, best_epoch = saver.save_checkpoint(epoch, metric=epoch_loss)
                if best_metric is not None:
                    print('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
                if epoch != 0 and epoch % save_every == 0:
                    saver.save(epoch, metric=epoch_loss)

                # 新增：早停检查（基于验证集loss）
        # early_stopper(epoch_loss)
        if current_val_loss is not None:
            early_stopper(current_val_loss)
            if early_stopper.early_stop:
                print(f"早停触发于 epoch {epoch}")
                return test_metrics


        update_summary(epoch, train_metrics, val_metrics,test_metrics,filename = os.path.join(
            output_dir, 'summary.csv'), write_header=(epoch == 0))
        # 新增：每个epoch结束后更新学习率
        scheduler.step()

    return test_metrics

def infer_subset(model, data_loader, criterion, optimizer, output_dir='',num_zernike=24,gen_aberration=None):
    ideal_abb,_ = gen_aberration.gen(C=torch.tensor([0]*num_zernike))
    ideal_peak = torch.max(ideal_abb)

    # plot a subset of the test set
    test_metrics = dict()
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))

    targets = np.array([]).reshape(0,num_zernike)
    outputs = np.array([]).reshape(0,num_zernike)

    for epoch in trange(1, desc="Epochs"):
        result = []
        for phase in ['test']:
            model.eval()
            running_loss = 0.0
            strehl = 0

            for data, target in data_loader[phase]:	
                data, target = data.cuda(), target.cuda()

                with torch.set_grad_enabled(phase == "train"):
                    output = model(data)
                    loss = criterion(output, target)
                targets = np.vstack((targets,target.cpu().detach().to(torch.float).numpy()))
                outputs = np.vstack((outputs,output.cpu().detach().to(torch.float).numpy()))

                corrected_c = target[0]-output[0]

                #strehl ratio calculation
                corrected,_ = gen_aberration.gen(C=corrected_c)
                strehl_r = torch.max(corrected)/ideal_peak
                strehl += strehl_r

                running_loss += loss.item() * data.size(0)

            strehl_avg = strehl/len(data_loader[phase].sampler)

            epoch_loss = np.sqrt(running_loss / len(data_loader[phase].sampler))
     
            # monitor learning rate
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            _log_stats(writer,epoch,phase, epoch_loss)
            rmsw = np.sqrt(((outputs - targets) ** 2).mean(0).sum())
            result.append('{} LR: {:.4f} Loss: {:.4f}  RMSW: {:.4f} '.format(phase, lr, epoch_loss,rmsw))
            test_metrics = OrderedDict([('loss', epoch_loss),('strehl',strehl_avg),('rmsw',rmsw)])

        update_summary(epoch, dict(), dict(),test_metrics,filename = os.path.join(
            output_dir, 'summary.csv'), write_header=(epoch == 0))

   
    return test_metrics


def main():

    args, args_text = parse_args_train()
    print("当前 data_path =", args.data_path)  # 打印读取到的 data_path
    print("当前 resume_path =", args.resume_path)  # 可选，验证模型路径
    torch.multiprocessing.set_start_method('spawn')
    args.num_zernike = len(args.zernike)

    #seed
    os.environ['PYTHONHASHSEED']=str(args.seed)
    seed_worker(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)  

    # output directory
    exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = get_outdir('./output/'+args.folder_name, exp_name)
    print(output_dir)

    # model
    model = ResNet50(args.num_zernike,channel=len(args.channel)).cuda()
    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 新增学习率调度器，每20个epoch学习率减半
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 新增
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    precision = torch.float
    # loss
    criterion = nn.MSELoss().cuda()

    epoch_start = 0
    # resume training
    if args.resume_path:
        if os.path.isfile(args.resume_path):
            print("=> loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint["scaler"])
            print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume_path, checkpoint['epoch']))
            epoch_start = checkpoint['epoch']

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 新增：同步更新调度器

    # data size for psf data
    indices = list(range(args.data_size))
    val_split = int(np.floor(args.val_size * args.data_size))
    test_split = int(np.floor(args.test_size * args.data_size))

    ds=None
    if args.dataset=="psf":
        ds = AberrationDataset(dataset_size=args.data_size,num_zernike=args.num_zernike,precision=precision,bias_z=args.bias_z,val_test_size = val_split+test_split, zernike=args.zernike,bias_val=args.bias_val,npts=args.npts)
    elif args.dataset=="psfNoisy":
        ds = NoisyDataset(dataset_size=args.data_size,num_zernike=args.num_zernike,precision=precision,bias_z=args.bias_z,val_test_size = val_split+test_split, zernike=args.zernike,bias_val=args.bias_val,npts=args.npts,z_range=args.z_range)
    elif args.dataset=="realPsf":
        ds = PSF_RealDataset(data_path=args.data_path[0], channel=args.channel)
        args.data_size = len(ds)
        indices = list(range(args.data_size))
        val_split = int(np.floor(args.val_size * args.data_size))
        test_split = int(np.floor(args.test_size * args.data_size))

    train_idx, val_idx, test_idx = indices[test_split+val_split:], indices[:val_split], indices[val_split:test_split+val_split]
    
    # training size
    if args.dataset=="realpsf" and args.train_size<1.0:
        remove_idx = round(len(train_idx)*(1-args.train_size))
        train_idx = train_idx[:-remove_idx]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    #seed before data load
    seed_worker(args.seed)
    kwargs = {'num_workers': args.threads, 'pin_memory': False,'worker_init_fn':seed_worker, 'generator':g}
    dataloader = {"train": DataLoader(ds, batch_size=args.train_batch,sampler=train_sampler,**kwargs),
                "val": DataLoader(ds, batch_size= args.valid_batch,sampler=val_sampler,**kwargs),
                "test": DataLoader(ds, batch_size= 1,sampler=test_sampler,**kwargs)}
    print(f"trainloader len {len(dataloader['train'])} valloader len {len(dataloader['val'])} testloader len {len(dataloader['test'])}")

    # save training config
    saver = CheckpointSaver(model=model, optimizer=optimizer, scaler=scaler, args=args,
        checkpoint_dir=output_dir,  decreasing=True, max_history=args.checkpoint_hist)
    
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)
    if args.infer:
        gen_aberrationStr = AberrationStrehl(args.image_size,device='cuda',precision=precision,bias_z=args.bias_z, zernike=args.zernike,bias_val=[0],npts=args.npts)
        infer_subset(model,dataloader , criterion, optimizer, output_dir=output_dir,num_zernike=args.num_zernike,gen_aberration=gen_aberrationStr)
    else:
        train(model,dataloader , criterion, optimizer, scaler, scheduler=scheduler, epoch_start=epoch_start,num_epochs=args.epochs,saver=saver,output_dir=output_dir,save_every=args.save_every)


if __name__ == '__main__':
    main()
