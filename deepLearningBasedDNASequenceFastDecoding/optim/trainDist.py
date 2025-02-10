import argparse
import datetime
import os
import sys
import torch
import dataset

import torch.utils.data.distributed
import torch.utils.data
import DeepLearningBasedDNASequenceFastDecoding.deepLearningBasedDNASequenceFastDecoding.optim.metrics as metrics
import DeepLearningBasedDNASequenceFastDecoding.deepLearningBasedDNASequenceFastDecoding.optim.distUtil as distUtil
import torch.distributed as dist

from os.path import join
from typing import TYPE_CHECKING
from DeepLearningBasedDNASequenceFastDecoding.deepLearningBasedDNASequenceFastDecoding.models import leopardPytorch, swinTransformer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

mets = {
    'loss':torch.nn.BCELoss(),
    'pr_auc':metrics.PR_AUC_sk,
    'iou':metrics.IOU,
    'dice':metrics.dice_coef
}

def log(*msg):
    if distUtil.is_master() or not args.dist:
        print(*msg)

def main(arg_overrides={},yeild_prauc = False):
    global args
    args = get_args()
    # override args
    for k,v in arg_overrides.items():
        setattr(args,k,v)

    is_master = not args.dist or distUtil.is_master()

    # log
    if is_master:
        log('setting up log...')
        exp_name = args.exp_name +'/'+ datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        log_dir = join(args.log_dir,exp_name)
        os.makedirs(join(log_dir,'tb'),exist_ok=True)
        os.makedirs(join(log_dir,'ckpt'),exist_ok=True)
        log('logging to',log_dir)
        tb_writer = SummaryWriter(log_dir=join(log_dir,'tb'))
        tb_writer.add_text('args',str(args))

    # redirect output to file
    if is_master:
        os.makedirs('hyperopt',exist_ok=True)
    #sys.stdout = open(join(log_dir,f'out{dist_util.get_rank()}.log'),'w')
    #sys.stderr = open(join(log_dir,f'err{dist_util.get_rank()}.log'),'w')

    # setup distributed training
    if args.dist:
        distUtil.setup_dist()
        print(f'Process start using device {distUtil.dev()} on machine {distUtil.get_hostname()}')
        device = distUtil.dev()
    else:
        device = torch.device('cuda:0')

    # data
    log(f'loading data from {join(args.data_dir,"train")}...')
    ds = dataset.DNADataset(join(args.data_dir,'train'))
    val_ds = dataset.DNADataset(join(args.data_dir,'val'))

    #ds.get_feature(3,32,range(0, 1),True)
    #val_ds.get_feature(3,32,range(0, 1),True)
    
    if args.dist:
        sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True)
        print(f'sampler_rank: {sampler.rank}')
        print('world_size: ', dist.get_world_size())
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle = False)

        dl = torch.utils.data.DataLoader(ds,args.batch_size//dist.get_world_size(), sampler = sampler)
        val_dl = torch.utils.data.DataLoader(val_ds,args.batch_size//dist.get_world_size(), sampler = val_sampler)
    else:
        dl = torch.utils.data.DataLoader(ds,args.batch_size, shuffle = True)
        val_dl = torch.utils.data.DataLoader(val_ds,args.batch_size, shuffle = False)

    print('dl_size: ', len(dl))
    for i, (x,y) in enumerate(dl):
        if i==0:
            print(x.shape)
        pass
    print('steps',i)


    # Prepare to train
    log('setting up model...')

    if args.model == 'swin':
        model = swinTransformer.SwinTransformer()
    elif args.model == 'leopard':
        model = leopardPytorch.LeopardUnet(1,ds.feature.shape[-1],dropout=args.dropout,num_blocks=args.num_blocks,initial_filter=args.initial_filter,size_kernel=args.kernel_size,scale_filter=args.scale_filter)
    else:
        raise ValueError('model not supported')

    if args.dist:
        # use sync bn
        model =  torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model = model.to(device)
    if is_master:
        log(sum(p.numel() for p in model.parameters() if p.requires_grad))
        tb_writer.add_text('model',str(model))

    log(torch.cuda.device_count())
    model.to(device).train()
    log(1.5)
    if args.dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[distUtil.get_local_rank()], output_device=distUtil.get_local_rank())
    log(2)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight,dtype=torch.float32).to(device))
    optim = torch.optim.Adam(model.parameters())
    log(3)
    # Trainging loop
    for epoch in range(args.num_epochs):
        log('epoch',epoch)
        model.train()
        if args.dist:
            sampler.set_epoch(epoch)
        training_loss_sum = 0
        step = 0
        for x, y in dl:
            #log('step',step)
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            loss = crit(prediction,y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            training_loss_sum += loss.detach()
            step += 1
        
        if args.dist:
            dist.all_reduce(training_loss_sum, op=dist.ReduceOp.SUM)
            training_loss_sum = training_loss_sum /dist.get_world_size()
        if is_master:
            tb_writer.add_scalar('train/loss', training_loss_sum/step, epoch)

        if not (epoch % args.val_freq):
            val_results = run_val(val_dl,model)
            
            # Get mean of metrics across all processes
            if args.dist:
                for name, val in val_results.items():
                    dist.all_reduce(val,op=dist.ReduceOp.SUM)
                    val_results[name] = val/dist.get_world_size()

            # Log metrics
            if is_master:
                for name, v in val_results.items():
                    tb_writer.add_scalar('val/'+name,v,epoch)
                log(f'epoch {epoch} val results: {val_results}')

            if yeild_prauc:
                yield val_results['pr_auc']

        if not (epoch % args.save_freq) and is_master:
            path = join(log_dir,'ckpt',f'{epoch}.pt')
            log('saving ckpt to',path)
            torch.save(model.module.state_dict(),path)

        if args.dist:
            dist.barrier()
            

def run_val(dl,model):
    log('running metrics')
    model.eval()
    preds = []
    ys = []
    for x, y in dl:
        with torch.no_grad():
            x, y = x.to(distUtil.dev()), y.to(distUtil.dev())
            prediction = torch.sigmoid(model(x))
            ys.append(y)
            preds.append(prediction)
    ys = torch.cat(ys,0).flatten().cpu()
    preds = torch.cat(preds,0).flatten().cpu()

    results = {}
    for name, m in mets.items():
        log(f'evaluating {name}...')
        results[name] = m(preds,ys).to(distUtil.dev())
    return results
    


def get_args():
    parser = argparse.ArgumentParser(description="DeepTF")
    parser.add_argument('-tf', '--transcription_factor', default='CTCF', type=str, nargs='+',
                        help='transcript factor')
    parser.add_argument('-m', '--models', default='cnn', type=str,
                        help='model architecture')
    parser.add_argument('-l', '--input_length', default=10240, type=int,
                        help='length of input sequence')
    parser.add_argument('-v', '--vocab_size', default=5, type=int,
                        help='vocabulary size of the input')
    parser.add_argument('-n', '--number_of_samples', default=10000, type=int,
                        help='number of samples in each draw (default draw 10000 examples)')
    parser.add_argument('-d', '--draw_frequency', default=1, type=int,
                        help='draw frequency (default draw 1 time)')
    parser.add_argument('-r', '--random_seed', default=1, type=int,
                        help='fix random seed')
    parser.add_argument('-f', '--path', default='/home/nckuhpclab07/eri24816/2022-APAC-HPC-AI/', type=str,
                        help='save data to path')
    parser.add_argument('--dist', type=bool, default=True,
                        help='use torch DDP to perform distributed training')
    parser.add_argument('--data_dir',  type=str,
                        help='data directory')
    parser.add_argument('--val_freq', default=5,  type=int, 
                        help='num of epochs to evaluate val metrics')
    parser.add_argument('--save_freq', default=5,  type=int,
                        help='num of epochs to save model')
    parser.add_argument('--num_epochs', default=1000,  type=int,
                        help='num of epochs to train')
    parser.add_argument('--exp_name', default='',  type=str,
                        help='experiment name')
    parser.add_argument('-b', '--batch_size', default=256, type=int,help='number of samples in each batch')
    parser.add_argument('--log_dir', default='log', type=str)

    parser.add_argument('--dropout', default=0,  type=float, help='dropout rate')
    parser.add_argument('--pos_weight', default=1,  type=float, help='bce positive weight')    
    parser.add_argument('--num_blocks', default=5, type=int,help='number of unet blocks')
    parser.add_argument('--initial_filter', default=15, type=int,help='filters in first unet block')
    parser.add_argument('--kernel_size', default=7, type=int,help='')
    parser.add_argument('--scale_filter', default=1.5, type=float,help='')

    parser.add_argument('--model', default='leopard', type=str,help='leopard/swin')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    for _ in main():
        pass