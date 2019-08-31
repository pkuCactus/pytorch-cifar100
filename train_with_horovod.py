# use the ResNet of torchvision which conv1 is 7x7/s2 following 2x2/s2 max pooling, can only achieve ~60 top1 accuracy(what is the reason)
# use the resnet which provided by the author, ie, 3x3/s1 conv1  without following pooling, coudl achieve even 78.9 top1 accuracy.
from __future__ import print_function, division
# import moxing as mox
import os
import sys
import time
import torch
import torch.nn as nn
import torch.utils.data
from torch.optim import SGD
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed as distributed
import torchvision
from torch.optim.lr_scheduler import _LRScheduler
import logging
import argparse
import math
sys.path.append('cache')
from models.resnet import resnet101

args = argparse.Namespace(warm=1, lr=0.1)
args.epochs = 200
args.MILESTONES = [60, 120, 160, 200]

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

use_hvd = True
if use_hvd:
    import horovod.torch as hvd
    from horovod.torch.mpi_ops import allreduce_async
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def create_logger(path):
    # set up logger
    if not os.path.exists(path):
        os.makedirs(path)

    log_file = 'log_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s | %(filename)-10s | line %(lineno)-3d: %(message)s'
    logging.basicConfig(filename=os.path.join(path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(head))
    logger.addHandler(console)

    return logger

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ResNetCifar(nn.Module):
    def __init__(self, num_classes, pretrained=None):
        super(ResNetCifar, self).__init__()
        self.num_classes = num_classes
        self.backbone = torchvision.models.resnet101(num_classes=100)
#         if pretrained:
#             self.backbone.load_state_dict(torch.load(pretrained))
#         self.backbone.fc = nn.Linear(2048, self.num_classes)
#         nn.init.kaiming_normal_(self.backbone.fc.weight)
#         nn.init.constant_(self.backbone.fc.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        return x

is_master = not use_hvd or (hvd.rank() == 0)

batch_size = 128
base_batch_size = 128
    
def main():
    logger = create_logger('logs')
    
    normalize = torchvision.transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
                                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                torchvision.transforms.RandomHorizontalFlip(),
                                torchvision.transforms.RandomRotation(15),
                                torchvision.transforms.ToTensor(),
                                normalize])
    
    train_dataset = torchvision.datasets.CIFAR100(root='./', transform=transform)
    if use_hvd:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=12)
    iter_per_epoch = len(train_loader)
    logger.info('Iters per epcoh: %d'%iter_per_epoch)
    
    # net = ResNetCifar(100, pretrained='resnet101-5d3b4d8f.pth')
    net = resnet101()
    # logger.info(net)
    net = net.cuda()
    
    scale = batch_size / base_batch_size
    if use_hvd:
        scale *= hvd.size()
    logger.info('Scale: {}'.format(scale))
    
    MILESTONES = [math.ceil(x/scale) for x in args.MILESTONES]
    logger.info(MILESTONES)
    
    epochs = math.ceil(args.epochs / scale)
    logger.info('training {} epochs'.format(epochs))
    lr = args.lr * scale
    optimizer = SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    if use_hvd:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.named_parameters())
    
        hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2)

    criterion = nn.CrossEntropyLoss().cuda()
    losses = AverageMeter()
    top1 = AverageMeter()

    for epoch in range(1, epochs):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        if is_master:
            val(net, logger)
        net.train()
        for idx, (data, label) in enumerate(train_loader):
            if epoch <= args.warm:
                warmup_scheduler.step()
            data, label = data.cuda(), label.cuda()
            
            optimizer.zero_grad()
            out = net(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            prec = accuracy(out, label)[0]
            losses.update(loss.item(), data.size(0))
            top1.update(prec.item(), data.size(0))

            if idx % 20 == 0:
                logger.info(
                    'rank: {}, epoch {}/{}, iterations {}/{}, lr: {:.6f}, Loss: {loss.val:.4f}({loss.avg:.4f}), '
                    'Prec: {top1.val:.4f}%({top1.avg:.4f}%)'.format(hvd.rank() if use_hvd else 0, 
                                                    epoch, epochs, idx, len(train_loader),
                                                    optimizer.param_groups[0]['lr'], loss=losses, top1=top1))
        if is_master and epoch % 10 == 0:
            torch.save(net.state_dict(), 'models/{:04}_{:04}.pth'.format(epoch, epochs))
            
def val(net, logger=logging):
    normalize = torchvision.transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
                                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                normalize])

    test_dataset = torchvision.datasets.CIFAR100(root='./', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=12)
    net.eval()
    net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    prec = 0
    test_loss = 0
    for idx, (data, label) in enumerate(test_loader):
        data, label = data.cuda(), label.cuda()
        with torch.no_grad():
            out = net(data)
        loss = criterion(out, label)
        acc = accuracy(out, label)[0]
        test_loss += loss.item()
        _, preds = out.max(1)
        prec += preds.eq(label).sum()
    logger.info('\n')
    logger.info('Loss: {:.4f}, Prec: {:.4f}%'.format(test_loss*1./len(test_dataset), prec.item()*100./len(test_dataset)))
    logger.info('\n')
        
        
if __name__ == '__main__':
    main()
