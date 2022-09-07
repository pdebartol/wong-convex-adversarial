import problems as pblm
import argparse
import torch
from torch import nn
from tqdm import tqdm
from autoattack.autopgd_base import APGDAttack
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import multiprocessing
from wide_resnet_imagenet64 import  wide_resnet_imagenet64
from wide_resnet_cifar import wide_resnet_cifar
import wandb
import subprocess
import os
import shutil

class ModelNormWrapper(torch.nn.Module):
    def __init__(self, model, means, stds):
        super(ModelNormWrapper, self).__init__()
        self.model = model
        self.means = torch.Tensor(means).float().view(3, 1, 1).cuda()
        self.stds = torch.Tensor(stds).float().view(3, 1, 1).cuda()

    def forward(self, x):
        x = (x - self.means) / self.stds
        return self.model.forward(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(epoch, model, criterion, optimizer,  dataloader, std_training=0):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    correct_adv= 0
    total = 0
    apgd = APGDAttack(model, n_restarts=1, n_iter=10, verbose=False,
                eps=args.epsilon, norm=args.norm, eot_iter=1, rho=.75, seed=args.seed)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device).long()
        inputs_adv = apgd.perturb(inputs, targets) 
        optimizer.zero_grad()
        outputs_adv = model(inputs_adv)
        outputs = model(inputs)
        loss = criterion(outputs_adv, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        correct_adv  += outputs_adv.argmax(1).eq(targets).sum().item()
    print(f'epoch:{epoch} , train_acc:{100.*correct/total} , train_robust_acc:{100.*correct_adv/total}')
   # wandb.log({"epoch": epoch, "train_accuracy": 100.*correct/total,
     #"train_robust_accuracy": 100.*correct_adv/total })

# Testing clean accuracy
def test_pgd(epoch, model, dataloader):
    model.eval()
    correct = 0
    total = 0
    apgd = APGDAttack(model, n_restarts=3, n_iter=10, verbose=False,
                eps=args.epsilon, norm=args.norm, eot_iter=1, rho=.75, seed=args.seed)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_adv = apgd.perturb(inputs, targets) 
            outputs   = model(inputs_adv)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        #wandb.log({"epoch":epoch, "test_clean_accuracy":100.*correct/total})
    # Save checkpoint.
    acc = 100.*correct/total
    return acc

# Testing clean accuracy
def test_clean(epoch, model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs   = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        #wandb.log({"epoch":epoch, "test_clean_accuracy":100.*correct/total})
    # Save checkpoint.
    acc = 100.*correct/total
    return acc

if __name__ == "__main__": 
    # Parse arguments
    parser = argparse.ArgumentParser(description=' PGD Training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=0, type=float, help='weight decay')
    parser.add_argument('--epochs', default=150, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--epsilon', default=1.0, type=float, help='random seed')
    parser.add_argument('--std_training', default=0, type=int, help='random seed')
    parser.add_argument('--model', default="small", type=str, help='random seed')
    parser.add_argument('--dataset', default="mnist", type=str, help='random seed')
    parser.add_argument('--norm', default="L2", type=str, help='random seed')
    parser.add_argument('--momentum', default=0.9, type=float, help='random seed')
    parser.add_argument('--weight_decay', default=0, type=float, help='random seed')
    parser.add_argument('--optim', default="adam", type=str, help='random seed')

    args = parser.parse_args()

    if args.dataset =='mnist':
        if args.model == 'small':   
            model = pblm.mnist_model().cuda()
        elif args.model =='large':
            model = pblm.mnist_model_large().cuda()
        train_loader, test_loader = pblm.mnist_loaders(args.batch_size)
        scheduler_milestones = [40,80]

    elif args.dataset == 'cifar':
        if args.model == 'small':   
            model = pblm.cifar_model().cuda()
            scheduler_milestones = [75,90]
        elif args.model == 'large':
            model = pblm.cifar_model_large().cuda()
            scheduler_milestones = [75,90]
        elif args.model =='resnet':
            model = pblm.cifar_model_resnet(N=1,factor=1).cuda()
            scheduler_milestones = [75,90]
        elif args.model=='wideresnet':
            model = wide_resnet_cifar().cuda()
            model = torch.nn.DataParallel(model)
            scheduler_milestones = [100,150]
        
        trainset = datasets.CIFAR10('./data', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
        ]))
        testset = datasets.CIFAR10('./data', train=False, 
            transform=transforms.Compose([transforms.ToTensor()]))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
            shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, pin_memory=True)
        model = ModelNormWrapper(model, stds = [0.2023, 0.1994, 0.2010],means = [0.4914, 0.4822, 0.4465])
        

    elif args.dataset == 'tiny_imagenet':
        tmpdir = os.getenv('TMPDIR')
        shutil.copy2('tinyimagenet_download.sh', tmpdir)
        os.chdir(tmpdir)
        rc = subprocess.call(tmpdir +  '/tinyimagenet_download.sh')
        train_data = datasets.ImageFolder( tmpdir + '/tiny-imagenet-200' + '/train',
                                        transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(56, padding_mode='edge'),
                                            transforms.ToTensor()
                                        ]))
        test_data = datasets.ImageFolder(tmpdir + '/tiny-imagenet-200' + '/val',
                                        transform=transforms.Compose([
                                            transforms.CenterCrop(56),
                                            transforms.ToTensor()]))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                             num_workers=min(multiprocessing.cpu_count(), 4))
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size // 5, pin_memory=True,
                                            num_workers=min(multiprocessing.cpu_count(), 4))

        model = wide_resnet_imagenet64(in_planes=16,widen_factor=10).cuda()
        model = torch.nn.DataParallel(model)
        model = ModelNormWrapper(model, stds = [0.2302, 0.2265, 0.2262],means =[0.4802, 0.4481, 0.3975])
        scheduler_milestones = [100,150]



    criterion = nn.CrossEntropyLoss().to(device)
    if args.optim =='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim =='sgd':
         optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler =  MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=0.1)
    
    #wandb.init(project="image_experiments"  ,entity="sml-eth",
     #  group=args.dataset+'_'+args.model+'_madry',
      #   settings=wandb.Settings(start_method="fork"))
   # wandb.config.update(args) 

    best_acc = 0.
    for epoch in tqdm(range(args.epochs)):
        train(epoch, model, criterion, optimizer,  train_loader, args.std_training)
        acc = test_clean(epoch, model, test_loader)
        print(f'epoch: {epoch}, test_acc:{acc}')
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if args.std_training == 0:      
                torch.save(state, f'/cluster/work/yang/pier/{args.dataset}_{args.norm}/madry_{args.model}_eps_{args.epsilon}.pth')
            else:
                torch.save(state, f'/cluster/work/yang/pier/{args.dataset}_{args.norm}/std_{args.model}_eps_{args.epsilon}.pth')

            best_acc = acc
        scheduler.step()


