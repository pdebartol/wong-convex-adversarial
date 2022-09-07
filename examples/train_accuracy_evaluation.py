import problems as pblm
import torch
import numpy as np
import pdb
from autoattack import AutoAttack
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
from tqdm import tqdm
import pandas as pd

class ModelNormWrapper(torch.nn.Module):
    def __init__(self, model, means, stds):
        super(ModelNormWrapper, self).__init__()
        self.model = model
        self.means = torch.Tensor(means).float().view(3, 1, 1).cuda()
        self.stds = torch.Tensor(stds).float().view(3, 1, 1).cuda()

    def forward(self, x):
        x = (x - self.means) / self.stds
        return self.model.forward(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset arguments
    parser.add_argument("--batch_size", type = int, default = 4096)
    parser.add_argument("--seed", type = int, default = 64)
    parser.add_argument("--mode", type = str, default = "madry")
    parser.add_argument("--model", type = str, default = "small")
    parser.add_argument("--dataset", type = str, default = "mnist")
    parser.add_argument("--norm", type = str, default = "L2")



    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.dataset == 'mnist':
        if args.norm == "L2":
            budgets=[0.25, 0.5, 0.75,1.0,1.25,1.5,1.75,2.0]
        else:
            budgets = [0.1 ,0.2, 0.3 ,0.4]
        _, test_loader = pblm.mnist_loaders(args.batch_size)
        if args.model =='small':
            model = pblm.mnist_model().cuda()
        elif args.model == 'large':
            model = pblm.mnist_model_large().cuda()

    elif args.dataset == 'cifar':
        if args.norm == 'L2':
            budgets = [36/255]
        else:
            budgets = [2/255,8/255]
        train_loader, _ = pblm.cifar_loaders(args.batch_size)
        if args.model =='small':
            model = pblm.cifar_model().cuda()
        elif args.model == 'large':
            model = pblm.cifar_model_large().cuda()





    if args.mode == 'wong':
        df = {'eps':[], 'robust_error':[], 'clean_error':[]}
        for eps in budgets:
            ckpt = torch.load(f'/cluster/home/pdebartol/convex_adversarial/examples/{args.dataset}_{args.norm}/wong_{args.model}_eps_{eps}.pth')['state_dict'][0]
            model.load_state_dict(ckpt)
            model.eval()
            if args.dataset == 'cifar':
                model_norm = ModelNormWrapper(model, means=[0.485, 0.456, 0.406],
                                     stds=[0.225, 0.225, 0.225])
            n = 0
            correct = 0
            for x,y in train_loader:
                correct+= model_norm(x.cuda()).argmax(1).eq(y.cuda()).sum().detach()
                n += y.shape[0]
            print(f'mode: {args.mode}, eps: {eps}, train accuracy: {correct/n}')
         


    elif args.mode == 'madry':
        df = {'eps':[], 'robust_error':[], 'clean_error':[]}
        if args.dataset == 'cifar':
                model = ModelNormWrapper(model,  stds = [0.2023, 0.1994, 0.2010],means = [0.4914, 0.4822, 0.4465])
        for eps in budgets:
            ckpt = torch.load(f'/cluster/home/pdebartol/convex_adversarial/examples/{args.dataset}_{args.norm}/madry_{args.model}_eps_{eps}.pth')['net']
            model.load_state_dict(ckpt)
            model.eval()
            n = 0
            correct = 0
            for x,y in train_loader:
                correct+= model(x.cuda()).argmax(1).eq(y.cuda()).sum().detach()
                n += y.shape[0]
            print(f'mode: {args.mode}, eps: {eps}, train accuracy: {correct/n}')

    elif args.mode == 'zhang':
        df = {'eps':[], 'robust_error':[], 'clean_error':[]}
        for eps in budgets:
            ckpt = torch.load(f'/cluster/home/pdebartol/convex_adversarial/examples/{args.dataset}_linf/zhang_{args.model}_eps_{eps}.pth')['state_dict']
            if args.dataset == 'cifar':
                eps /= 255
            model.load_state_dict(ckpt)
            model.eval()
            if args.dataset=='cifar':
                model_norm = ModelNormWrapper(model, stds = [0.2023, 0.1994, 0.2010],means = [0.4914, 0.4822, 0.4465])
            
            n=0
            correct=0
            for x,y in train_loader:
                correct += model_norm(x.cuda()).argmax(1).eq(y.cuda()).sum().detach()
                n += y.shape[0]
            print(f'mode: {args.mode}, eps: {eps}, train accuracy: {correct/n}')



    
  
