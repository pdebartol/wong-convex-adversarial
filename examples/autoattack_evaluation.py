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
from wide_resnet_cifar import wide_resnet_cifar
import os, shutil, subprocess, multiprocessing
from wide_resnet_imagenet64 import  wide_resnet_imagenet64

class ModelNormWrapper(torch.nn.Module):
    def __init__(self, model, means, stds):
        super(ModelNormWrapper, self).__init__()
        self.model = model
        self.means = torch.Tensor(means).float().view(3, 1, 1).cuda()
        self.stds = torch.Tensor(stds).float().view(3, 1, 1).cuda()

    def forward(self, x):
        x = (x - self.means) / self.stds
        return self.model.forward(x)

def error_per_class(y_test,y_pred):
    class_err = []
    for c in y_test.unique():
        mask = y_test==c
        class_err.append( (1-y_pred[mask].eq(y_test[mask]).sum()/y_test[mask].shape[0]).detach().cpu().numpy())
    return class_err

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
            budgets = [9/255,18/255, 36/255, 72/255, 108/255, 144/255]
        else:
            #budgets = [2/255,8/255]
            budgets=[2/255, 8/255, 16/255] # wide resnet
        testset = datasets.CIFAR10('./data', train=False, 
            transform=transforms.Compose([transforms.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, pin_memory=True)
        if args.model =='small':
            model = pblm.cifar_model().cuda()
        elif args.model == 'large':
            model = pblm.cifar_model_large().cuda()
        elif args.model =='resnet':
            model = pblm.cifar_model_resnet(N=1,factor=1).cuda()
        elif args.model == 'wideresnet':
            model = wide_resnet_cifar().cuda()
            model = torch.nn.DataParallel(model)

    
    elif args.dataset == 'tiny_imagenet':
        if args.norm == 'L2':
            budgets = [150/255,300/255,450/255]
        elif args.norm == 'Linf':
            budgets = [1/255, 2/255, 3/255]

        tmpdir = os.getenv('TMPDIR')
        shutil.copy2('tinyimagenet_download.sh', tmpdir)
        os.chdir(tmpdir)
        rc = subprocess.call(tmpdir +  '/tinyimagenet_download.sh')
        test_data = datasets.ImageFolder(tmpdir + '/tiny-imagenet-200' + '/val',
                                        transform=transforms.Compose([
                                            transforms.CenterCrop(56),
                                            transforms.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size // 5, pin_memory=True,
                                            num_workers=min(multiprocessing.cpu_count(), 4))
            
        model = wide_resnet_imagenet64(in_planes=16,widen_factor=10).cuda()
        model = torch.nn.DataParallel(model)

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0).cuda()
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0).cuda()




    if args.mode == 'wong':
        df = {'eps':[], 'robust_error':[], 'clean_error':[], 'class_error':[], 'robust_class_error':[]}
        for eps in budgets:
            ckpt = torch.load(f'/cluster/home/pdebartol/convex_adversarial/examples/{args.dataset}_{args.norm}/wong_{args.model}_eps_{eps}.pth')['state_dict'][0]
            model.load_state_dict(ckpt)
            model.eval()
            if args.dataset == 'cifar':
                model_norm = ModelNormWrapper(model, means=[0.485, 0.456, 0.406],
                                     stds=[0.225, 0.225, 0.225])
                model_norm.eval()
                y_pred = []
                for x,y in test_loader:
                    y_pred.append(model_norm(x.cuda()).argmax(1))
                y_pred = torch.cat(y_pred)
                class_error = error_per_class(y_test,y_pred)
                clean_error =1- y_pred.eq(y_test).sum()/y_test.shape[0]

                adversary = AutoAttack(model_norm, norm=args.norm, eps=eps, version='plus')
                adv_complete = adversary.run_standard_evaluation(x_test, y_test,
                        bs=args.batch_size)
                robust_error =1- model_norm(adv_complete).argmax(1).eq(y_test).sum()/y_test.shape[0]
                robust_class_error = error_per_class(y_test, model_norm(adv_complete).argmax(1))

            else:
                adversary = AutoAttack(model, norm=args.norm, eps=eps, version='plus')
                clean_error =1- model(x_test).argmax(1).eq(y_test).sum()/y_test.shape[0]
                adv_complete = adversary.run_standard_evaluation(x_test, y_test,
                        bs=args.batch_size)
                robust_error =1- model(adv_complete).argmax(1).eq(y_test).sum()/y_test.shape[0] 
            print(f'cert mode, eps: {eps}, clean error: {clean_error}, robust_error: {robust_error}')
            df['eps'].append(eps)
            df['robust_error'].append(robust_error.cpu().detach().numpy())
            df['clean_error'].append(clean_error.cpu().detach().numpy())
            df['class_error'].append(class_error)
            df['robust_class_error'].append(robust_class_error)

    elif args.mode == 'madry':
        df = {'eps':[], 'robust_error':[], 'clean_error':[], 'class_error':[], 'robust_class_error':[]}
        if args.dataset == 'cifar':
                model = ModelNormWrapper(model,  stds = [0.2023, 0.1994, 0.2010],means = [0.4914, 0.4822, 0.4465])
        elif args.dataset == 'tiny_imagenet':
                model = ModelNormWrapper(model, stds = [0.2302, 0.2265, 0.2262],means =[0.4802, 0.4481, 0.3975])

        for eps in budgets:
            ckpt = torch.load(f'/cluster/work/yang/pier/{args.dataset}_{args.norm}/madry_{args.model}_eps_{eps}.pth')['net']
            model.load_state_dict(ckpt)
            model.eval()
            y_pred = []
            for x,y in test_loader:
                y_pred.append(model(x.cuda()).argmax(1))
            y_pred = torch.cat(y_pred)
            class_error = error_per_class(y_test,y_pred)
            clean_error =1- y_pred.eq(y_test).sum()/y_test.shape[0]
            print(f'clean error: {clean_error}')
            adversary = AutoAttack(model, norm=args.norm, eps=eps,  version='plus')
            adv_complete = adversary.run_standard_evaluation(x_test, y_test,
                    bs=args.batch_size)
            y_pred = []
            for x in adv_complete:
                y_pred.append(model(x.unsqueeze(dim=0).cuda()).argmax(1))
            y_pred = torch.cat(y_pred)
            robust_error =1- y_pred.eq(y_test).sum()/y_test.shape[0]
            robust_class_error = error_per_class(y_test, y_pred)
         
            print(f'cert mode, eps: {eps}, clean error: {clean_error}, robust_error: {robust_error}')
            df['eps'].append(eps)
            df['robust_error'].append(robust_error.cpu().detach().numpy())
            df['clean_error'].append(clean_error.cpu().detach().numpy())
            df['class_error'].append(class_error)
            df['robust_class_error'].append(robust_class_error)


    elif args.mode == 'zhang':
        df = {'eps':[], 'robust_error':[], 'clean_error':[], 'class_error':[], 'class_error':[], 'robust_class_error':[]}
        for eps in budgets:
            ckpt = torch.load(f'/cluster/home/pdebartol/convex_adversarial/examples/{args.dataset}_{args.norm}/zhang_{args.model}_eps_{eps}.pth')['state_dict']
            model.load_state_dict(ckpt)
            model.eval()
            if args.dataset=='cifar':
                model_norm = ModelNormWrapper(model, stds = [0.2023, 0.1994, 0.2010],means = [0.4914, 0.4822, 0.4465])
                adversary = AutoAttack(model_norm, norm=args.norm, eps=eps, version='plus')
                y_pred = []
                for x,y in test_loader:
                    y_pred.append(model_norm(x.cuda()).argmax(1))
                y_pred = torch.cat(y_pred)
                class_error = error_per_class(y_test,y_pred)

                clean_error =1- y_pred.eq(y_test).sum()/y_test.shape[0]
                adv_complete = adversary.run_standard_evaluation(x_test, y_test,
                        bs=args.batch_size)
                robust_error =1- model_norm(adv_complete).argmax(1).eq(y_test).sum()/y_test.shape[0]
                robust_class_error = error_per_class(y_test, model_norm(adv_complete).argmax(1))
 
            else:
                adversary = AutoAttack(model, norm=args.norm, eps=eps, version='plus')
                clean_error =1- model(x_test).argmax(1).eq(y_test).sum()/y_test.shape[0]
                adv_complete = adversary.run_standard_evaluation(x_test, y_test,
                        bs=args.batch_size)
                robust_error =1- model(adv_complete).argmax(1).eq(y_test).sum()/y_test.shape[0]

            print(f'cert mode, eps: {eps}, clean error: {clean_error}, robust_error: {robust_error}')
            df['eps'].append(eps)
            df['robust_error'].append(robust_error.cpu().detach().numpy())
            df['clean_error'].append(clean_error.cpu().detach().numpy())
            df['class_error'].append(class_error)
            df['robust_class_error'].append(robust_class_error)


   
    df = pd.DataFrame.from_dict(df)
    df.to_csv(f'/cluster/work/yang/pier/{args.dataset}_{args.norm}/{args.mode}_{args.model}.csv')
    np.save(f'/cluster/work/yang/pier/{args.dataset}_{args.norm}/{args.mode}_{args.model}_robust.npy', df['robust_class_error'])
    np.save(f'/cluster/work/yang/pier/{args.dataset}_{args.norm}/{args.mode}_{args.model}_std.npy', df['class_error'])
  
