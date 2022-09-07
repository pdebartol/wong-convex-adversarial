#epstrain="0.00784313725490196 0.03137254901960784 0.06274509803921569" #LINF
#epstrain="0.07058823529 0.21176470588 0.28235294117" #L2
epstrain="0.4235294117647059" #L2

norm="L2"
for eps in ${epstrain}; do 
        cmd="python madry_training.py --epsilon ${eps} --model resnet --dataset cifar --batch_size 128 --epochs 150 --optim sgd  --lr 0.05 --weight_decay 0.0 --norm ${norm}"
        bsub  -n 16 -W 23:59 -R "rusage[mem=1000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=20240]" ${cmd}
        sleep 1
done
 