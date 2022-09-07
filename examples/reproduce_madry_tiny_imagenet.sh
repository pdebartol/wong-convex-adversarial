#epstrain="0.5882352941176471 1.1764705882352942 1.7647058823529411" #L2
epstrain="0.00392156862745098 0.00784313725490196 0.011764705882352941" #Linf
for eps in ${epstrain}; do 
        cmd="python madry_training.py --eps ${eps} --dataset tiny_imagenet --model wide_resnet --optim sgd  --lr 0.1 --weight_decay 0.0005 --epochs 200 --batch_size 512 --norm Linf"
        bsub  -n 16 -W 40:59 -R "rusage[mem=1000,ngpus_excl_p=4]"  -R "select[gpu_mtotal0>=20240]"  ${cmd}
        sleep 1
done
 