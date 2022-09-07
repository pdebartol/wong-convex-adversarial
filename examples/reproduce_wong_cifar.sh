#epstrain="0.00784313725490196 0.03137254901960784"
#epstrain="0.03529411764705882 0.07058823529 0.1411764705882353 0.21176470588235294 0.28235294117 0.4235294117647059" #L2
epstrain="0.28235294117"
for eps in ${epstrain}; do 
        cmd="python cifar.py --epsilon ${eps}  --model resnet  --resnet_N 1 --resnet_factor 1 --epochs 60 --starting_epsilon=0.001  --verbose 200 --prefix tmp/cifar --cascade 1 --norm_test l2_normal --norm_train l2_normal --proj 50"
        bsub  -n 16 -W 119:59 -R "rusage[mem=1000,ngpus_excl_p=1]"  -R "select[gpu_mtotal0>=20240]" ${cmd}
        sleep 1
done

