epstrain="0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0"
for eps in ${epstrain}; do 
        cmd="python madry_training.py --epsilon ${eps} --model small --dataset mnist"
        bsub  -n 16 -W 7:59 -R "rusage[mem=1000,ngpus_excl_p=1]"  ${cmd}
        sleep 1
done
 