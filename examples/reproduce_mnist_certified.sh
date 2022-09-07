epstrain="0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0"
for eps in ${epstrain}; do 
        cmd="python mnist.py --epsilon ${eps} --model small  --prefix mnist --norm_test l2 --proj 50 --norm_train l2_normal"
        bsub  -n 16 -W 7:59 -R "rusage[mem=1000,ngpus_excl_p=1]" ${cmd}
        sleep 1
done

