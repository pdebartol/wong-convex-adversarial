epstrain="0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0"
for eps in ${epstrain}; do 
        cmd="python mnist.py --epsilon ${eps} --model large  --prefix mnist --test_batch_size 1 --norm_train l2_normal --norm_test l2 --proj 50"
        bsub  -n 16 -W 23:59 -R "rusage[mem=1000,ngpus_excl_p=1]"  -R "select[gpu_mtotal0>=20240]" ${cmd}
        sleep 1
done


