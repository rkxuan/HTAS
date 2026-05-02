for dataset in "mbpp" "asynchow"
do
    for temperature in "0.333" "3.0"
    do
        python /root/HATS/main/run_${dataset}.py --topology=Hybrid --temperature=$temperature
    done
done