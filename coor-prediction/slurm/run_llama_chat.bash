MODEL='llama2-chat'

model_size='0'
for p_type in 0 1
do
    for p_length in 'zero-shot'
    do
        sbatch  --job-name=$MODEL.$p_type.$p_length.$model_size template_small.slurm $MODEL $model_size $p_type $p_length;
    done;
done;