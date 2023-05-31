p_type='close'
for p_length in 'zero-shot' '3-shot'
do
    for state in 'true' 'false'
    do
        sbatch  --job-name=$p_type.$p_length.$state template.slurm $p_type $p_length $state;
    done;
done;