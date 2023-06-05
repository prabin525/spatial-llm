for p_length in 'zero-shot' '3-shot'
do
    sbatch  --job-name=$p_length.predict_coor template.slurm $p_length;
done;