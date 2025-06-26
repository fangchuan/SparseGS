python3 train.py \
    --source_path data/spatialgen-hypersim-eval/val/ai_047_006 \
    --model_path output/spatialgen-hypersim/ai_047_006 \
    --beta 5.0 \
    --lambda_pearson 0.05 \
    --lambda_local_pearson 0.15 \
    --box_p 32 \
    --p_corr 0.5 \
    --lambda_diffusion 0.001 \
    --SDS_freq 0.1 \
    --step_ratio 0.99 \
    --lambda_reg 0.1 \
    --prune_sched 20000 \
    --prune_perc 0.98 \
    --prune_exp 7.5 \
    --iterations 30000 \
    -r 1

python3 render.py \
    --source_path data/spatialgen-hypersim-eval/val/ai_047_006 \
    --model_path output/spatialgen-hypersim/ai_047_006 \
    --no_load_depth \
    --iteration 30000
