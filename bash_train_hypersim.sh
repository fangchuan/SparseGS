# pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# List of scenes
hypersim_scenes=(
    'ai_003_006'
    'ai_014_010'
    'ai_016_007'
    'ai_017_005'
    # 'ai_017_006'
    # 'ai_018_003'
    # 'ai_018_008'
    # 'ai_037_008'
    # 'ai_039_006'
    # 'ai_043_005'
)
# Loop through each scene 
for i in "${!hypersim_scenes[@]}"; do
    scene="${hypersim_scenes[$i]}"

    echo "Reconstructing scene: $scene"

    python3 train.py \
        --source_path data/spatialgen-hypersim-eval-32view/val/$scene \
        --model_path output/spatialgen-hypersim-32view/${scene}_warp20_sparsegs \
        --beta 5.0 \
        --lambda_pearson 0.05 \
        --lambda_local_pearson 0.15 \
        --box_p 32 \
        --p_corr 0.5 \
        --lambda_diffusion 0.000 \
        --SDS_freq 0.1 \
        --step_ratio 0.99 \
        --lambda_reg 0.4 \
        --prune_sched 20000 \
        --prune_perc 0.88 \
        --prune_exp 7.5 \
        --iterations 30000 \
        -r 1 \
        --warp_reg_start_itr 100

    python3 render.py \
        --source_path data/spatialgen-hypersim-eval-32view/val/$scene \
        --model_path output/spatialgen-hypersim-32view/${scene}_warp20_sparsegs \
        --no_load_depth \
        --iteration 30000

    echo "Finished processing scene: $scene"
    echo "----------------------------------"
done