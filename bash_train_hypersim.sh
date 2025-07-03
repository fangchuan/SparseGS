# pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# List of scenes
hypersim_scenes=(
    'ai_003_006'
    'ai_014_010'
    'ai_016_007'
    'ai_017_005'
    'ai_017_006'
    'ai_018_003'
    'ai_018_008'
    'ai_037_008'
    'ai_039_006'
    'ai_043_005'
)
# Loop through each scene 
for i in "${!hypersim_scenes[@]}"; do
    scene="${hypersim_scenes[$i]}"

    echo "Reconstructing scene: $scene"

    python3 train.py \
        --source_path data/spatialgen-hypersim-eval-32view/val/$scene \
        --model_path output/spatialgen-hypersim-32view/${scene}_lpips_radegs \
        --beta 5.0 \
        --lambda_pearson 0.05 \
        --lambda_local_pearson 0.15 \
        --box_p 32 \
        --p_corr 0.5 \
        --lambda_warp_reg 0.4 \
        --iterations 7000 \
        -r 1 \
        --warp_reg_start_itr 3000

    python3 render.py \
        --source_path data/spatialgen-hypersim-eval-32view/val/$scene \
        --model_path output/spatialgen-hypersim-32view/${scene}_lpips_radegs \
        --no_load_depth \
        --iteration 7000

    echo "Finished processing scene: $scene"
    echo "----------------------------------"
done