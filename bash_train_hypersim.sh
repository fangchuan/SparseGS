# pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# List of scenes
spatialgen_scenes=(
    'ai_047_006'
)
# Loop through each scene 
for i in "${!spatialgen_scenes[@]}"; do
    scene="${spatialgen_scenes[$i]}"

    echo "Reconstructing scene: $scene"

    python3 train.py \
        --source_path data/spatialgen-spatialgen-eval/val/$scene \
        --model_path output/spatialgen-spatialgen/${scene}_warp20_radegs \
        --beta 5.0 \
        --lambda_pearson 0.05 \
        --lambda_local_pearson 0.15 \
        --box_p 32 \
        --p_corr 0.5 \
        --lambda_diffusion 0.000 \
        --SDS_freq 0.1 \
        --step_ratio 0.99 \
        --lambda_reg 0.4 \
        --iterations 30000 \
        -r 1 \
        --warp_reg_start_itr 100

    python3 render.py \
        --source_path data/spatialgen-spatialgen-eval/val/$scene \
        --model_path output/spatialgen-spatialgen/${scene}_warp20_radegs \
        --no_load_depth \
        --iteration 30000

    echo "Finished processing scene: $scene"
    echo "----------------------------------"
done