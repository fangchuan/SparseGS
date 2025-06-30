# pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# List of scenes
spatialgen_scenes=(
    '20241106_3FO4K5HXIXL1_perspective_room_279'
    '20241106_3FO4K5HXIXL1_perspective_room_281'
    '20241106_3FO4K5HXJBY5_perspective_room_240'
    '20241118_3FO4K5G1C7DO_perspective_room_507'
    '20241118_3FO4K5FXC4C9_perspective_room_821'
    '20241118_3FO4K5FXFJPA_perspective_room_812'
    '20241118_3FO4K5FXGOTM_perspective_room_1116'
    '20241118_3FO4K5G1CO3T_perspective_room_628'
    '20241118_3FO4K5G1EOBF_perspective_room_585'
    '20241118_3FO4K5G1X4MX_perspective_room_1112'
)
# Loop through each scene 
for i in "${!spatialgen_scenes[@]}"; do
    scene="${spatialgen_scenes[$i]}"

    echo "Reconstructing scene: $scene"

    python3 train.py \
        --source_path data/spatialgen-spatialgen-eval/val/$scene \
        --model_path output/spatialgen-spatialgen/${scene}_warp20_fixed \
        --beta 5.0 \
        --lambda_pearson 0.05 \
        --lambda_local_pearson 0.15 \
        --box_p 32 \
        --p_corr 0.5 \
        --lambda_diffusion 0.000 \
        --SDS_freq 0.1 \
        --step_ratio 0.99 \
        --lambda_reg 0.2 \
        --prune_sched 20000 \
        --prune_perc 0.98 \
        --prune_exp 7.5 \
        --iterations 30000 \
        -r 1 \
        --warp_reg_start_itr 100

    python3 render.py \
        --source_path data/spatialgen-spatialgen-eval/val/$scene \
        --model_path output/spatialgen-spatialgen/${scene}_warp20_fixed \
        --no_load_depth \
        --iteration 30000

    echo "Finished processing scene: $scene"
    echo "----------------------------------"
done