# pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
# cd to the directory containing the train.py and render.py scripts
curr_working_dir=$(dirname "$0")
cd $curr_working_dir
echo $curr_working_dir

DATA_DIR=/data-nas/experiments/zhenqing/diffsplat/out/hypersim_sd15_8rgbsscm_warp_512/Text23D_exp_hypersim_controlnet_32view_ckpt9k_new
# List of scenes
hypersim_scenes=(
    'ai_010_004'
    'ai_010_005'
)
# Loop through each scene 
for i in "${!hypersim_scenes[@]}"; do
    scene="${hypersim_scenes[$i]}"

    echo "Reconstructing scene: $scene"

    python3 train.py \
        --source_path $DATA_DIR/val/$scene \
        --model_path sparseradegs_out/Text23D_exp_hypersim_controlnet_32view_ckpt9k_new/${scene}_lpips_radegs \
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
        --source_path $DATA_DIR/val/$scene \
        --model_path sparseradegs_out/Text23D_exp_hypersim_controlnet_32view_ckpt9k_new/${scene}_lpips_radegs \
        --no_load_depth \
        --iteration 7000

    echo "Finished processing scene: $scene"
    echo "----------------------------------"
done