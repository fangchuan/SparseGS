# pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# List of scenes
spatialgen_scenes=(
    "20241118_3FO4K5G1TRLX_perspective_room_442"
    "20241118_3FO4K5FWIQ9R_perspective_room_1341"
    "20241118_3FO4K5G1EHMD_perspective_room_464"
    "20241118_3FO4K5FX09PP_perspective_room_555"
    "20241118_3FO4K5I0YCTK_perspective_room_536"
    "20241118_3FO4K5G1H696_perspective_room_731"
    "20241118_3FO4K5G1F61K_perspective_room_464"
    "20241118_3FO4K5G1XLD3_perspective_room_698"
    "20241118_3FO4K5FXFJPA_perspective_room_812"
    "20241118_3FO4K5FX2XCI_perspective_room_544"
    "20241118_3FO4K5G1DMJ4_perspective_room_862"
    "20241118_3FO4K5FXHTXY_perspective_room_864"
    "20241118_3FO4K5FX1EU2_perspective_room_1184"
    "20241118_3FO4K5FX9J2H_perspective_room_1261"
    "20241118_3FO4K5FX5596_perspective_room_1099"
    "20241118_3FO4K5G1C7DO_perspective_room_506"
    "20241118_3FO4K5FXDQ6Q_perspective_room_657"
    "20241118_3FO4K5FXHJWV_perspective_room_1227"
    "20241118_3FO4K5G1FTFR_perspective_room_1075"
    "20241118_3FO4K5FX5PBC_perspective_room_1064"
    "20241118_3FO4K5I19AKJ_perspective_room_392"
    "20241118_3FO4K5FXC7OA_perspective_room_438"
    "20241118_3FO4K5G1XLD3_perspective_room_695"
    "20241118_3FO4K5FXC4C9_perspective_room_821"
    "20241118_3FO4K5FXGOTM_perspective_room_1116"
    "20241118_3FO4K5I0OSGP_perspective_room_406"
    "20241118_3FO4K5G1C7DO_perspective_room_507"
    "20241118_3FO4K5G1W67N_perspective_room_596"
    "20241118_3FO4K5FWTT53_perspective_room_1057"
    "20241118_3FO4K5FX5IMA_perspective_room_913"
    "20241118_3FO4K5FWU0T5_perspective_room_1151"
    "20241118_3FO4K5FWHL5F_perspective_room_1281"
    "20241118_3FO4K5FWWL3W_perspective_room_535"
    "20241118_3FO4K5FX6H2K_perspective_room_728"
    "20241118_3FO4K5G1B29C_perspective_room_805"
)
# Loop through each scene 
for i in "${!spatialgen_scenes[@]}"; do
    scene="${spatialgen_scenes[$i]}"

    echo "Reconstructing scene: $scene"

    python3 train.py \
        --source_path data/exp_text2scene_spatialgen_16view/val/$scene \
        --model_path output/exp_text2scene_spatialgen_16view/${scene}_lpips_radegs \
        --beta 5.0 \
        --lambda_pearson 0.05 \
        --lambda_local_pearson 0.15 \
        --box_p 32 \
        --p_corr 0.5 \
        --lambda_warp_reg 0.4 \
        --iterations 7000 \
        -r 1 \
        --warp_reg_start_itr 3000 \
        --sh_degree 0

    python3 render.py \
        --source_path data/exp_text2scene_spatialgen_16view/val/$scene \
        --model_path output/exp_text2scene_spatialgen_16view/${scene}_lpips_radegs \
        --no_load_depth \
        --iteration 7000

    echo "Finished processing scene: $scene"
    echo "----------------------------------"
done

# # Loop through each scene 
# for i in "${!spatialgen_scenes[@]}"; do
#     scene="${spatialgen_scenes[$i]}"

#     echo "Reconstructing scene: $scene"

#     python3 train.py \
#         --source_path data/spatialgen-spatialgen-eval/val/$scene \
#         --model_path output/spatialgen-spatialgen/${scene}_lpips_radegs \
#         --beta 5.0 \
#         --lambda_pearson 0.05 \
#         --lambda_local_pearson 0.15 \
#         --box_p 32 \
#         --p_corr 0.5 \
#         --lambda_warp_reg 0.4 \
#         --iterations 7000 \
#         -r 1 \
#         --warp_reg_start_itr 3000

#     python3 render.py \
#         --source_path data/spatialgen-spatialgen-eval/val/$scene \
#         --model_path output/spatialgen-spatialgen/${scene}_lpips_radegs \
#         --no_load_depth \
#         --iteration 7000

#     echo "Finished processing scene: $scene"
#     echo "----------------------------------"
# done