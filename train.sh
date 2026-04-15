# dataset=('miku_shaungxue_daxi_single_nowide' 'cook_spinach' 'cut_roasted_beef' 'flame_salmon' 'flame_steak' 'sear_steak')
# dataset=('miku_shaungxue_daxi_single_nowide' 'miku_shaungxue_daxi_single_wide')
dataset=('miku_shaungxue_daxi_single_nowide')

config='n3v'
device=0
full='full'
Full='Full'
short='test_short'
Short='Test'

# # Hybrid-GS SHORT
# for data in ${dataset[@]};
# do
#     model_path=output/hybrid_gs/${data}/${short}
#     source_path=datasets/${data}/${short}
#     # 在yaml配置文件中恢复了model_path和source_path的默认值，因此这里不需要再传入参数了。
#     CUDA_VISIBLE_DEVICES=${device} python3 train.py --config configs/${config}/HybridGS_${Short}.yaml  --model_path $model_path \
#     --source_path $source_path 
#     CUDA_VISIBLE_DEVICES=${device} python3 render.py --config configs/${config}/HybridGS_${Short}.yaml  --model_path $model_path \
#     --source_path $source_path --start_checkpoint $model_path/chkpnt_8000.pth
# done

# Hybrid-GS Full
for data in ${dataset[@]};
do
    model_path=output/hybrid_gs/${data}/${full}
    source_path=datasets/${data}/${full}
    # 在yaml配置文件中恢复了model_path和source_path的默认值，因此这里不需要再传入参数了。
    CUDA_VISIBLE_DEVICES=${device} python3 train.py --config configs/${config}/HybridGS_${Full}.yaml  --model_path $model_path \
    --source_path $source_path 
    CUDA_VISIBLE_DEVICES=${device} python3 render.py --config configs/${config}/HybridGS_${Full}.yaml  --model_path $model_path \
    --source_path $source_path --start_checkpoint $model_path/chkpnt_10000.pth
done

# # SwinGS SHORT
# for data in ${dataset[@]};
# do
#     model_path=output/swings/${data}/${short}
#     source_path=datasets/${data}/${short}
#     # 在yaml配置文件中恢复了model_path和source_path的默认值，因此这里不需要再传入参数了。
#     CUDA_VISIBLE_DEVICES=${device} python3 train.py --config configs/${config}/SwinGS_${Short}.yaml  --model_path $model_path \
#     --source_path $source_path 
#     CUDA_VISIBLE_DEVICES=${device} python3 render.py --config configs/${config}/SwinGS_${Short}.yaml  --model_path $model_path \
#     --source_path $source_path --start_checkpoint $model_path/chkpnt_8000.pth
# done

# # SwinGS Full
# for data in ${dataset[@]};
# do
#     model_path=output/swings/${data}/${full}
#     source_path=datasets/${data}/${full}
#     # 在yaml配置文件中恢复了model_path和source_path的默认值，因此这里不需要再传入参数了。
#     CUDA_VISIBLE_DEVICES=${device} python3 train.py --config configs/${config}/SwinGS_${Full}.yaml  --model_path $model_path \
#     --source_path $source_path 
#     CUDA_VISIBLE_DEVICES=${device} python3 render.py --config configs/${config}/SwinGS_${Full}.yaml  --model_path $model_path \
#     --source_path $source_path --start_checkpoint $model_path/chkpnt_10000.pth
# done


# # 3D4DGS SHORT
# for data in ${dataset[@]};
# do
#     model_path=output/3d4dgs/${data}/${short}
#     source_path=datasets/${data}/${short}
#     # 在yaml配置文件中恢复了model_path和source_path的默认值，因此这里不需要再传入参数了。
#     CUDA_VISIBLE_DEVICES=${device} python3 train.py --config configs/${config}/3D4DGS_${Short}.yaml  --model_path $model_path \
#     --source_path $source_path 
#     CUDA_VISIBLE_DEVICES=${device} python3 render.py --config configs/${config}/3D4DGS_${Short}.yaml  --model_path $model_path \
#     --source_path $source_path --start_checkpoint $model_path/chkpnt_8000.pth
# done
# 3D4DGS Full
for data in ${dataset[@]};
do
    model_path=output/3d4dgs/${data}/${full}
    source_path=datasets/${data}/${full}
    # 在yaml配置文件中恢复了model_path和source_path的默认值，因此这里不需要再传入参数了。
    CUDA_VISIBLE_DEVICES=${device} python3 train.py --config configs/${config}/3D4DGS_${Full}.yaml  --model_path $model_path \
    --source_path $source_path 
    CUDA_VISIBLE_DEVICES=${device} python3 render.py --config configs/${config}/3D4DGS_${Full}.yaml  --model_path $model_path \
    --source_path $source_path --start_checkpoint $model_path/chkpnt_40000.pth
done

