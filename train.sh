dataset=('coffee_martini' 'cook_spinach' 'cut_roasted_beef' 'flame_salmon' 'flame_steak' 'sear_steak')
config='n3v'
device=0

for data in ${dataset[@]};
do
    model_path=output/${data}
    source_path=<your_dataset_path>/${data}
    # 在yaml配置文件中恢复了model_path和source_path的默认值，因此这里不需要再传入参数了。
    CUDA_VISIBLE_DEVICES=${device} python main.py --config configs/${config}/default.yaml  --model_path $model_path \
    --source_path $source_path 
    
done

# 2. 创建虚拟环境 (在你的 BiShe 文件夹下)
cd ~/BiShe
python3 -m venv myenv

# 3. 激活环境
source myenv/bin/activate