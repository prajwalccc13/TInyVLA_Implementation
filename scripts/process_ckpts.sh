#!/bin/bash

LLM_MODEL_SIZE=410M

# 源文件夹路径
source_dir="/data/junjiewen/droid_results/checkpoint_all/pythia_${LLM_MODEL_SIZE}/vanilla_pythia_pt_f_vit_act_new_view/llavaPythia-v0-robot-action-pp_tennis_lora_pretrain/"

target_dir="/data/junjiewen/droid_results/checkpoint_all/pythia_${LLM_MODEL_SIZE}_pure/vanilla_pythia_pt_f_vit_act_new_view/llavaPythia-v0-robot-action-pp_tennis_lora_pretrain"

mkdir -p $target_dir
# 目标文件夹路径


# 要排除的文件夹名的通配符
exclude_pattern="global_step*"

# 递归复制文件夹，并排除匹配指定通配符的文件夹
echo "copying checkpoints from $source_dir to $target_dir"
rsync -av --exclude="$exclude_pattern" --exclude="$exclude_pattern/**" "$source_dir/" "$target_dir/"

echo 'tranfer checkpoints to non_lora_trainables.bin'
for dir in "$source_dir"/*/ ; do
    # 检查文件夹名称是否包含'checkpoint'
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
      if ! find "$dir" -mindepth 1 -type f -name "non_lora_trainables.bin" | grep -q .; then
        cd "$dir" || exit
        python ./zero_to_fp32.py ./ ${target_dir}/$(basename "$dir")/non_lora_trainables.bin
        # cp $OUTPUT/non_lora_trainables.bin $dir
        fi
    fi
done

# 进入目标目录
cd "/data/junjiewen/droid_results/checkpoint_all" || exit

# 压缩目录并指定相对路径
#tar -czvf "pythia_${LLM_MODEL_SIZE}.tar.gz" "pythia_${LLM_MODEL_SIZE}_pure"
#echo "compress checkpoints to /data/junjiewen/droid_results/checkpoint_all/pythia_${LLM_MODEL_SIZE}.tar.gz"
#
# 删除临时文件夹
#rm -r "pythia_${LLM_MODEL_SIZE}_pure"

