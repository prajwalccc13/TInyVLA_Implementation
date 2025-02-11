#!/bin/bash
LLM_MODEL_SIZE=410M # specify the model size

ACTION_HEAD=droid_diffusion # specify action policy head type
cp -r /data/team/wjj/model_param/${ACTION_HEAD}_results/checkpoint_all/pythia_410M/vanilla_pythia_pt_f_vit/llavaPythia-v0-robot-action-1dot7t_pretrain_lora2/checkpoint-3000/preprocessor_config.json $pretrain_path
# define OUTPUT path
OUTPUT=/data/private/data/model_param/${ACTION_HEAD}_results/checkpoint_all/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-robot-action-5mt_tennis_mug_drawer_cube_flower_lora_all_stateEncoder_moreLayerNorm # specify the output path and name like '5mt_tennis_mug_drawer_cube_flower_lora_all_stateEncoder_moreLayerNorm'

if [ -d "$OUTPUT" ]; then
   echo 'output exists'
else
   echo '!!output not exists!!'
   mkdir -p $OUTPUT
fi
# backup the train scripts
cp ./scripts/train.sh $OUTPUT

deepspeed --master_port 29600 --num_gpus=8 --num_nodes=1 ./train_act_pythia.py \
  --deepspeed scripts/zero2.json \
  --lora_enable True \
  --lora_module 'vit llm' \
  --load_pretrain False \
  --pretrain_image_size 320 \
  --lora_r 64 \
  --lora_alpha 256 \
  --non_lora_lr 2e-5 \
  --task_name "5mt_tennis_mug_drawer_cube_flower_succ_t0001_s0-0" \
  --model_name_or_path /data/team/wjj/model_param/weights/pythia_$LLM_MODEL_SIZE/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune \
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower True \
  --freeze_backbone True \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 10000 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 50 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.005 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --action_head $ACTION_HEAD \
  --policy_class $ACTION_HEAD \
  --use_state True \
  --concat "token_cat" \
  --window_size 6 \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log

for dir in "$OUTPUT"/*/ ; do
    # 检查文件夹名称是否包含'checkpoint'
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp /data/private/weights/openai/clip-vit-large-patch14-336/preprocessor_config.json $dir
    fi
done

