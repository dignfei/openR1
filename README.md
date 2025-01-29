# reproduction of DeepSeek R1 Zero


openr1 is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) in gsm8k . 

Through RL, the base LM develops self-verification and search abilities all on its own 


## Fast Start at single GPU

```
docker  run --name openr1  --gpus all -itd  -v "$(pwd)/outputs:/root/code/outputs"  agimaker/openr1:0.2
```
To start training with a single command, the training results will be saved in the 'outputs' folder of the current directory. By default, the GRPO is used to train the qwen2.5-0.5B model, and training can commence with a single 3090, 4090, or 5090 GPU.

## Train on multiple GPUs

```
git clone https://github.com/dignfei/openr1
cd openr1
docker  run --name openr1  --gpus all -itd  -v "$(pwd)/:/root/codes/"  agimaker/openr1:0.2 sh -c " PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/root/openR1/data/gsm8k/train.parquet \
    data.val_files=/root/openR1/data/gsm8k/test.parquet \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=/root/openR1/models/Qwen2.5-0.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rl_grpo_gsm8k' \
    trainer.experiment_name='Qwen2.5-0.5B-Instruct_function_rm_seq_packing' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15"

```
The parameters trainer.n_gpus_per_node=2 and actor_rollout_ref.rollout.tensor_model_parallel_size=2 indicate the use of 2 GPUs. Adjust this number to match the quantity of GPUs you have. Additionally, trainer.nnodes=1 indicates that there is 1 host machine.


## Custom training data


**Data Preparation**
```
python main/gsm8k.py  --local_dir /root/openR1/data/gsm8k

```
Modify the content of main/gsm8k.py, changing "gsm8k" to your own dataset.

**Custom model**


```
actor_rollout_ref.model.path=/root/openR1/models/Qwen2.5-0.5B
```
Modify the launch parameters to change /root/openR1/models/Qwen2.5-0.5B to the path of your own model.


## Citation
```
@misc{tinyzero,
author       = {dignfei},
title        = {openR1},
howpublished = {https://github.com/dignfei/openR1},
note         = {Accessed: 2025-01-24},
year         = {2025}
}
```
