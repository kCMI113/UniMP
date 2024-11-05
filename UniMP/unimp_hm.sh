lr=0.001
bsz=16
modelname="4b"
subset="hm"
gamma=2
nsteps=1

python mmrec.py \
--pretrained_model_name_or_path=${modelname} \
--dataset_resampled \
--mmrec_path="../data/${subset}/" \
--subset=${subset} \
--batch_size=${bsz} \
--task=rec \
--use_reweight \
--gamma=${gamma} \
--gradient_accumulation_steps=${nsteps} \
--num_epochs=50 \
--lr_scheduler=constant \
--delete_previous_checkpoint \
--learning_rate=${lr} \
--wandb_project=mmrec \
--external_save_dir=mmrec-3b \
--run_name="mmrec-hm-len820-reweight-${gamma}-${nsteps}-answer-nonsemantic-${lr}-constant-${modelname}-b${bsz}-${subset}" \
--warmup_steps_ratio=0.01 \
--save_hf_model \
--do_test \
--single_task \
--report_to_wandb \
