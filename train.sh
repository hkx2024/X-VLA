
accelerate launch \
    --mixed_precision bf16 \
    train.py \
    --models '2toINF/X-VLA-Pt' \
    --train_metas_path ./libero_goal_16/info.json \
    --learning_rate 1e-4 \
    --learning_coef 0.1 \
    --iters 50000 \
    --freeze_steps 1000 \
    --warmup_steps 2000