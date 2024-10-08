
# CUDA_VISIBLE_DEVICES=0,1 python src/train.py \
# python src/train.py \

# BATCH_GPU=2
# BATCH_GPU=8      # duo     ( 9.6GB)
BATCH_GPU=16       # duo     (16.0GB)
# BATCH_GPU=32     # duo     ()
# BATCH_GPU=256    # daugman (40.0GB)

DATA_MODULE=casia_webface                          # original (whole dataset, 10k classes)
# DATA_MODULE=casia_webface_1000CLASS_SEQUENTIAL   # Bernardo
# DATA_MODULE=casia_webface_2000CLASS_SEQUENTIAL   # Bernardo
# DATA_MODULE=casia_webface_5000CLASS_SEQUENTIAL   # Bernardo
# DATA_MODULE=casia_webface_imgs_crops             # Bernardo

# ThreeDMM_LOSS_LAMBDA=0.001
# ThreeDMM_LOSS_LAMBDA=0.005
# ThreeDMM_LOSS_LAMBDA=0.05               # same as 'identity_consistency_loss_lambda'

# BFM_LOSS_LAMBDA=0.001
# BFM_LOSS_LAMBDA=0.005
# BFM_LOSS_LAMBDA=0.01
# BFM_LOSS_LAMBDA=0.05                    # same as 'identity_consistency_loss_lambda'



# BFM_ID_LAMBDA=0.0
BFM_ID_LAMBDA=0.01
# BFM_ID_LAMBDA=0.05

# BFM_EXP_LAMBDA=0.0
BFM_EXP_LAMBDA=0.01
# BFM_EXP_LAMBDA=0.05

# BFM_POSE_LAMBDA=0.0
BFM_POSE_LAMBDA=0.01
# BFM_POSE_LAMBDA=0.05



# DATALOADER_DETERMINISTIC=false          # default (using data augmentation)
DATALOADER_DETERMINISTIC=true             # Bernardo (no data augmentation)



CKPT_PATH=null                          # default
# CKPT_PATH='experiments_WITH_BFM_CONSISTENCY_CONSTRAINTS/dcface/e:10_spatial_dim:5_bias:0.0_casia_ir50_08-29_0/checkpoints/last.ckpt'  # (BATCH_GPU=16, BFM_ID_LAMBDA=0.0, BFM_EXP_LAMBDA=0.01, BFM_POSE_LAMBDA=0.01, DATALOADER_DETERMINISTIC=true)



python src/train_with_BFM_consistency_constraints.py \
        prefix=e:10_spatial_dim:5_bias:0.0_casia_ir50 \
        datamodule.total_gpu_batch_size=$BATCH_GPU \
        datamodule=$DATA_MODULE \
        datamodule.deterministic=$DATALOADER_DETERMINISTIC \
        lightning.max_epochs=10 \
        recognition=casia_ir50 \
        recognition_eval=default \
        reconstruction=bfm \
        optimizer.optimizer_model.lr=1e-04 \
        datamodule.img_size=112 \
        model=default \
        trainer.sampler.variance_type='learned_range' \
        model.unet_config.params.gradient_checkpointing=false \
        model.unet_config.params.condition_type=crossatt_and_stylemod \
        model.unet_config.params.condition_source=patchstat_spatial_and_image \
        label_mapping=v4 \
        external_mapping=v4_dropout \
        external_mapping.dropout_prob=0.3 \
        losses.identity_consistency_loss_lambda=0.05 \
        losses.identity_consistency_loss_source=mix \
        losses.identity_consistency_loss_version=simple_mean \
        losses.identity_consistency_mix_loss_version=polynomial_1 \
        losses.identity_consistency_loss_center_source=id_image \
        losses.identity_consistency_loss_time_cut=0.0 \
        losses.identity_consistency_loss_weight_start_bias=0.0 \
        losses.bfm_consistency_id_lambda=$BFM_ID_LAMBDA \
        losses.bfm_consistency_exp_lambda=$BFM_EXP_LAMBDA \
        losses.bfm_consistency_pose_lambda=$BFM_POSE_LAMBDA \
        losses.spatial_consistency_loss_lambda=0.0 \
        losses.latent_mixup_loss_lambda=0.0 \
        losses.face_contour_loss_lambda=0.0 \
        datamodule.trim_outlier=true \
        model.unet_config.freeze_unet=false \
        callbacks.model_checkpoint.save_top_k=1 \
        external_mapping.spatial_dim=5 \
        ckpt_path=$CKPT_PATH
