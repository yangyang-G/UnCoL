SAVEPATH=$1
DATADIR=$2
CHECKPOINTMEDSAM=$3

METHOD=UnCoL
DATASET=OASIS2D
LABELEDNUM=15
GPU=0
FOLDNUM=0
BATCHSIZE=4
LABELEDBS=2
CONSISTENCY=0.1
NUMCLASSES=5

# ########## Stage 1: Data-efficient Pretraining  ########## 
# STAGE=PRETRAIN
# python ./train_2d.py --labeled_num $LABELEDNUM \
#                      --data_dir $DATADIR --dataset $DATASET \
#                      --method $METHOD \
#                      --save_path $SAVEPATH --gpu $GPU --num_classes $NUMCLASSES \
#                      --mode $STAGE --consistency $CONSISTENCY \
#                      --batch_size $BATCHSIZE \
#                      --fold_num $FOLDNUM --checkpoint_sam $CHECKPOINTMEDSAM


# ########## Stage 2: Semi-supervised Fine-tuning ########## 
# CHECKPOINTDIRUNET=$3
# UWEIGHT=0.5
# STAGE=SSL
# python ./train_2d.py --labeled_num $LABELEDNUM \
#                      --data_dir $DATADIR --dataset $DATASET \
#                      --method $METHOD \
#                      --save_path $SAVEPATH --gpu $GPU --num_classes $NUMCLASSES \
#                      --mode $STAGE --consistency $CONSISTENCY \
#                      --batch_size $BATCHSIZE --labeled_bs $LABELEDBS \
#                      --checkpoint_unet $CHECKPOINTDIRUNET \
#                      --fold_num $FOLDNUM  --u_weight $UWEIGHT \
#                      --checkpoint_sam $CHECKPOINTMEDSAM


########## Inference ########## 
CHECKPOINTFINAL=$4
python ./test_2d.py --data_dir $DATADIR --fold_num $FOLDNUM \
                  --checkpoint_dir $CHECKPOINTFINAL \
                  --gpu $GPU --num_classes $NUMCLASSES 
