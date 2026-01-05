SAVEPATH=$1
DATADIR=$2
CHECKPOINTMEDSAM=$3

METHOD=UnCoL
DATASET=Pancreas-CT
NUMCLASSES=2

LABELEDNUM=12
GPU=0
FOLDNUM=0
BATCHSIZE=4
LABELEDBS=2
CONSISTENCY=0.1

# ########## Stage 1: Data-efficient Pretraining  ########## 
# STAGE=PRETRAIN
# python ./train_3d.py --labeled_num $LABELEDNUM \
#                      --data_dir $DATADIR --dataset $DATASET \
#                      --method $METHOD \
#                      --save_path $SAVEPATH --gpu $GPU --num_classes $NUMCLASSES \
#                      --mode $STAGE --consistency $CONSISTENCY \
#                      --batch_size $BATCHSIZE \
#                      --fold_num $FOLDNUM --checkpoint_sam $CHECKPOINTMEDSAM


# ########## Stage 2: Semi-supervised Fine-tuning ########## 
# CHECKPOINTDIRVNET=$4
# UWEIGHT=0.5
# STAGE=SSL
# python ./train_3d.py --labeled_num $LABELEDNUM \
#                      --data_dir $DATADIR --dataset $DATASET \
#                      --method $METHOD \
#                      --save_path $SAVEPATH --gpu $GPU --num_classes $NUMCLASSES \
#                      --mode $STAGE --consistency $CONSISTENCY \
#                      --batch_size $BATCHSIZE --labeled_bs $LABELEDBS \
#                      --checkpoint_dir $CHECKPOINTDIRVNET \
#                      --fold_num $FOLDNUM  --u_weight $UWEIGHT \
#                      --checkpoint_sam $CHECKPOINTMEDSAM


########## Inference ########## 
CHECKPOINTFINAL=$5
python test_3d.py --data_dir $DATADIR --fold_num $FOLDNUM \
                  --model_path $CHECKPOINTFINAL \
                  --gpu $GPU --num_classes $NUMCLASSES
