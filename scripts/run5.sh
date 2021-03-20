#!/bin/bash
export ANTSPATH=${HOME}/bin/ants/bin/
export PATH=${ANTSPATH}:$PATH
DATA_PATH="../"
PATH_TO_SAVE_A=$DATA_PATH"clean/n1/clean_list/level_0.5-0.7/noise_labels_0.5/clean_selected_0.5.txt"
NOISE_TO_SAVE_A=$DATA_PATH"noisy/n1/noise_list/level_0.5-0.7/noise_labels_0.5/noise_selected_0.5.txt"
PATH_TO_SAVE_B=$DATA_PATH"clean/n2/clean_list/level_0.5-0.7/noise_labels_0.5/clean_selected_0.5.txt"
NOISE_TO_SAVE_B=$DATA_PATH"noisy/n2/noise_list/level_0.5-0.7/noise_labels_0.5/noise_selected_0.5.txt"
PATH_TO_RESTORE_A=$DATA_PATH"model/n1/level_0.5-0.7/noise_labels_0.5/select_0.4/eyes_160"
PATH_TO_RESTORE_B=$DATA_PATH"model/n2/level_0.5-0.7/noise_labels_0.5/select_0.4/Generator/generator_160.h5"
MODEL_TO_SAVE_A=$DATA_PATH"model/n1/level_0.5-0.7/noise_labels_0.5/select_0.5"
MODEL_TO_SAVE_B=$DATA_PATH"model/n2/level_0.5-0.7/noise_labels_0.5/select_0.5"
NOISE_TO_CHOOSE=$DATA_PATH"update_list/level_0.5-0.7/noise_labels_0.5/select_0.5/intersection.txt"
NOISE_TO_CORRECT=$DATA_PATH"update_list/level_0.5-0.7/noise_labels_0.5/select_0.4/intersection.txt"
SAVE_CORRECTION_LABEL=$DATA_PATH"dataset/source/level_0.5-0.7/noise_labels_0.5_new"
DEVKIT_DIR=$DATA_PATH"update_list/level_0.5-0.7/noise_labels_0.5/select_0.5"
POSAL_SAVE_PATH=$DATA_PATH"disc_small/source/level_0.5-0.7/noise_mask_0.5_new/"

sudo python $DATA_PATH/Network-1/SSL-adapt.py --data-list-target $NOISE_TO_CORRECT --restore-from $PATH_TO_RESTORE_A --save $SAVE_CORRECTION_LABEL --devkit_dir $DEVKIT_DIR &
pid1=$!
sudo python $DATA_PATH/Network-2/SSL-adapt.py --list_path $NOISE_TO_CORRECT --load_from $PATH_TO_RESTORE_B --data_save_path $POSAL_SAVE_PATH &
pid2=$!
wait $pid1
wait $pid2
echo 'Iteration 1 done'

sudo python $DATA_PATH/Network-1/DA_weight.py --remember_rate 0.5 --noise_rate 0.5 --save_selected_samples $PATH_TO_SAVE_A --restore-from $PATH_TO_RESTORE_A --noise_selected_samples $NOISE_TO_SAVE_A &
pid3=$!
sudo python $DATA_PATH/Network-2/train_refuge.py --remember_rate 0.5 --noise_rate 0.5 --save_selected_samples $PATH_TO_SAVE_B --restore-from $PATH_TO_RESTORE_B --noise_selected_samples $NOISE_TO_SAVE_B &
pid4=$!
wait $pid3
wait $pid4
echo 'Iteration 2 done'

sudo python $DATA_PATH/Network-1/DA_weight.py --load_selected_samples $PATH_TO_SAVE_B  --restore-from $PATH_TO_RESTORE_A --snapshot-dir $MODEL_TO_SAVE_A &
pid5=$!
sudo python $DATA_PATH/Network-2/train_refuge.py --load_selected_samples $PATH_TO_SAVE_A  --restore-from $PATH_TO_RESTORE_B --weight-root $MODEL_TO_SAVE_B &
pid6=$!
wait $pid5
wait $pid6
echo 'Iteration 3 done'

sudo python $DATA_PATH/choose.py --list_a $NOISE_TO_SAVE_A --list_b $NOISE_TO_SAVE_B --list_noise_save $NOISE_TO_CHOOSE &
pid7=$!
wait $pid7
echo 'Iteration 4 done'

