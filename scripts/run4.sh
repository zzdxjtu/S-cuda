#!/bin/bash
export ANTSPATH=${HOME}/bin/ants/bin/
export PATH=${ANTSPATH}:$PATH
DATA_PATH="/extracephonline/medai_data2/zhengdzhang/eyes/qikan/"
PATH_TO_SAVE_A=$DATA_PATH"correction/n1/clean_list/level_0.5-0.7/noise_labels_0.1/sample_selected_0.7_new.txt"
NOISE_TO_SAVE_A=$DATA_PATH"correction/n1/noise_list/level_0.5-0.7/noise_labels_0.1/noise_selected_0.1_4.txt"
PATH_TO_SAVE_B=$DATA_PATH"correction/n2/clean_list/level_0.5-0.7/noise_labels_0.1/selected_sample_0.7_new.txt"
NOISE_TO_SAVE_B=$DATA_PATH"correction/n2/noise_list/level_0.5-0.7/noise_labels_0.1/noise_sample_0.1_4.txt"
PATH_TO_RESTORE_A=$DATA_PATH"correction/n1/snapshots/level_0.5-0.7/noise_labels_0.1/select_0.5_new/eyes_200"
PATH_TO_RESTORE_B=$DATA_PATH"correction/n2/weights/level_0.5-0.7/noise_labels_0.1/select_0.5_new/Generator/generator_200.h5"
MODEL_TO_SAVE_A="/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/n1/snapshots/level_0.5-0.7/noise_labels_0.1/select_0.7_new"
MODEL_TO_SAVE_B="/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/n2/weights/level_0.5-0.7/noise_labels_0.1/select_0.7_new"
NOISE_TO_CHOOSE="/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/update_list/level_0.5-0.7/noise_labels_0.1/select_0.1_4/jiao.txt"
NOISE_TO_CORRECT="/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/update_list/level_0.5-0.7/noise_labels_0.1/select_0.1_3/jiao.txt"
SAVE_CORRECTION_LABEL="/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/new-dataset/level_0.5-0.7/noise_labels_0.1_new"
DEVKIT_DIR="/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/update_list/level_0.5-0.7/noise_labels_0.1/select_0.1_3"
POSAL_SAVE_PATH="/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/new-dataset_2/level_0.5-0.7/noise_mask_0.1_new/"

sudo python $DATA_PATH/Network-1/SSL-adapt.py --data-list-target $NOISE_TO_CORRECT --restore-from $PATH_TO_RESTORE_A --save $SAVE_CORRECTION_LABEL --devkit_dir $DEVKIT_DIR &
pid1=$!
sudo python $DATA_PATH/Network-2/SSL-adapt.py --list_path $NOISE_TO_CORRECT --load_from $PATH_TO_RESTORE_B --data_save_path $POSAL_SAVE_PATH
pid2=$!
wait $pid1
wait $pid2
echo 'Iteration 1 done'

python $DATA_PATH/Network-1/DA_weight.py --remember_rate 0.7 --noise_rate 0.1 --save_selected_samples $PATH_TO_SAVE_A --restore-from $PATH_TO_RESTORE_A --noise_selected_samples $NOISE_TO_SAVE_A &
pid3=$!
python $DATA_PATH/Network-2/train_refuge.py --remember_rate 0.7 --noise_rate 0.1 --save_selected_samples $PATH_TO_SAVE_B --restore-from $PATH_TO_RESTORE_B --noise_selected_samples $NOISE_TO_SAVE_B &
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

