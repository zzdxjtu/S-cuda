#!/bin/bash
export ANTSPATH=${HOME}/bin/ants/bin/
export PATH=${ANTSPATH}:$PATH
DATA_PATH="/extracephonline/medai_data2/zhengdzhang/eyes/qikan/"
PATH_TO_SAVE_A=$DATA_PATH"correction_scratch/n1/clean_list/level_0.5-0.7/noise_labels_0.9/sample_selected_0.1_2.txt"
NOISE_TO_SAVE_A=$DATA_PATH"correction_scratch/n1/noise_list/level_0.5-0.7/noise_labels_0.9/noise_selected_0.3.txt"
PATH_TO_SAVE_B=$DATA_PATH"correction_scratch/n2/clean_list/level_0.5-0.7/noise_labels_0.9/selected_sample_0.1_2.txt"
NOISE_TO_SAVE_B=$DATA_PATH"correction_scratch/n2/noise_list/level_0.5-0.7/noise_labels_0.9/noise_sample_0.3.txt"
PATH_TO_RESTORE_A=$DATA_PATH"correction_scratch/n1/snapshots/level_0.5-0.7/noise_labels_0.9/select_0.1/eyes_40"
PATH_TO_RESTORE_B=$DATA_PATH"correction_scratch/n2/weights/level_0.5-0.7/noise_labels_0.9/select_0.1/Generator/generator_40.h5"
MODEL_TO_SAVE_A="/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction_scratch/n1/snapshots/level_0.5-0.7/noise_labels_0.9/select_0.1_2"
MODEL_TO_SAVE_B="/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction_scratch/n2/weights/level_0.5-0.7/noise_labels_0.9/select_0.1_2"
NOISE_TO_CHOOSE="/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction_scratch/update_list/level_0.5-0.7/noise_labels_0.9/select_0.3/jiao.txt"

python $DATA_PATH/Network-1/DA_weight.py --remember_rate 0.1 --noise_rate 0.3 --save_selected_samples $PATH_TO_SAVE_A --restore-from $PATH_TO_RESTORE_A  --noise_selected_samples $NOISE_TO_SAVE_A &
pid1=$!
python $DATA_PATH/Network-2/train_refuge.py --remember_rate 0.1 --noise_rate 0.3 --save_selected_samples $PATH_TO_SAVE_B --restore-from $PATH_TO_RESTORE_B --noise_selected_samples $NOISE_TO_SAVE_B &
pid2=$!
wait $pid1
wait $pid2
echo 'Iteration 1 done'

sudo python $DATA_PATH/Network-1/DA_weight.py --load_selected_samples $PATH_TO_SAVE_B  --restore-from $PATH_TO_RESTORE_A --snapshot-dir $MODEL_TO_SAVE_A &
pid3=$!
sudo python $DATA_PATH/Network-2/train_refuge.py --load_selected_samples $PATH_TO_SAVE_A  --restore-from $PATH_TO_RESTORE_B --weight-root $MODEL_TO_SAVE_B &
pid4=$!
wait $pid3
wait $pid4
echo 'Iteration 2 done'

sudo python $DATA_PATH/choose.py --list_a $NOISE_TO_SAVE_A --list_b $NOISE_TO_SAVE_B --list_noise_save $NOISE_TO_CHOOSE &
pid5=$!
wait $pid5
echo 'Iteration 3 done'


