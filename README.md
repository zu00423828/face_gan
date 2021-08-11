preprocess data

python data_preprocess/preprocess.py --input_dir $data_root --output_dir $out_dir  -t 20  -c 0.5  -color True

split train val data

python data_preprocess/split.py --input_dir $data_root --output_dir $data_root --train_frac 0.8

train 

python face_reenactor_train.py --input_dir /home/yuan/hdd/avspeech_preprocess/preprocess4/ --resize 256 --backward_ratio 2 --batch_size 8 --max_epoch 200 --out_checkpoint avspeech_part_test11 --log avspeech_part_test11
