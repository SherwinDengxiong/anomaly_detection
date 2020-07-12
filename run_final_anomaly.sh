#! /bin/bash/

rm -rf test_result *.pc testing_feature_multi testing_feature_single result data/*.pkl

#conda create -n cityscene python=3.7
#source activate cityscene
#pip install -r ./requirements.txt

#DATA_PATH=/home/sherwin/dataset/cityscene_test
DATA_PATH=$1

python -W ignore find_single_testing_feature.py $DATA_PATH
python -W ignore find_multi_testing_feature.py $DATA_PATH
python -W ignore generate_txt_task1.py $DATA_PATH
python -W ignore MIL_generate_proposal_train.py
python -W ignore pgcn_test.py thumos14 results/citytrainRGB.pth.tar test_result -j1
python -W ignore eval_detection_results.py thumos14 test_result  --nms_threshold 0.35
python -W ignore generate_csv_task2.py

