This project is for anomaly detection
task 1 is to provide score for each frame in the video
task 2 is to generate a csv file which contain anomaly cluster and anomaly section

this code is based on python 3.7
to setup the environment, please follow the instruction.

set the working directory as the folder that contain the code.

step 1
conda create -n cityscene python=3.7
step 2
source activate cityscene

step 3
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html 

step 4 
pip install -r ./requirements.txt

step 5 run the bash file
bash run_final_anomaly.sh <you_testset_path>

the result  task1 and task2 will show in the ./result/ folder
