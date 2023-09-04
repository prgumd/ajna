cd /media/nitin/Education/PRGEyeOmni/Logs
cd /media/nitin/Education/PRGEyeOmni/Logs2
cd /home/nitin/GIT/prgeyeomni/Code/SimilarityNet
cd /home/nitin/GIT/prgeyeomni/Code/SimilarityNet
./Train.py --LoadCheckPoint=1 --GPUDevice=1 --NumEpochs=200
./Train.py --LoadCheckPoint=1 --GPUDevice=0 --NumEpochs=200 --LossFuncName=SL2-1 --CheckPointPath=/media/nitin/Education/PRGEyeOmni/CheckPoints2/ --LogsPath=/media/nitin/Education/PRGEyeOmni/Logs2/
tensorboard --logdir=./
tensorboard --logdir=./ --port=8008
