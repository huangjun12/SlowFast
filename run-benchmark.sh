#remove exit checkpoint firstly
#rm checkpoints/* -rf 

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

start_time=$(date +%s)

python tools/run_net.py \
       --cfg SLOWFAST.yaml \
       DATA.PATH_TO_DATA_DIR data/all \
       NUM_GPUS 8 \
       LOG_PERIOD 1

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "8card bs=64, 1 epoch, 400 class, preciseBN 200 iter time is $(($cost_time/60))min $(($cost_time%60))s"
