export CUDA_VISIBLE_DEVICES=0
python main.py --phase test --dataset_name oil-chinese --lambda_A 500.0 --lambda_B 500.0 --epoch 100  #

python main.py --phase test --dataset_name facades --lambda_A 1000.0 --lambda_B 1000.0  --epoch 200 #

python main.py --phase test --dataset_name sketch-photo --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100 #

python main.py --phase test --dataset_name day-night --lambda_A 1000.0 --lambda_B 1000.0 --epoch 1000 --GAN_type wgan    #
