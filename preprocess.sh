#!/usr/bin/env bash

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>log_preprocess_quest3_leeds.txt 2>&1

### @TODO Update and uncomment dataset names, frame number, required paths, and conda/venv as required ###

## Results
#DATASET=(
#'flir_dof_mb_greenballmotion_scene1_1'
#'flir_noisy_greenballmotion'
#'flir_noisy_rainbowballmotion'
#'flir_noisy_scleich_2'
#'rainbow_ball_2'
#'nikon_AutoMode'
#'nikon_varying_focus'
#'nikon_varying_ISO'
#)
# Provide a single frame number from the images which exhibit all distortions
#FRAME=('15' '15' '5' '33' '13' '500' '350' '350')


CWD=$(pwd)
#ROOT_DIR=<path/to/data/>
# https://github.com/swz30/Restormer
#DENOISE_DIR=<path/to/Restormer/>
# https://github.com/dasongli1/Shift-Net
#DEMOTION_DIR=<path/to/Shift-Net/>
# https://github.com/lingyanruan/DRBNet
#DEFOCUS_DIR=<path/to/DRBNet/>
# https://github.com/NVIDIA/flownet2-pytorch
#FLOW_DIR=<path/to/flownet2-pytorch/>
# https://github.com/isl-org/MiDaS
#DEPTH_DIR=<path/to/MiDaS/>

### @TODO This script assume the original images for dataset/scene is provided in the $DENOISE_DIR/demo/<DATASET> folder ###

for i in "${!DATASET[@]}"
do
  d=${DATASET[$i]}
  echo DATASET $d
  date
######### 1. RUN DENOISING ###########
  cd $DENOISE_DIR
  echo RUN DENOISING
  conda activate Restormer
  python demo.py --task Real_Denoising --input_dir ./demo/$d --result_dir ./demo/$d --tile 256
  conda deactivate
######### 2. COPY ORIGINAL IMAGES ###########
  from_dir=$DENOISE_DIR"/demo/"$d
  echo Copy input images from
  echo $from_dir
  cd $from_dir
  to_dir=$ROOT_DIR"/original/"$d
  echo to
  echo $to_dir
  mkdir $to_dir
  ls *.png
  mv *.png $to_dir
  frame=$(printf 'frame_%06d.png' $((${FRAME[$i]})))
  echo Copy frame $frame
  cd $to_dir
  cp $frame "../"$d".png"
########## 3. COPY DENOISED IMAGES ##########
  from_dir=$from_dir"/Real_Denoising"
  echo Copy denoised images from
  echo $from_dir
  cd $from_dir
  to_dir=$ROOT_DIR"/denoised/"$d
  echo to
  echo $to_dir
  mkdir $to_dir
  ls *.png
  mv *.png $to_dir
  frame=$(printf 'frame_%06d.png' $((${FRAME[$i]})))
  echo Copy frame $frame
  cd $to_dir
  cp $frame "../"$d".png"
  to_dir=$DEMOTION_DIR"/dataset/custom/blur/"$d
  echo to
  echo $to_dir
  mkdir -p $to_dir
  ls *.png
  cp *.png $to_dir
done
######## 4. RUN MOTION DE-BLUR ###########
date
cd $DEMOTION_DIR"/inference"
echo RUN MOTION DE-BLURRING
conda activate Shift-Net
python test_deblur.py --one_len 2 --save_image
conda deactivate

for i in "${!DATASET[@]}"
do
  d=${DATASET[$i]}
  echo $d
  date
############ 5. COPY DEMOTION BLURRED IMAGES ########
  from_dir=$DEMOTION_DIR"/infer_results/custom_gopro/"$d
  echo Copy Demotion blurred images from
  echo $from_dir
  cd $from_dir
  to_dir=$DEFOCUS_DIR"/input/"$d
  echo to
  echo $to_dir
  mkdir -p $to_dir
  ls *.png
  cp *.png $to_dir
  to_dir=$FLOW_DIR"/data/original/"$d
  echo to
  echo $to_dir
  mkdir -p $to_dir
  ls *.png
  cp *.png $to_dir
  to_dir=$ROOT_DIR"/de_motion_blurred/"$d
  echo to
  echo $to_dir
  mkdir $to_dir
  ls *.png
  mv *.png $to_dir
  frame=$(printf '%03d.png' $((${FRAME[$i]})))
  echo Copy frame $frame
  cd $to_dir
  cp $frame "../"$d".png"
############ 6. RUN FOCUS DE-BLUR #############
  cd $DEFOCUS_DIR
  echo RUN FOCUS DE-BLURRING
  conda activate consistent_ar
  python run.py --net_mode single --eval_data $d --save_images --input input/$d
########### 7. COPY DEFOCUS BLURRED IMAGES ##########
  from_dir=$DEFOCUS_DIR"/results/defocus_deblur/"$d"/single"
  temp_dir=$(ls $from_dir)
  from_dir=$from_dir"/"$temp_dir"/output"
  echo Copy Defocus deblurred images from
  echo $from_dir
  cd $from_dir
  to_dir=$DEPTH_DIR"/input/"$d
  echo to
  echo $to_dir
  mkdir -p $to_dir
  ls *.png
  cp *.png $to_dir
  to_dir=$ROOT_DIR"/de_focus_blurred/"$d
  echo to
  echo $to_dir
  mkdir $to_dir
  ls *.png
  mv *.png $to_dir
  frame=$(printf '%03d.png' $((${FRAME[$i]})))
  echo Copy frame $frame
  cd $to_dir
  cp $frame "../"$d".png"
########### 8. RUN DEPTH ESTIMATION #########
  cd $DEPTH_DIR
  echo RUN DEPTH ESTIMATION
  python run.py --model_type dpt_beit_large_512 --input_path input/$d --output_path output/$d --grayscale
########### 9. COPY DEPTH IMAGES #################
  from_dir=$DEPTH_DIR"/output/"$d
  echo Copy Depth images from
  echo $from_dir
  cd $from_dir
  to_dir=$ROOT_DIR"/depth/"$d
  echo to
  echo $to_dir
  mkdir $to_dir
  ls *.png
  mv *.png $to_dir
  frame=$(printf '%03d.png' $((${FRAME[$i]})))
  echo Copy frame $frame
  cd $to_dir
  cp $frame "../"$d".png"
########## 10. RUN FLOW ESTIMATION #############
  cd $FLOW_DIR
  echo RUN FLOW ESTIMATION
  python main.py --inference --model FlowNet2 --inference_dataset ImagesFromFolder --inference_dataset_root data/original/$d --resume checkpoints/flownet2.pth --save_flow --inference_visualize
  conda deactivate
########## 11. COPY FLOW #######################
  from_dir=$FLOW_DIR"/work/inference/"$d"/run.epoch-0-flow-field"
  echo Copy Optical Flow from
  echo $from_dir
  cd $from_dir
  to_dir=$ROOT_DIR"/flow/"$d
  echo to
  echo $to_dir
  mkdir $to_dir
  ls *.flo
  mv *.flo $to_dir
  frame=$(printf '%06d.flo' $((${FRAME[$i]})))
  echo Copy frame $frame
  cd $to_dir
  cp $frame "../"$d".flo"
  date
done