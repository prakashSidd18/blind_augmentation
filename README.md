# Blind Augmentation: Calibration-free Camera Distortion Model Estimation for Real-time Mixed-reality Consistency

[![Project Page](https://img.shields.io/badge/Project-Page-green?logo=googlechrome&logoColor=green)](https://prakashsidd18.github.io/projects/blind_augmentation/)
[![YouTube Video](https://img.shields.io/badge/YouTube-Video-red?logo=youtube&logoColor=red)](https://youtu.be/YhK5wjmVYjg)
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=red)]() -->


[Siddhant Prakash](https://prakashsidd18.github.io/)<sup>1</sup>, [David R. Walton](https://drwalton.github.io/)<sup>2</sup>, [Rafael K. dos Anjos](https://rafaelkuffner.github.io/)<sup>3</sup>, [Anthony Steed](https://wp.cs.ucl.ac.uk/anthonysteed/)<sup>1</sup>, [Tobias Ritschel](https://www.homepages.ucl.ac.uk/~ucactri/)<sup>1</sup>


<sup>1</sup>University College London, <sup>2</sup>Birmingham City University, <sup>3</sup>University of Leeds<br>


![Teaser](images/teaser.png "Teaser Image")
<div style="text-align: justify"> 
Our method “blindly” estimates a model of noise, motion blur (MB) and depth of field (DoF) from input frames (left), i.e., without
		requiring any known calibration markers / objects. The model can then augment other images with virtual objects that appear visually
		consistent (middle).
    </div>

### Table of Contents
- [Abstract](#abstract) 
- [Dataset, Code & Preprocess](#code)
    - [Dataset Download](#1-dataset-download) 
    - [Install](#2-setup)
    - [Run](#3-running-on-standard-dataset)
    - [Pre-process](#4-preprocess)
- [Real-time Unity Demo](#demo)
- [Citation](#citation)

<a name="abstract"></a>
## Abstract 

<div style="text-align: justify">
Real camera footage is subject to noise, motion blur (MB) and depth of field (DoF). 
In some applications these might be considered distortions to be removed, but in others it is important to model them because it would be ineffective, or interfere with an aesthetic choice, to simply remove them. 
In augmented reality applications where virtual content is composed into a live video feed, we can model noise, MB and DoF to make the virtual content visually consistent with the video. 
Existing methods for this typically suffer two main limitations. 
First, they require a camera calibration step to relate a known calibration target to the specific cameras response. 
Second, existing work require methods that can be (differentiably) tuned to the calibration, such as slow and specialized neural networks. 
We propose a method which estimates parameters for noise, MB and DoF instantly, which allows using off-the-shelf real-time simulation methods from e.g., a game engine in compositing augmented content. 
Our main idea is to unlock both features by showing how to use modern computer vision methods that can remove noise, MB and DoF from the video stream, essentially providing
self-calibration. 
This allows to auto-tune any black-box real-time noise+MB+DoF method to deliver fast and high-fidelity augmentation consistency.
</div>


<a name="code"></a>
## Dataset, Code & Preprocess

We provide code to run optimization and compositing along with dataset for 7 dataset/scenes used in the original paper. 

We have run pre-process on the 7 dataset and provide all intermediary frames along with compositing frames. 

For new dataset/scenes we point to original implementation of off-the-shelf methods which we used in this paper. 

<a name="1-dataset-download"></a>
### 1. Dataset Download

All 7 dataset/scenes can be dowloaded from [here](https://1drv.ms/u/c/a220080a7f502ec5/EcKLUxvCr0FErE8oHVezmxkBULjS83dSpiBC_6LZXcWm4w?e=1rUrYY) (~7GB).

We also provide a smaller dataset (with 2 scenes) to quickly run our single-frame optimization (see [Run](#3-running-on-standard-dataset)) [here](https://1drv.ms/u/c/a220080a7f502ec5/EcKLUxvCr0FErE8oHVezmxkBULjS83dSpiBC_6LZXcWm4w?e=1rUrYY) (~300MB). 


Once downloaded, unzip the zipped file to store data in the following folder structure:

```sh
.
.
├── blur_utils.py
├── composite_module.py
├── data
│   ├── composites
│   │   ├── flir_noisy_greenballmotion
│   │   ├── flir_noisy_rainbowballmotion
│   ├── de_focus_blurred
│   │   ├── flir_noisy_greenballmotion
│   │   ├── flir_noisy_rainbowballmotion
│   ├── de_motion_blurred
│   │   ├── flir_noisy_greenballmotion
│   │   ├── flir_noisy_rainbowballmotion
│   ├── denoised
│   │   ├── flir_noisy_greenballmotion
│   │   ├── flir_noisy_rainbowballmotion
│   ├── depth
│   │   ├── flir_noisy_greenballmotion
│   │   ├── flir_noisy_rainbowballmotion
│   ├── flow
│   │   ├── flir_noisy_greenballmotion
│   │   ├── flir_noisy_rainbowballmotion
│   └── original
│       ├── flir_noisy_greenballmotion
│       ├── flir_noisy_rainbowballmotion
.
.
├── main_optimization_timing.py
├── main_real_time_update_params.py
├── main_single_frame_optimization_composite.py
.
.
```
The `<path/to/dataset/>` is the path to `data` folder. The original captured image are stored in `./data/original/<DATASET>/` folder. 

Pre-process can be run on new dataset by providing captured frames in this folder. See [Pre-process](#4-preprocess) for more details.

Composite frames are dataset/scene specific and are rendered using Blender. A sample Blender file with automated scripts are provided in the `./data/composite/` folder.



<a name="2-setup"></a>
### 2. Install

1. Clone this repo, then `cd` into it.  
2. Create a virtual environment (recommended).

    ```sh
    # Ensure Python version >= 3.10.14
    python -s -m venv .venv_blind_augmentation
    source .venv_blind_augmentation/bin/activate
    ```

2. Install local requirements:

    ```sh
    python -s -m pip install -r requirements.txt
    ```

All requirements to run code will be installed in the virtual environment.

<a name="3-running-on-standard-dataset"></a>
### 3. Run

1. Download the standard dataset (see above) and copy the `<path/to/dataset/>`.
2. Run single frame optimization & compositing using the script `main_single_frame_optimization_composite.py` specifying the `<path/to/dataset/>` as argument. If no path is specified, the script assumes `<path/to/dataset/>` as current folder `./data/`.

    ```sh
    # Run optimization on single-frame and composite
    python main_single_frame_optimization_composite.py <path/to/dataset/>
    ```
3. We also provide script to run multi-frame optimization `main_real_time_update_params.py` and time the optimization code `main_optimization_timing.py`.

    ```sh
    # Run optimization on multi-frames and composite
    python main_real_time_update_params.py <path/to/dataset/>

    # Time the optimization 
    python main_optimization_timing.py <path/to/dataset/>
    ```
The output of the script will be generated in `./output/` folder in the current directory. 

The scripts can generate results from multiple dataset/scenes, which can be toggled ON/OFF as desired within the script. A separate folder inside the output folder for each dataset will be created.
The ouput videos can be found in respective dataset folder as `./output/<DATASET>_result_/*.mp4`. 

For example, results for dataset `flir_noisy_rainbowballmotion` will be stored as `./output/flir_noisy_rainbowballmotion_result_/flir_noisy_rainbowballmotion_result_blurred_composite.mp4` and a comparison with naive approach as `./output/flir_noisy_rainbowballmotion_result_/flir_noisy_rainbowballmotion_result_naive_vs_ours.mp4`.

<a name="4-preprocess"></a>
### 4. Pre-process

<a name="demo"></a>
## Real-time Unity Demo

We provide the Unity project to run our demo [here]() (~2GB).



<a name="citation"></a>
## Citation

```
@Article{prakash2025blind,
  author  = "Prakash, Siddhant and Walton, David R. and dos Anjos, Rafael K. and Steed, Anthony and Ritschel, Tobias",
  title   = "Blind Augmentation: Calibration-free Camera Distortion Model Estimation for Real-time Mixed-reality Consistency",
  journal = "IEEE Transactions on Visualization and Computer Graphics (IEEEVR 2025)",
  year    = "2025",
	  
}
```
