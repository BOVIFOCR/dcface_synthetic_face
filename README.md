# BOVIFOCR Instructions

## 1. Main requirements
- Python==3.9
- CUDA==11.2
- numpy==1.23.1
- mxnet==1.9.1
- torch==2.3.0
- torchvision==0.18.0
- pytorch-lightning==1.7.1
- opencv-python==4.9.0.80

## 2. Create environment
```
CONDA_ENV=dcface_synthetic_face
conda create -y --name $CONDA_ENV python=3.9
conda activate $CONDA_ENV

conda env config vars set CUDA_HOME="/usr/local/cuda-11.2"; conda deactivate; conda activate $CONDA_ENV
conda env config vars set LD_LIBRARY_PATH="$CUDA_HOME/lib64"; conda deactivate; conda activate $CONDA_ENV
conda env config vars set PATH="$CUDA_HOME:$CUDA_HOME/bin:$LD_LIBRARY_PATH:$PATH"; conda deactivate; conda activate $CONDA_ENV
```

## 3. Clone this repository and install requirements
```
cd ~
git clone https://github.com/BOVIFOCR/dcface_synthetic_face.git
cd dcface_synthetic_face
./install.sh
```
* If it fails, try to run `pip3 install -r requirements.txt` and download model weights from the [link](https://drive.google.com/drive/folders/1ePqFN2eDo0l31aQOkW_U5Mcrlcl83j19?usp=share_link)


</br>

#### - Creating Synthetic ID Images (New Subjects)
```
cd dcface/stage1/unconditional_generation
python unconditional_sampling.py \
      --attention_resolutions 16 \
      --class_cond False \
      --diffusion_steps 1000 \
      --num_samples 16 \
      --batch_size 8 \
      --image_size 256 \
      --learn_sigma True \
      --noise_schedule linear \
      --num_channels 128 \
      --num_head_channels 64 \
      --num_res_blocks 1 \
      --resblock_updown True \
      --use_fp16 False \
      --use_scale_shift_norm True \
      --timestep_respacing 100 \
      --down_N 32 \
      --range_t 20 \
      --save_dir unconditional_samples
```
* 16 new images will be saved into the folder `dcface/stage1/unconditional_generation/unconditional_samples`


</br>

#### - Stylize images

```
cd dcface/src
python synthesis.py --id_images_root sample_images/id_images/sample_57.png --style_images_root sample_images/style_images/woman
```
* The image `sample_57.png` will be stylized from images `sample_5.png, sample_14.png, sample_20.png, sample_45.png, sample_63.png, sample_69.png` and saved into the folder `dcface/generated_images/dcface_3x3/id:sample_57/sty:list_woman`

</br></br></br>


# DCFace: Synthetic Face Generation with Dual Condition Diffusion Model
Official repository for the paper
DCFace: Synthetic Face Generation with Dual Condition Diffusion Model (CVPR 2023).

- Arxiv: [https://arxiv.org/abs/2304.07060](https://arxiv.org/abs/2304.07060)
- Main paper: [main.pdf](assets/main.pdf)
- Supplementary: [supp.pdf](assets/supp.pdf)

### Pipepline
![Demo](assets/pipeline2.gif)
### ID Consistent Generation
![Demo](assets/pipeline.gif)


### Installation

1. one-liner : `install.sh`

If above fails, then
```
pip install -r requirements.txt
# and download model weights from the link below
```

## Image generation

### Sample Code with given example images
We provide the sample code to generate images with the pretrained weights. 
The sample aligned images are provided in the repository.


- Download the pretrained weights from the [link](https://drive.google.com/drive/folders/1ePqFN2eDo0l31aQOkW_U5Mcrlcl83j19?usp=share_link)
- Place the `pretrained_models` directory under `dcface` (same level as `src`)
- Run
```
cd dcface/src
python synthesis.py --id_images_root sample_images/id_images/sample_57.png --style_images_root sample_images/style_images/woman
```

### Optional 
One can also generate new subject images and prepare custom style images.

#### 1. Creating ID Images (New Subjects)
Unconditional ID image generation is done in `dcface/stage1/unconditional_generation`
Take a look at the `README.md` in that directory for instructions on how to generate new ID images.

#### 2. Creating Style Images
Any aligned images can serve as style images. We provive some sample images in `sample_images/style_images` directory.
For anyone who wants to use their own style images, one should align the images first.
Take a look at the `README.md` in `dcface/stage1/style_bank` directory for instructions on how to align images.

#### 3. Using 1. and 2. to generate dataset

Assuming that you followed 1. and 2. you will have an `id_image` and `style_images` directory.
For the sake of explaination, let's say
- ID image is `<Project_root>/dcface/stage1/unconditional_generation/unconditional_samples_aligned/00011.png`
- Style directory is `<Project_root>/dcface/stage1/style_bank/style_images/raw_aligned`
Then to combine these run by pointing at these paths,
```
cd dcface/src
python synthesis.py \
        --id_images_root <Project_root>/dcface/stage1/unconditional_generation/unconditional_samples_aligned/00011.png \
        --style_images_root <Project_root>/dcface/stage1/style_bank/style_images/raw_aligned
```
The result will be saved at `<Project_root>/dcface/generated_images/`


### Training generator

#### Dataset preparation
- Download casia webface dataset from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
    - Place it under `$DATA_ROOT` (ex: `/data/`). 
    - ex: `/data/faces_webface_112x112`

#### Download model weights
- Download all pretrained weights from the [link](https://drive.google.com/drive/folders/1ePqFN2eDo0l31aQOkW_U5Mcrlcl83j19?usp=share_link)
- Place the `pretrained_models` directory under `dcface` (same level as `src`)

#### Run
```
cd dcface
bash src/scripts/train.sh
```


### Dataset Release
DCFace synthetic dataset can be downloaded from [link](https://drive.google.com/drive/folders/1bbG2P3pz81ujj-Ss1mOLol3qnQhc4nBJ?usp=sharing)
- [dcface_0.5m_oversample_xid.zip](https://drive.google.com/file/d/1z8pN_UrERxOZ-k86ZHWCFwCB0EYdTw4P/view?usp=share_link) 
- [dcface_1.2m_oversample_xid.zip](https://drive.google.com/file/d/1veiWBFoAASo_dPzq0I-MasueOuGIfDx4/view?usp=share_link) 

The format of the downloaded file is in `rec` format. 
- you can convert it to `png` using the script. 
- rec file will be useful for the face recognition training script provided in the repository. (to be released soon)

```bash
cd dcface/convert
python record.py --rec_path <path_to_rec_file> --save_path <path_to_save_png>
# ex
# <path_to_rec_file> : dcface_0.5m_oversample_xid/record
# <path_to_save_png> : dcface_0.5m_oversample_xid/images
```

### Training Face Recognition Model

- to be released soon
