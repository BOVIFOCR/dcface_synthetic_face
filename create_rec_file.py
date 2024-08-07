'''
https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/docs/prepare_custom_dataset.md

# directories and files for yours datsaets
/image_folder
├── 0_0_0000000
│   ├── 0_0.jpg
│   ├── 0_1.jpg
│   ├── 0_2.jpg
│   ├── 0_3.jpg
│   └── 0_4.jpg
├── 0_0_0000001
│   ├── 0_5.jpg
│   ├── 0_6.jpg
│   ├── 0_7.jpg
│   ├── 0_8.jpg
│   └── 0_9.jpg
├── 0_0_0000002
│   ├── 0_10.jpg
│   ├── 0_11.jpg
│   ├── 0_12.jpg
│   ├── 0_13.jpg
│   ├── 0_14.jpg
│   ├── 0_15.jpg
│   ├── 0_16.jpg
│   └── 0_17.jpg
├── 0_0_0000003
│   ├── 0_18.jpg
│   ├── 0_19.jpg
│   └── 0_20.jpg
├── 0_0_0000004


# 0) Dependencies installation
pip install opencv-python
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y

# 1) create train.lst using follow command
python -m mxnet.tools.im2rec --list --recursive train image_folder

# 2) create train.rec and train.idx using train.lst using following command
python -m mxnet.tools.im2rec --num-thread 16 --quality 100 train image_folder

Finally, you will obtain three files: train.lst, train.rec, and train.idx, where train.idx and train.rec are utilized for training.
'''

import os, sys
import argparse
import re
import random



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input-path', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_SYMLINKS_1000CLASS_RANDOM', help='')
    parser.add_argument('--output-path', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112_1000CLASS_RANDOM', help='The output dataset path.')
    args = parser.parse_args()

    if not os.path.isdir(args.input_path):
        print(f"Error: {args.input_path} is not a valid directory")
        sys.exit(1)

    
