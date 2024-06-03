import os, sys
import numpy as np
import glob
from argparse import ArgumentParser
import shutil


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/datasets2/2nd_frcsyn_cvpr2024/datasets/synthetic/dcface/dcface_0.5m_oversample_xid/imgs')
    parser.add_argument('--img_ext', type=str, default='.png')
    parser.add_argument('--str_pattern', type=str, default='')
    parser.add_argument('--output_path', type=str, default='/datasets2/bjgbiesseck/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids')
    args = parser.parse_args()
    return args


def find_all_files(folder_path, extensions=['.jpg', '.png']):
    image_paths = []
    num_found_files = 0
    for root, _, files in os.walk(folder_path):
        for ext in extensions:
            pattern = os.path.join(root, '*' + ext)
            matching_files = glob.glob(pattern)
            image_paths.extend(matching_files)
            num_found_files += len(matching_files)
            print(f'    num_found_files: {num_found_files}', end='\r')
    print('')
    return sorted(image_paths)


def copy_imgs_to_folder(imgs_paths=[], str_pattern='', output_path=''):
    num_copied_files = 0
    for idx_img_path, img_path in enumerate(imgs_paths):
        if str_pattern != '' and str_pattern in img_path:
            print(f'    Copying: {num_copied_files}', end='\r')
            subj_folder = img_path.split('/')[-2]
            subj_output_path = os.path.join(output_path, subj_folder)
            os.makedirs(subj_output_path, exist_ok=True)
            shutil.copy(img_path, subj_output_path)
            num_copied_files += 1
            # sys.exit(0)
    print('')


def main(args):
    if args.output_path == '':
        args.output_path = args.input_path + '_copy'
    else:
        args.output_path = os.path.join(args.output_path, os.path.basename(args.input_path)+'_copy')
    os.makedirs(args.output_path, exist_ok=True)

    imgs_paths = find_all_files(args.input_path, extensions=[args.img_ext])
    # print('imgs_paths:', imgs_paths)

    copy_imgs_to_folder(imgs_paths, args.str_pattern, args.output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)