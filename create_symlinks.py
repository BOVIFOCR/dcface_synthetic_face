import os, sys
import argparse
import re
import random


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def natural_sort(data_list=['']):
    data_list.sort(key=natural_keys)
    return data_list


def find_image_files(folder_path):
    img_ext = ['.jpg', '.jpeg', '.png']
    found_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_lower = file.lower()
            for ext in img_ext:
                if file_lower.endswith(ext):
                    found_files.append(os.path.join(root, file))
                    break
    sorted_paths = natural_sort(found_files)
    return sorted_paths


def get_unique_classes_paths(imgs_paths):
    classes_paths = [os.path.dirname(img_path) for img_path in imgs_paths]
    unique_classes_paths = list(set(classes_paths))
    sorted_unique_classes_paths = natural_sort(unique_classes_paths)
    return sorted_unique_classes_paths


def create_symlinks(input_path='', orig_class_paths=[''], output_path=''):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for idx_class, orig_class_path in enumerate(orig_class_paths):
        print(f'{idx_class+1}/{len(orig_class_paths)}', end='\r')
        symlink = orig_class_path.replace(input_path, output_path)
        # print(f'idx_class {idx_class} - class_path: {class_path} - symlink: {symlink}')
        os.symlink(orig_class_path, symlink)
    print('')

    '''
    for root, dirs, files in os.walk(input_path):
        # Compute the relative path from the input path
        relative_path = os.path.relpath(root, input_path)
        # Create the corresponding directory in the output path
        target_dir = os.path.join(output_path, relative_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        for file in files:
            # Full path of the original file
            original_file = os.path.join(root, file)
            # Full path of the symlink
            symlink = os.path.join(target_dir, file)
            # Create the symlink
            if not os.path.exists(symlink):
                os.symlink(original_file, symlink)
                print(f"Created symlink: {symlink} -> {original_file}")
    '''





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create symbolic links of dataset files.')
    parser.add_argument('--input-path', type=str, help='The input dataset path.')
    parser.add_argument('--output-path', type=str, help='The output dataset path.')
    parser.add_argument('--num-classes', type=int, default=1000, help='')   # -1 means all classes
    parser.add_argument('--selection-method', type=str, default='random', help='random,sequential')

    args = parser.parse_args()

    if not os.path.isdir(args.input_path):
        print(f"Error: {args.input_path} is not a valid directory")
        sys.exit(1)

    print('Searching images files...')
    imgs_paths = find_image_files(args.input_path)
    unique_class_paths = get_unique_classes_paths(imgs_paths)
    # print('imgs_paths:', imgs_paths)
    # print('unique_class_paths:', unique_class_paths)

    selected_class_paths = unique_class_paths
    if args.num_classes > 0:
        print('Selecting classes...')
        if args.selection_method == 'random':
            selected_class_paths = random.sample(unique_class_paths, args.num_classes)

        elif args.selection_method == 'sequential':
            selected_class_paths = unique_class_paths[:args.num_classes]
    # print('selected_class_paths:', selected_class_paths)

    print('Creating symbolic links...')
    create_symlinks(args.input_path, selected_class_paths, args.output_path)
    # print("Symlinks creation completed successfully.")
