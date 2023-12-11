import requests
import zipfile
from PIL import Image
import os
import random
import shutil
from tqdm import tqdm
from typing import List, Tuple, Dict


def create_dataset(url: str) -> None:
    """
    downloads the dataset from the specified url and unzips it in the 'dataset' target folder
    :param url: dataset url
    :return: None
    """
    response = requests.get(url)
    dataset_file_name = 'dataset.zip'
    open(dataset_file_name, "wb").write(response.content)
    dest_folder = 'dataset'
    with zipfile.ZipFile(dataset_file_name, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)


def create_train_test_directories(train_folder: str, test_folder: str) -> None:
    """
    creates train and test directories and the inner cats and dogs ones
    :param train_folder: name of train folder
    :param test_folder: name of test folder
    :return: None
    """
    os.makedirs(train_folder + 'cats/', exist_ok=True)
    os.makedirs(train_folder + 'dogs/', exist_ok=True)
    os.makedirs(test_folder + 'cats/', exist_ok=True)
    os.makedirs(test_folder + 'dogs/', exist_ok=True)


def train_test_split(imgs_paths: List[str], train_split: float, train_dir_path: str, test_dir_path: str) -> List[str]:
    """
    Copies images to train or test folder randomly, keeping the same class
    :param imgs_paths:  list of imgs path
    :param train_split: train set percentage
    :param train_dir_path: train folder path
    :param test_dir_path: test folder path
    :return: List of imgs path failed for this operation
    """
    num_train_imgs = int(len(imgs_paths) * train_split)
    random.seed(42)  # for reproducibility purposes
    random.shuffle(imgs_paths)
    fails = []
    for i, img_path in tqdm(enumerate(imgs_paths), total=len(imgs_paths),
                            desc=f"{imgs_paths[0].split('/')[-2]} splitting"):
        try:
            # img_reshaped = Image.open(img_path).resize(target_resolution)
            # if i < num_train_imgs: img_reshaped.save(f"{train_dir_path}{img_path.split('/')[-1]}")
            # else: img_reshaped.save(f"{test_dir_path}{img_path.split('/')[-1]}")
            if i < num_train_imgs:
                shutil.copy(img_path, f"{train_dir_path}{img_path.split('/')[-1]}")
            else:
                shutil.copy(img_path, f"{test_dir_path}{img_path.split('/')[-1]}")
        except:
            fails.append(img_path)
    return fails


def get_images_stats(dir: str) -> Tuple[List[str], Dict[Tuple[int, int], List[str]]]:
    """
    given a directory, returns list of images, resolutions, and corrupted file names
    :param dir: directory
    :return: 3 lists: images path, resolutions, corrupted fil names
    """
    resolutions = {}
    imgs_file_names = os.listdir(dir)
    file_names = []
    corrupted_file_names = []
    for img in imgs_file_names:
        try:
            composed_path = os.path.join(dir, img)
            img_size = Image.open(composed_path).size
            if img_size not in resolutions: resolutions[img_size] = 0
            resolutions[img_size] += 1
            file_names.append(composed_path)
        except:
            corrupted_file_names.append(composed_path)
    return file_names, resolutions, corrupted_file_names


def remove_corrupted_images(corrupted_imgs: List[str]) -> None:
    """
    removes specified images
    :param corrupted_imgs: list of paths
    :return: None
    """
    for cor_img in corrupted_imgs:
        if cor_img[-3:] == 'jpg':
            os.remove(cor_img)


def average_resolution(resolutions: Dict[Tuple[int, int], int]) -> Dict[str, int]:
    """
    calculate average resolution
    :param resolutions: dict{(width: int, height:int): count:int} key is (width,height) pixel measure,
    count is how many images have that key resolution
    :return: average resolution dict {'width': int, 'height': int}
    """
    avg = {'width': 0, 'height': 0}
    counter = 0
    for res in resolutions:
        count = resolutions[res]
        counter += count
        avg['width'] += res[0] * count
        avg['height'] += res[1] * count
    avg['width'] //= counter
    avg['height'] //= counter
    return avg
