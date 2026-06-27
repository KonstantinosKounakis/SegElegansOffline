#!/usr/bin/python3

from utils import *

import argparse

parser=argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='Folder with images to analyze')
parser.add_argument('-x','--extension', type=str, help='File extension of your images')
parser.add_argument('-b', '--batch', type=int, default=4, help='Sub-crops to analyze. Default =4, increase to 9 or 16 if your device can handle it.')
parser.add_argument('-p','--processes', type=int, default=1, help='Number of parallel processes to run for analysis. Default =1, increase if your device can handle it.')

args = parser.parse_args()

test_images = args.input
print('Input folder:', test_images)

save_dir = os.path.join(test_images,'Output')
##if not os.path.exists(save_dir):
    ##os.makedirs(save_dir)

print('Output:', save_dir)

end_gray_image = args.extension
print('Extension:', end_gray_image)

batch_crop_img = args.batch

if batch_crop_img<4:
    batch_crop_img = 4

if batch_crop_img>16:
    batch_crop_img = 16

print('Image sub-crops:', batch_crop_img)

N_process = args.processes

if N_process == 0:
    N_process = 1

if N_process > 16:
    N_process = 16

if N_process >= 2:
    Parallel_process = True
else:
    Parallel_process = False

print('Parallel processes to run:', N_process)


checkpoint_BODY = os.path.join('Models', 'Body', 'Checkpoint')
network_BODY = os.path.join(checkpoint_BODY, 'model.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Evaluating with', device)

path_SKELETON = os.path.join(save_dir,'0_SKELETON/')
path_SEGMENTATION = os.path.join(save_dir,'0_SEGMENTATION/')

if not os.path.exists(path_SKELETON):
    os.makedirs(path_SKELETON)

if not os.path.exists(path_SEGMENTATION):
    os.makedirs(path_SEGMENTATION)

list_images = sorted(list_files(test_images, end_gray_image))

with tqdm(total=len(list_images), unit='img') as pbar:
    for name_image in list_images:
        # name_image = list_images[q]
        name_image_ = os.path.splitext(name_image)[0]
        name_image_save = name_image_ + '.bmp'
        path_image_gray = os.path.join(test_images, name_image)

        image_gray = np.asarray(Image.open(path_image_gray))  # read gray image
        if len(image_gray.shape) > 2:
            image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
        h, w = image_gray.shape

        if not os.path.exists(os.path.join(path_SEGMENTATION, name_image_save)):
            image_seg, image_skl = get_image_network(device=device, dir_checkpoint=network_BODY,
                                            in_size=512, image_gray=image_gray, batch_img=batch_crop_img)
            cv2.imwrite(os.path.join(path_SEGMENTATION, name_image_save), image_seg)
            cv2.imwrite(os.path.join(path_SKELETON, name_image_save), image_skl)
        else:
            image_seg = cv2.imread(os.path.join(path_SEGMENTATION, name_image_save))
            image_skl = cv2.imread(os.path.join(path_SKELETON, name_image_save), cv2.IMREAD_GRAYSCALE)
        pbar.update(1)


print('Post processing...')

area_percentage = 60
area_min = 600
kernel_size = 3
angle_value = 20

path_summary_save = '0_summary_results'
path_complete_mask = '1_complete_mask'
path_edge_small_mask = '1_edge_small_mask'
path_overlap_mask = '1_overlap_mask'
path_all_rois_results = '1_all_rois_results'


path_summary_save = os.path.join(save_dir, path_summary_save)
path_complete_mask = os.path.join(save_dir, path_complete_mask)
path_edge_small_mask = os.path.join(save_dir, path_edge_small_mask)
path_overlap_mask = os.path.join(save_dir, path_overlap_mask)

print('Folder complete mask:', path_complete_mask)
print('Folder edge and small mask:', path_edge_small_mask)
print('Folder overlap masks:', path_overlap_mask)
print('Folder save_results:', path_summary_save)

import numpy as np
from skimage import measure
from joblib import Parallel, delayed

args = {'Parallel_process': Parallel_process,
        'path_images': test_images,
        'path_SEGMENTATION': path_SEGMENTATION,
        'path_SKELETON': path_SKELETON,
        'path_complete_mask': path_complete_mask,
        'path_edge_small_mask': path_edge_small_mask,
        'path_overlap_mask': path_overlap_mask,
        'path_summary_save': path_summary_save,
        'area_percentage': area_percentage,
        'area_min': area_min,
        'kernel_size': kernel_size,
        'angle_value': angle_value
        }

def post_processing(args, name_image):
    name_image_ = os.path.splitext(name_image)[0]
    name_image_save = name_image_ + '.bmp'
    seg_path=os.path.join(args['path_SEGMENTATION'], name_image_save)
    image_seg = cv2.imread(seg_path)

    save_path = os.path.join(args['path_summary_save'], name_image_) + '.png'
    if not os.path.exists(save_path):
        skl_path=os.path.join(args['path_SKELETON'], name_image_save)
        image_skl = cv2.imread(skl_path)
        image_skl = (cv2.cvtColor(image_skl, cv2.COLOR_BGR2GRAY) > 0) * 255

        area_min = args['area_min']
        angle_value = args['angle_value']
        kernel_size = args['kernel_size']
        area_percentage = args['area_percentage']

        path_images = args['path_images']
        path_complete_mask = args['path_complete_mask']
        path_edge_small_mask = args['path_edge_small_mask']
        path_overlap_mask = args['path_overlap_mask']
        path_summary_save = args['path_summary_save']

        if not os.path.exists(path_complete_mask):
            try:
              os.makedirs(path_complete_mask)
            except FileExistsError:
              pass

        if not os.path.exists(path_edge_small_mask):
            try:
              os.makedirs(path_edge_small_mask)
            except FileExistsError:
              pass

        if not os.path.exists(path_overlap_mask):
            try:
              os.makedirs(path_overlap_mask)
            except FileExistsError:
              pass

        if not os.path.exists(path_summary_save):
            try:
              os.makedirs(path_summary_save)
            except FileExistsError:
              pass


        # ***************** Improve edge detection ****************************************************
        edge_final = check_edge_worms(image_seg, kernel_size)

        # ***************** none overlappings and overlappings *******************************************
        none_overlappings, overlapping = obtain_overlappings(image_seg, edge_final, kernel_size + 2)
        labels_overlapping = measure.label(overlapping, background=0)
        labels_none_overlapping = measure.label(none_overlappings, background=0)

        # ************************** None-overlappings ***************************************************
        true_overlaps = check_overlapping(labels_overlapping, labels_none_overlapping)
        mask_worms = get_none_overlapping(labels_none_overlapping, true_overlaps, area_min, kernel_size)  # none-overl
        mask_worms_Dims = worms2NDims(mask_worms, kernel_size + 2)  # each dimension is a worm
        results_masks_NO = check_noneOverlapping(mask_worms_Dims, area_percentage)  # Check good/bad masks

        # ************************** overlappings ********************************************************
        mask_overlaps_Dims = overlapping_worms(true_overlaps, mask_worms, labels_overlapping,
                                               labels_none_overlapping, image_skl, area_min,
                                               kernel_size+2, angle_value)

        # ***************************** Save imgs results *****************************************************
        name_image_final = os.path.join(path_summary_save, name_image_) + '.png'
        path_image_gray = os.path.join(path_images, name_image)
        image_gray = imread_image(path_image_gray)  # read gray image
        if len(image_gray.shape) > 2:
            image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
        save_results_mask(name_image_final, image_gray, results_masks_NO, mask_overlaps_Dims, 1)  # RGB
        complete_mask_name = os.path.join(path_complete_mask, name_image)
        save_mask_tif(complete_mask_name, results_masks_NO['worms_good'])
        edge_small_mask_name= os.path.join(path_edge_small_mask,name_image)
        save_mask_tif(edge_small_mask_name, results_masks_NO['worms_bads'])
        overlap_mask_name=os.path.join(path_overlap_mask, name_image)
        save_mask_tif(overlap_mask_name, mask_overlaps_Dims)

warnings.filterwarnings("ignore", message=".*OME series cannot handle discontiguous storage")

if Parallel_process:
    Parallel(n_jobs=N_process, verbose=1, backend='multiprocessing')(
        delayed(post_processing)(args, name_image) for name_image in list_images)
else:
    with tqdm(total=len(list_images), unit='img') as pbar:
        for name_image in list_images:
            post_processing(args, name_image)
            pbar.update(1)


## Save all ROIS together.

path_all_rois_results = os.path.join(save_dir, path_all_rois_results)

from utils import list_files, save_mask_rois
if not os.path.exists(path_all_rois_results):
    os.makedirs(path_all_rois_results)

list_images = sorted(list_files(test_images, end_gray_image))
with tqdm(total=len(list_images), unit='img') as pbar:
    for name_image in list_images:
        path_good_mask = os.path.join(path_complete_mask, name_image)
        print(path_good_mask)
        image_good_mask = read_tiff_mask(path_good_mask)
        path_overlap_mask_mask=os.path.join(path_overlap_mask, name_image)
        print(path_overlap_mask_mask)
        image_overlap_mask = read_tiff_mask(path_overlap_mask_mask)
        path_reject_mask=os.path.join(path_edge_small_mask, name_image)
        print(path_reject_mask)
        image_reject_mask = read_tiff_mask(path_reject_mask)
        name_image_ = os.path.splitext(name_image)[0]
        image_total_mask=np.concatenate((image_good_mask,image_overlap_mask,image_reject_mask),axis=0)
        name_zip_save = name_image_ + '.zip'
        path_zip_save = os.path.join(path_all_rois_results, name_zip_save)
        save_mask_rois(path_zip_save, image_total_mask)
        pbar.update(1)

folder_rois_results = '2_curated_rois_results'
folder_rois_results = os.path.join(save_dir, folder_rois_results)

if not os.path.exists(folder_rois_results):
    os.makedirs(folder_rois_results)

list_images = sorted(list_files(path_complete_mask, end_gray_image))
with tqdm(total=len(list_images), unit='img') as pbar:
    for name_image in list_images:
        path_good_mask = os.path.join(path_complete_mask, name_image)
        print(path_good_mask)
        image_good_mask = read_tiff_mask(path_good_mask)
        path_overlap_mask_mask = os.path.join(path_overlap_mask, name_image)
        print(path_overlap_mask_mask)
        image_overlap_mask = read_tiff_mask(path_overlap_mask_mask)
        name_image_ = os.path.splitext(name_image)[0]
        image_total_mask = np.concatenate((image_good_mask, image_overlap_mask), axis=0)
        name_zip_save = name_image_ + '.zip'
        path_zip_save = os.path.join(folder_rois_results, name_zip_save)
        save_mask_rois(path_zip_save, image_total_mask)
        pbar.update(1)

print('All done!')