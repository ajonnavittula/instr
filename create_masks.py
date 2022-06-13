import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
import cv2

def coco2mask():

    # Specify path of .json file, images and masks directory
    path = 'ABB_stereo.json'
    imgs_path = '/home/ws1/instr/SPS_dataset/left_rgb'
    mask_path = '/home/ws1/instr/SPS_dataset/gt'

    coco = COCO(path)

    img_ids_0 = coco.getImgIds(catIds=[])

    # Get total number of categories present in dataset to scale labels intensity
    all_cats = coco.getCatIds()
    nm_cats = len(all_cats)
    # Label 0 is left out for background, starting at intensity 10
    labels = np.linspace(10, 20, nm_cats).astype(int)

    images_path = [imgs_path]
    images_ids = next(os.walk(images_path[0]))[2]

    j  = 1
    for k in img_ids_0:
        img = coco.imgs[k]
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        anns_img = np.zeros((img['height'],img['width']))

        i = 1;

        for ann in anns:

            # This will generate a fixed label for a fixed category ID (i.e. semantic segmentation)
            lbl = labels[all_cats.index(ann['category_id'])]
            anns_img = np.maximum(anns_img,coco.annToMask(ann)*lbl)

            # This will generate a new label for every object (i.e. instance segmentation)
            # anns_img = np.maximum(anns_img,coco.annToMask(ann)*i)
            # i = i + 1

        # Change if you want the naming to be different
        j = j + 1
        # print(j, len(images_ids))
        msk_path = mask_path + '/' + coco.loadImgs(k)[0]['file_name']
        cv2.imwrite(msk_path, anns_img.astype(np.uint8))

if __name__ == "__main__":
    coco2mask()
