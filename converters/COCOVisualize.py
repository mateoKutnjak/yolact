import json
import sys
import imageio
import matplotlib.pyplot as plt
import cv2
import random


def search_images_by_id(_id):
    for _ in valid['images']:
        if _['id'] == _id:
            return _


def search_categories_by_id(_id):
    for _ in valid['categories']:
        if _['id'] == _id:
            return _


def plot_polygon(mask, polygons):
    plt.imshow(mask)

    for polygon in polygons:
        plt.scatter(polygon[0::2], polygon[1::2], s=2)
    plt.show()


with open(sys.argv[1], 'r') as f:
    valid = json.load(f)

for ann in valid['annotations']:
    if random.random() < 0.02:

        ann_images = search_images_by_id(ann['image_id'])
        ann_category = search_categories_by_id(ann['category_id'])

        bbox = list(map(int, ann['bbox']))

        rgb = imageio.imread(sys.argv[1] + '/../../images/' + str(ann_images['file_name']))
        rgb = cv2.rectangle(rgb, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 5)
        plot_polygon(rgb, ann['segmentation'])

