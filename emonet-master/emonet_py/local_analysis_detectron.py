import os
import brambox as bb
import lightnet as ln
import numpy as np
import torch
from PIL import Image
from lightnet.models import YoloV3
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from img_resize_preproc import ImgResize
from explanations_emonet import get_visualizations, plot_cam
from pytorch_grad_cam.utils.image import scale_cam_image, scale_accross_batch_and_channels, show_cam_on_image
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import polygons_to_bitmask
from shapely.geometry import Polygon
import pickle


class Detectron:

    def __init__(self, img):
        self.cfg = get_cfg()
        self.img = img
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = 'cpu'
        predictor = DefaultPredictor(self.cfg)
        self.outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(self.outputs["instances"].to("cpu"))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()


    def compare(self):
        heatmap = np.load("cam_grad_dataset/cam_grad_" + file_name + ".npy")  # Assuming heatmap is saved as a NumPy array
        # load image and heatmap
        image = self.img
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB for consistency
        # resize heatmap
        resized_heatmap = resize_heatmap(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
        # generate binary mask from heatmap
        binary_mask = generate_binary_mask(resized_heatmap, 0.5)
        # overlay heatmap on image
        heatmap_overlay = overlay_heatmap_on_image(image_rgb, resized_heatmap)
        # filter polygons by overlap
        filtered_indices = filter_polygons_by_overlap(self.outputs['instances'], resized_heatmap, 0.1)
        # visualize result
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        # visualize without filtering
        visualize_overlay(image, self.outputs['instances'], metadata, heatmap_overlay)
        # visualize with filtering
        visualize_filtered_predictions(image_rgb, self.outputs['instances'], filtered_indices, metadata, heatmap_overlay)


# resize the heatmap
def resize_heatmap(heatmap, target_size):
    return cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)

# generate masks from heatmap
def generate_binary_mask(heatmap, threshold):
    return (heatmap >= threshold).astype(np.uint8)

# overlay heatmap
def overlay_heatmap_on_image(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  # Convert heatmap to RGB
    overlay = cv2.addWeighted(image, alpha, heatmap_colored, 1 - alpha, 0)
    return overlay

# filter polygons by overlap
def filter_polygons_by_overlap(instances, binary_mask, overlap_threshold):
    filtered_indices = []
    for i, mask in enumerate(instances.pred_masks):
        mask = mask.numpy()
        #polygon = mask_to_polygon(mask)
        #polygon_mask = polygon_to_mask(polygon, binary_mask.shape)
        overlap = np.sum(binary_mask * mask) / np.sum(mask)
        if overlap >= overlap_threshold:
            filtered_indices.append(i)
    return filtered_indices


def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [Polygon(contour.reshape(-1, 2)) for contour in contours if contour.size > 0]

def polygon_to_mask(polygon, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for poly in polygon:
        cv2.fillPoly(mask, [np.array(poly.exterior.coords, dtype=np.int32)], 1)
    return mask

# visualize result
def visualize_filtered_predictions(image, instances, filtered_indices, metadata, heatmap_overlay):
    filtered_instances = instances[filtered_indices]
    v = Visualizer(image[:, :, ::-1], metadata=metadata)  # Convert image to RGB for Visualizer
    v = v.draw_instance_predictions(filtered_instances)
    result_image = v.get_image()[:, :, ::-1]  # Convert back to BGR for consistency with OpenCV
    plt.figure(figsize=(12, 8))
    plt.imshow(result_image)
    plt.imshow(heatmap_overlay, alpha=0.5)
    plt.axis('off')
    plt.show()

def visualize_overlay(image, instances, metadata, heatmap_overlay):
    v = Visualizer(image[:, :, ::-1], metadata=metadata)  # Convert image to RGB for Visualizer
    v = v.draw_instance_predictions(instances)
    result_image = v.get_image()[:, :, ::-1]  # Convert back to BGR for consistency with OpenCV
    plt.figure(figsize=(12, 8))
    plt.imshow(result_image)
    plt.imshow(heatmap_overlay, alpha=0.5)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    file_name = 'basketball_player1.jpg'
    file_path = 'test_images/' + file_name
    image = cv2.imread(file_path)
    detectron = Detectron(image)
    detectron.compare()

