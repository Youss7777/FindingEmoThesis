"""
A test script to create a pipeline that first uses YoLo v3 + OpenImages to detect faces

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
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


class LocalAnalysis:
    def __init__(self, device=torch.device('cpu')):
        self.device = device

        # Load names of OpenImage classes
        class_map = []
        with open("openimages.names", "r") as fin:
            for line in fin:
                line = line.strip()
                class_map.append(line)

        # Load YoloV3 model + OpenImage weights
        self.model = YoloV3(601)
        self.model.load(os.path.join(os.path.expanduser('~'), 'Desktop', 'Thesis', 'PytorchProject', 'emonet-master', 'emonet_py', 'yolov3-openimages.weights'))
        self.model.eval()
        self.model.to(device)

        thresh = 0.005

        # Create post-processing pipeline
        self.post = ln.data.transform.Compose([
            # GetBoxes transformation generates bounding boxes from network output
            ln.data.transform.GetMultiScaleAnchorBoxes(
                conf_thresh=thresh,
                network_stride=self.model.stride,
                anchors=self.model.anchors
            ),

            # Filter transformation to filter the output boxes
            ln.data.transform.NMS(
                iou_thresh=thresh
            ),

            # Miscelaneous transformation that transforms the output boxes to a brambox dataframe
            ln.data.transform.TensorToBrambox(
                class_label_map=class_map,
            )
        ])

        img_resize = ImgResize(width=608, height=608)
        self.transform = transforms.Compose([img_resize])

    def confidence_cutoff(self, df, threshold):
        df.loc[df['confidence'] < threshold, 'importance'] = 0
        return df

    def add_importance(self, df, heatmap):
        importance = []
        for index, row in df.iterrows():
            x_min = int(row["x_top_left"])
            x_max = int(row["x_top_left"] + row["width"])
            y_min = int(row["y_top_left"])
            y_max = int(row["y_top_left"] + row["height"])
            # region inside the bounding box
            bounded_region = heatmap[y_min:y_max, x_min:x_max]
            # define importance as the average activation inside that region
            importance.append(np.mean(bounded_region))
        df["object_importance"] = importance
        return df

    def local_analysis(self, file_path, file_name, explanation_method='gradcam', nb_objects=0, show_output=False):
        """
        Perform local analysis on single image.
        """
        img_path = os.path.join(file_path)

        # load the corresponding grad-cam heatmap
        if explanation_method == 'gradcam':
            grayscale_cam = np.load("cam_grad_dataset/cam_grad_" + file_name + ".npy")
        if explanation_method == 'gradcampp':
            grayscale_cam = np.load("cam_grad_pp_dataset/cam_grad_pp_" + file_name + ".npy")
        if explanation_method == 'ablationcam':
            grayscale_cam = np.load("cam_ablation_dataset/cam_ablation_" + file_name + ".npy")
        if explanation_method == 'scorecam':
            grayscale_cam = np.load("cam_score_dataset/cam_score_" + file_name + ".npy")
        if explanation_method == 'eigen':
            grayscale_cam = np.load("cam_eigen_dataset/cam_eigen_" + file_name + ".npy")
        if explanation_method == 'liftcam':
            grayscale_cam = np.load("cam_lift_dataset/cam_lift_" + file_name + ".npy")
        if explanation_method == 'lrpcam':
            grayscale_cam = np.load("cam_lrp_dataset/cam_lrp_" + file_name + ".npy")
        if explanation_method == 'limecam':
            grayscale_cam = np.load("cam_lime_dataset/cam_lime_" + file_name + ".npy")
        with torch.no_grad():
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img_tensor = self.transform(img)

            # resize heatmap to input image
            original_size_img = img.size
            grayscale_cam = cv2.resize(grayscale_cam, original_size_img)
            grayscale_cam_pil = Image.fromarray(grayscale_cam)
            grayscale_cam_tensor = self.transform(grayscale_cam_pil)
            grayscale_cam_scaled = grayscale_cam_tensor.numpy()[0, :]

            # get output of Yolo
            output_tensor = self.model(img_tensor.unsqueeze(0).to(self.device))

            # post-processing
            output_df = self.post(output_tensor)
            proc_img = img_tensor.cpu().numpy().transpose(1, 2, 0)

            # superimpose image and gradcam heatmap
            cam = show_cam_on_image(proc_img, grayscale_cam_scaled, use_rgb=True)
            pil_img = Image.fromarray(cam)

            # add importance of bounding boxes
            df_complete = self.add_importance(output_df, grayscale_cam_scaled)

            # rename 'class_label' to 'detected_object' for more clarity later
            df_complete_return = df_complete.rename(columns={'class_label': 'detected_object',
                                                      'confidence': 'object_confidence'}).sort_values(by="object_importance", ascending=False)
            df_sorted = df_complete.sort_values(by="object_importance", ascending=False)

            if show_output:
                df_sorted = df_sorted.head(nb_objects)
                print(df_sorted)
                #bb.util.draw_boxes(pil_img, df_sorted, label=df_sorted.class_label)
                fig, ax = plt.subplots()
                ax.imshow(bb.util.draw_boxes(pil_img, df_sorted, label=df_sorted.class_label))
                ax.axis('off')
                object_list = '\n'.join([f'{obj}: {imp*100:0.1f}%' for obj, imp in zip(df_sorted['class_label'], df_sorted['object_importance'])])
                fig.text(0.5, 0.97, explanation_method+'\n'+object_list, ha='center', va='top', fontsize=9, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
                plt.subplots_adjust(top=0.83)
                plt.show()

        return df_complete_return
