from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import tempfile
import torch

class YoloSegment:
    def __init__(self):
        self.vehicle_classes = ["bicycle", "car", "motorcycle", "bus", "truck"]
        pass

    def load_model(self, model_name="yolov8s-seg.pt"):
        self.model = YOLO(model_name)

    def get_fg_mask(self, images):
        if images.dim() == 4:
            images = images.unsqueeze(0)

        B, S, C, H, W = images.shape

        images = images.view(B * S, C, H, W)
        
        results = []

        # =========================
        # 🔥 SAFE INFERENCE LOOP
        # =========================
        for img in images:
            img = img.detach().cpu()

            # CHW -> HWC
            img = img.permute(1, 2, 0).numpy()

            # float [0,1] -> uint8 [0,255]
            img = (img * 255).clip(0, 255).astype(np.uint8)

            img = np.ascontiguousarray(img)

            # 🚨 IMPORTANT: single image inference (NO batch list)
            result = self.model.predict(
                source=img,
                conf=0.5,
                task="segment",
                verbose=False
            )[0]

            results.append(result)

        fg_masks = []
        for result in results:
            img_h, img_w = result.orig_shape

            if result.masks is None:
                fg_masks.append(torch.zeros((img_h, img_w)))
                continue
    
            center_x_img = img_w / 2

            min_distance = float('inf')
            best_mask = None
            masks = result.masks.data

            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]
                if class_name not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_center_x = (x1 + x2) / 2
                distance = abs(box_center_x - center_x_img)

                if distance < min_distance:
                    min_distance = distance
                    best_mask = masks[i]

            if best_mask is None:
                fg_masks.append(torch.zeros(img_h, img_w))
            else:
                # torch → numpy
                fg_mask = best_mask.detach().cpu().numpy()

                # resize nếu cần
                if fg_mask.shape != (img_h, img_w):
                    fg_mask = cv2.resize(
                        fg_mask.astype(np.float32),
                        (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST
                    )

                # binary mask
                fg_mask = (fg_mask > 0.5).astype(np.uint8)

                fg_mask = torch.from_numpy(fg_mask)

            fg_masks.append(fg_mask)

        return torch.stack(fg_masks, dim=0)