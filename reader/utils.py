import math

import cv2
import numpy as np
import pyclipper
from PIL import Image, ImageDraw, ImageFont


def get_contour_precedence(contour, cols):
    origin = cv2.boundingRect(contour)
    return origin[1] * cols + origin[0]

def order_points(pts):
    """Sort points."""
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    _, unique_indices = np.unique(s, return_index=True)
    while len(unique_indices) < 4:
        for i in unique_indices:
            pts[i][0] -= 1
        s = pts.sum(axis=1)
        _, unique_indices = np.unique(s, return_index=True)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    selected_indices = [np.argmin(s).tolist(), np.argmax(s).tolist()]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    _, unique_indices = np.unique(diff, return_index=True)
    while len(unique_indices) < 4:
        for i in unique_indices:
            pts[i][0] -= 1
        diff = np.diff(pts, axis=1)
        _, unique_indices = np.unique(diff, return_index=True)
    if np.argmin(diff).tolist() in selected_indices and np.argmax(diff).tolist() in selected_indices:
        remain_indices = [index for index in range(4) if index not in selected_indices]
        if selected_indices[0] > selected_indices[1]:
            rect[1] = pts[remain_indices[1]]
            rect[3] = pts[remain_indices[0]]
        else:
            rect[1] = pts[remain_indices[0]]
            rect[3] = pts[remain_indices[1]]
    else:
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def get_rotate_crop_image(img, points, padding=5):
    """Rotate crop image.

    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    subject = [tuple(p) for p in points]
    po = pyclipper.PyclipperOffset()
    po.AddPath(subject, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    points = order_points(np.array(po.Execute(padding)[0], dtype=np.float32))
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32([[0, 0], [img_crop_width - 1, 0],
                          [img_crop_width - 1, img_crop_height - 1],
                          [0, img_crop_height - 1]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)

    return dst_img

def draw_ocr_box_txt(
        image,
        boxes,
        txts,
        scores=None,
        drop_score=0.5,
        font_path="times.ttf"
):
    """Draw OCR box in image."""
    image = Image.fromarray(image)
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][
            1]) ** 2)
        box_width = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][
            1]) ** 2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)

def imshow(img, backend="cv2", name="", resize=True, height=1000):
    """Visualize image."""
    if backend == "cv2":
        if resize:
            img_c = cv2.resize(img, (img.shape[1] * height // img.shape[0], height))
        else:
            img_c = img.copy()
        cv2.imshow(name, img_c)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        Image.fromarray(img[:, :, ::-1]).show(name)

def xyxy2xyxyxyxy(bboxes):
    new_bboxes = []
    for bbox in bboxes:
        new_bboxes.append([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])
    return new_bboxes
def visual_result(img, boxes, ocr_results):
    vis_img = draw_ocr_box_txt(img, xyxy2xyxyxyxy(boxes), list(zip(*ocr_results))[0], None, font_path='./times.ttf')
    _img = img.copy()
    vis_img[:img.shape[0], :img.shape[1], :] = _img[:img.shape[0], :img.shape[1]]
    imshow(vis_img, "")
