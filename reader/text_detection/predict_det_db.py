import os
from argparse import Namespace

import cv2
import numpy as np
from paddle import inference
from reader.text_detection.data.db_process import DBProcess
from reader.text_detection.postprocess.db_postprocess import DBPostProcess
import gc


class TextDetector(object):
    def __init__(self, type='mobile_general', args=None, gpu="0", gpu_mem=1200):
        assert type in ['mobile_general', 'server_general']
        self.device_id = int(gpu)
        current_file_path = os.path.realpath(__file__)
        current_dir = os.path.dirname(current_file_path)
        current_dir = os.path.abspath(current_dir)
        if type == 'mobile_general' and args is None:
            args = {"det_max_side_len": 1280, "det_db_thresh": 0.3, "det_db_box_thresh": 0.0,
                    "det_db_unclip_ratio": 1.6,
                    "det_model_dir": os.path.join(current_dir, '../weights/ch_ppocr_mobile_v1.1_det_infer'),
                    "enable_mkldnn": True, "gpu_mem": 8000, "device_id": 0, "use_zero_copy_run": False}
        args["use_gpu"] = self.device_id > -1
        args["gpu_mem"] = gpu_mem
        if "det_max_side_len" in args:
            preprocess_params = {'max_side_len': args["det_max_side_len"]}
        elif "det_image_shape" in args:
            preprocess_params = {"test_image_shape": args["det_image_shape"]}
        else:
            preprocess_params = {}
        postprocess_params = {}
        self.preprocess_op = DBProcess(preprocess_params)
        args = Namespace(**args)
        postprocess_params["thresh"] = args.det_db_thresh  # 0.3
        postprocess_params["box_thresh"] = args.det_db_box_thresh  # 0.5
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio  # 1.6
        self.postprocess_op = DBPostProcess(postprocess_params)
        model_dir = args.det_model_dir

        model_file_path = model_dir + "/model"
        params_file_path = model_dir + "/params"
        if not os.path.exists(model_file_path):
            model_file_path = model_dir + "/inference.pdmodel"
            params_file_path = model_dir + "/inference.pdiparams"
        config = inference.Config(model_file_path, params_file_path)

        if args.use_gpu:
            config.enable_use_gpu(args.gpu_mem, self.device_id)
            self.device = f'gpu:{self.device_id}'
        else:
            self.device = 'cpu'
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(6)
            if args.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()

        config.enable_memory_optim()
        config.disable_glog_info()

        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.switch_use_feed_fetch_ops(False)

        predictor = inference.create_predictor(config)
        input_names = predictor.get_input_names()
        for name in input_names:
            input_tensor = predictor.get_input_handle(name)
        output_names = predictor.get_output_names()
        output_tensors = []
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
        self.predictor = predictor
        self.input_tensor = input_tensor
        self.output_tensors = output_tensors

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img, hmin=160, wmin=160):
        ori_im = img.copy()
        h = max(hmin, ori_im.shape[0])
        w = max(wmin, ori_im.shape[1])
        if h > self.preprocess_op.max_side_len and w * self.preprocess_op.max_side_len / h < wmin:
            w = round(wmin * h / self.preprocess_op.max_side_len)
        if w > self.preprocess_op.max_side_len and h * self.preprocess_op.max_side_len / w < hmin:
            h = round(hmin * w / self.preprocess_op.max_side_len)
        zero_img = np.zeros((h, w, 3), np.uint8)
        zero_img[:ori_im.shape[0], :ori_im.shape[1]] = ori_im
        ori_im = zero_img

        im, ratio_list = self.preprocess_op(ori_im)
        if im is None:
            return None, 0
        im = im.copy()
        self.input_tensor.copy_from_cpu(im)
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()

            dt_boxes_list = self.postprocess_op(output, [ratio_list])
            dt_boxes = dt_boxes_list[0]
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
            outputs.append(dt_boxes)
        self.predictor.clear_intermediate_tensor()
        self.predictor.try_shrink_memory()
        del ori_im
        del im
        gc.collect()
        return outputs

    def process_img(self, img, hmin=160, wmin=160, return_first_result=True):
        outs = self(img, hmin=hmin, wmin=wmin)
        if return_first_result:
            outs = outs[0]
        return outs

    def process_batch(self, imgs, hmin=160, wmin=160, batch_size=16, return_first_result=True):
        if self.device == 'cpu':
            results = [self.__call__(img, hmin, wmin) for img in imgs]
            if return_first_result:
                results = [r[0] for r in results]
            else:
                line_bboxes, word_bboxes = zip(*results)
                results = (line_bboxes, word_bboxes)
            return results
        dt_boxes_list = []
        for irange in [(i, i + batch_size) for i in range(0, len(imgs), batch_size)]:
            batch_img = imgs[irange[0]:irange[1]]
            dt_boxes_list.extend(self.process_batch_gpu(batch_img, hmin, wmin, return_first_result))
        return dt_boxes_list

    def process_batch_gpu(self, imgs, hmin=160, wmin=160, return_first_result=True):
        new_imgs = []
        original_shapes = []
        for img in imgs:
            ori_im = img.copy()
            h = max(hmin, ori_im.shape[0])
            w = max(wmin, ori_im.shape[1])
            if h > self.preprocess_op.max_side_len and w * self.preprocess_op.max_side_len / h < wmin:
                w = wmin * h // self.preprocess_op.max_side_len
            if w > self.preprocess_op.max_side_len and h * self.preprocess_op.max_side_len / w < hmin:
                h = hmin * w // self.preprocess_op.max_side_len
            zero_img = np.zeros((h, w, 3), np.uint8)
            zero_img[:ori_im.shape[0], :ori_im.shape[1]] = ori_im
            ori_im = zero_img
            new_imgs.append(ori_im)
            original_shapes.append(ori_im.shape)

        batch_img, ratios = self.preprocess_op.process_batch(new_imgs)
        if batch_img is None:
            return None, 0
        result = []
        batch_img = batch_img.copy()
        self.input_tensor.copy_from_cpu(batch_img)
        self.predictor.run()
        for output_tensor in self.output_tensors:
            out = []
            output = output_tensor.copy_to_cpu()
            dt_boxes_list = self.postprocess_op(output, ratios)
            for dt_boxes, original_shape in zip(dt_boxes_list, original_shapes):
                dt_boxes = self.filter_tag_det_res(dt_boxes, original_shape)
                out.append(dt_boxes)
            result.append(out)
        if return_first_result:
            result = result[0]
        del batch_img
        del new_imgs
        gc.collect()
        return result

    def visualize(self, img, hmin=160, wmin=160):
        import colorsys
        import random

        def random_colors(N, bright=True):
            """
            Generate random colors.
            To get visually distinct colors, generate them in HSV space then
            convert to RGB.
            """
            brightness = 1.0 if bright else 0.7
            hsv = [(i / N, 1, brightness) for i in range(N)]
            colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
            random.shuffle(colors)
            return colors

        def apply_mask(image, mask, color, alpha=0.5):
            """Apply the given mask to the image.
            """
            for c in range(3):
                image[:, :, c] = np.where(mask == 1,
                                          image[:, :, c] *
                                          (1 - alpha) + alpha * color[c] * 255,
                                          image[:, :, c])
            return image

        def draw_text_det_res(dt_boxes, src_im, color):
            for box in dt_boxes:
                box = np.array(box).astype(np.int32).reshape(-1, 2)
                cv2.polylines(src_im, [box], True, color=color, thickness=2)
            return src_im

        img = img.copy()
        max_side = max(img.shape)
        hmin = max(hmin, round(hmin * max_side / self.preprocess_op.max_side_len))
        wmin = max(wmin, round(wmin * max_side / self.preprocess_op.max_side_len))
        h = max(hmin, img.shape[0])
        w = max(wmin, img.shape[1])
        zero_img = np.zeros((h, w, 3), np.uint8)
        zero_img[:img.shape[0], :img.shape[1]] = img
        img = zero_img
        im, ratio_list = self.preprocess_op(img)
        if im is None:
            return None, 0
        im = im.copy()
        self.input_tensor.copy_from_cpu(im)
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        for i, output in enumerate(outputs):
            dt_boxes_list = self.postprocess_op(output, [ratio_list])
            dt_boxes = dt_boxes_list[0]
            dt_boxes = self.filter_tag_det_res(dt_boxes, img.shape)

            pred = output

            pred = pred[:, 0, :, :]
            segmentation = pred > self.postprocess_op.thresh
            segmentation = cv2.resize(segmentation[0].astype(np.uint8), (img.shape[1], img.shape[0]))

            visualized_image = apply_mask(img, segmentation, random_colors(50)[i])
            color = random_colors(50)[i]
            color = (round(color[0] * 255), round(color[1] * 255), round(color[2] * 255))
            visualized_image = draw_text_det_res(dt_boxes, visualized_image, color)

        return visualized_image
