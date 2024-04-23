from reader.converter import Converter
from reader.ocr.vietocr.vietocr import Predictor as OcrPredictor
from reader.text_detection.predict_det_db import TextDetector
from reader.utils import get_rotate_crop_image, get_contour_precedence


class Reader(object):
    def __init__(self):
        self.ocr_model = OcrPredictor()
        self.text_detector = TextDetector()
        self.converter = Converter()

    def predict(self, image):
        pass

    def process_document(self, file_pdf_bytes):
        """
        xử lý văn bản với đầu vào là binary file
        :param file_pdf_bytes: rb file
        :return: [ content_page_0, content_page_1, content_page_2, ...]
        content_page_x = (ocr_results, lines) ocr_results là 1 list các text đi kèm với confident_score
                                              lines là các bounding box ứng với ocr
        """
        images, _, content = self.converter.from_bytes(file_pdf_bytes, try_get_content=True)
        images_no_content_indices = [idx_img for idx_img, (img, c) in enumerate(zip(images, content)) if len(c[0]) == 0]
        images_no_content = [images[i] for i in images_no_content_indices]
        bboxes_filelist = self.text_detector.process_batch(images_no_content)
        for bboxes, img, idx_img in zip(bboxes_filelist, images_no_content, images_no_content_indices):
            ocr_results, lines = [], []
            img_lines = []
            for box in bboxes:
                img_line = get_rotate_crop_image(img, box)
                xmin, ymin = box.min(0).tolist()
                xmax, ymax = box.max(0).tolist()
                lines.append((xmin, ymin, xmax, ymax))
                img_lines.append(img_line)
            texts, confs = self.ocr_model.process_batch(img_lines)
            ocr_results = list(zip(texts, confs))
            _os, _ls, _bs = [], [], []
            ocr_lines_boxes = sorted(
                zip(ocr_results, lines, bboxes), key=lambda x: get_contour_precedence(x[2], img.shape[0])
            )
            for o, l, b in ocr_lines_boxes:
                _os.append(o)
                _ls.append(l)
                _bs.append(b)
            ocr_results = _os
            lines = _ls
            content[idx_img] = (ocr_results, lines)
        return content
