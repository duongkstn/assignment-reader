"""Filename: dig_doc_utils/pdfhandle.py.

Company: VNPT-IT.
Project: VNPT Smart Reader.
Developer: Dao Ngoc An.
Created: 31/03/2023.
Description: PDF handler.
"""
import io
import re

import bogo
import cv2
import numpy as np
import pdfminer
import visen
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTChar
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from skimage.measure import label, regionprops
from visen.format import get_enter_code
from collections import Counter


def most_frequent(list_arr):
    occurence_count = Counter(list_arr)
    return occurence_count.most_common(1)[0][0]


class PDFHandle:
    """PDF handler."""

    @staticmethod
    def process_pdf_file(file_path, imgs):
        """Process pdf file."""
        fp = open(file_path, 'rb')
        parser = PDFParser(fp)
        document = PDFDocument(parser)
        if not document.is_extractable:
            return None
        contents = []
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page, img in zip(PDFPage.create_pages(document), imgs):
            img = np.ascontiguousarray(img)
            hreal, wreal = img.shape[:2]
            hxml, wxml = page.mediabox[3], page.mediabox[2]
            hscale, wscale = hreal / hxml, wreal / wxml
            # read the page into a layout object
            interpreter.process_page(page)
            layout = device.get_result()
            # extract text from this object
            contents.append(parse_obj(layout._objs, hscale, wscale, hxml, wxml, img, laparams))
        return contents

    @staticmethod
    def process_pdf_content(file_content, imgs):
        """Process pdf content."""
        fp = io.BytesIO(file_content)
        parser = PDFParser(fp)
        document = PDFDocument(parser)
        if not document.is_extractable:
            return None
        contents = []
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page, img in zip(PDFPage.create_pages(document), imgs):
            img = np.ascontiguousarray(img)
            hreal, wreal = img.shape[:2]
            hxml, wxml = page.mediabox[3], page.mediabox[2]
            hscale, wscale = hreal / hxml, wreal / wxml
            # read the page into a layout object
            interpreter.process_page(page)
            layout = device.get_result()
            # extract text from this object
            contents.append(parse_obj(layout._objs, hscale, wscale, hxml, wxml, img, laparams))
        return contents


def parse_obj(lt_objs, hscale, wscale, hxml, wxml, img, laparams):
    """Parse object."""
    ocr_results, bboxes = [], []
    vertical_lines = detect_vertical_lines(img, img.shape[0] // 30)
    vertical_lines = [revert_bbox(bbox, hxml, wscale, hscale) for bbox in vertical_lines]
    for tb_obj in lt_objs:
        if isinstance(tb_obj, pdfminer.layout.LTTextBoxHorizontal):
            for tl_obj in tb_obj:
                if isinstance(tl_obj, pdfminer.layout.LTTextLineHorizontal):
                    tl_obj = strip_line_horizontal(tl_obj)
                    for tl_obj_sub in split_textline_cross_vertical_lines(tl_obj, vertical_lines, hxml):
                        tl_obj_sub.bbox = get_bbox(tl_obj_sub.bbox, hxml, wscale, hscale)
                        text = clean_text(get_text_line_horizontal(tl_obj_sub))
                        if text == "ERRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRR":
                            return [], []
                        bbox = tl_obj_sub.bbox
                        if len(text) < 2 and bbox[3] - bbox[1] < bbox[2] - bbox[0]:
                            continue
                        if not len(text) or text == ' ':
                            continue
                        ocr_results.append((text, 1.0))
                        bboxes.append(bbox)
        elif isinstance(tb_obj, pdfminer.layout.LTFigure):
            new_page = pdfminer.layout.LTPage('1', tb_obj.bbox)
            for o in tb_obj:
                if type(o) in [pdfminer.layout.LTLine, pdfminer.layout.LTTextLineHorizontal, pdfminer.layout.LTChar,
                               pdfminer.layout.LTRect]:
                    new_page.add(o)
            new_page.analyze(laparams)
            _os, _bs = parse_obj(new_page, hscale, wscale, hxml, wxml, img, laparams)
            ocr_results.extend(_os)
            bboxes.extend(_bs)

    indices = [
        i[0] for i in sorted(enumerate(bboxes), key=lambda x: x[1][1] * (hxml * hscale) + x[1][0])
    ]
    ocr_results = [ocr_results[i] for i in indices]
    bboxes = [bboxes[i] for i in indices]
    return ocr_results, bboxes


def strip_line_horizontal(line_horizontal):
    new_line = pdfminer.layout.LTTextBoxHorizontal()
    chars = [c for c in line_horizontal._objs if isinstance(c, LTChar)]
    while len(chars):
        if not chars[0]._text.strip():
            del chars[0]
        else:
            break
    while len(chars):
        if not chars[-1]._text.strip():
            del chars[-1]
        else:
            break
    for char in chars:
        new_line.add(char)
    return new_line



def normalize_bbox(bbox, hxml):
    """Normalize bounding box."""
    bbox = list(bbox)
    bbox[1] = hxml - bbox[1]
    bbox[3] = hxml - bbox[3]
    bbox[1], bbox[3] = bbox[3], bbox[1]
    return bbox


def get_bbox(bbox, hxml, wscale, hscale):
    """Get bounding box."""
    bbox = normalize_bbox(bbox, hxml)
    bbox[0] *= wscale
    bbox[2] *= wscale
    bbox[1] *= hscale
    bbox[3] *= hscale
    return list(map(round, bbox))


def revert_bbox(bbox, hxml, wscale, hscale):
    """Revert bounding box."""
    bbox = list(bbox)
    bbox[0] /= wscale
    bbox[2] /= wscale
    bbox[1] /= hscale
    bbox[3] /= hscale
    return bbox


def detect_vertical_lines(src, vertical_size):
    """Detect vertical lines."""
    if len(src.shape) != 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src

    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    # vertical
    vertical = np.copy(bw)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    label_img = label(vertical)
    regions = regionprops(label_img)
    r_v = []
    bboxes = [(props.bbox[1], props.bbox[0], props.bbox[3], props.bbox[2]) for props in regions]
    bboxes = merge_bboxes(bboxes, 1)
    for bbox in bboxes:
        r_v.append(((bbox[0] + bbox[2]) // 2, bbox[1], (bbox[0] + bbox[2]) // 2, bbox[3]))
    return r_v


def split_textline_cross_vertical_lines(textline, vertical_lines, hxml):
    """Split text line cross vertical lines."""
    result = []
    textline_bbox = normalize_bbox(textline.bbox, hxml)
    cross_lines = []
    for hl in vertical_lines:
        if textline_bbox[0] < hl[0] < textline_bbox[2] and hl[1] < textline_bbox[1] < textline_bbox[3] < hl[3]:
            cross_lines.append(hl[0])
    cross_lines += [textline_bbox[0] - 5, textline_bbox[2]]
    cross_lines = sorted(cross_lines)
    segments = [(cross_lines[i], cross_lines[i + 1]) for i in range(len(cross_lines) - 1)]
    if len(segments) == 1:
        return [textline]
    for segment in segments:
        new_line = pdfminer.layout.LTTextLineHorizontal(0.1)
        for c in textline:
            if isinstance(c, pdfminer.layout.LTChar) and segment[0] < c.bbox[0] < segment[1]:
                new_line.add(c)
        result.append(new_line)
    return result


def merge_bboxes(bboxes, axis=1):
    """Merge bounding boxes."""
    # axis = 1 la merge theo x:
    lines = bboxes
    merged_line = [False for _ in range(len(bboxes))]
    _lines = [[lines[i]] for i in range(len(lines))]

    remove_lines = []
    for i in range(len(lines)):
        if merged_line[i]:
            continue
        merged_line[i] = True
        rec_i = lines[i]
        for j in range(0, len(lines)):
            if merged_line[j]:
                continue
            rec_j = lines[j]
            if axis == 1:
                if rec_i[0] == rec_j[0]:
                    _lines[i].append(lines[j])
                    merged_line[j] = True
                    remove_lines.append(j)
            elif axis == 0:
                if rec_i[1] == rec_j[1]:
                    _lines[i].append(lines[j])
                    merged_line[j] = True
                    remove_lines.append(j)
    for j in sorted(remove_lines, reverse=True):
        del _lines[j]
    for idx, line in enumerate(_lines):
        length = 0
        if axis == 1:
            for l in line:
                length += l[3] - l[1]
        else:
            for l in line:
                length += l[2] - l[0]

        line = sorted(line, key=lambda x: x[1])
        xmin = min([l[0] for l in line])
        ymin = min([l[1] for l in line])
        xmax = max([l[2] for l in line])
        ymax = max([l[3] for l in line])
        _lines[idx] = (xmin, ymin, xmax, ymax)
    return _lines


def get_text_line_horizontal(line_horizontal):
    """Get horizontal text line."""
    ltchars = [x for x in line_horizontal._objs if isinstance(x, LTChar)]
    space_chars = [x for x in ltchars if x._text == ' ']
    invalid_space_chars = [x for x in space_chars if x.bbox[2] - x.bbox[0] < 1]
    for space in invalid_space_chars:
        ltchars.remove(space)
    characters = [x._text for x in ltchars]
    return ''.join(characters)


def clean_text(text):
    """Clean text."""
    text = vni_to_telex(text)
    text = visen.clean_tone(text)
    vocab = """aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ """
    valid_characters = set(list(vocab))
    all_characters = set(list(text))
    unvalid_characters = all_characters.difference(valid_characters)
    for c in unvalid_characters:
        print("ERRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRR", c)
        text = "ERRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRR"
    return text


def vni_to_telex(text):
    """Convert vni to telex."""
    invalid_characters = ['–', '³', '²', 'ı', ]
    valid_characters = ['-', '3', '2', 'i', ]
    assert len(invalid_characters) == len(valid_characters)
    for i, v in zip(invalid_characters, valid_characters):
        text = text.replace(i, v)

    vni_patern = r'̣|́|̀|̉'
    vni_2_telex = {
        chr(803): '`cham`',
        chr(769): '`sac`',
        chr(768): '`huyen`',
        chr(777): '`hoi`',
    }
    telex_characters = ['`cham`', '`sac`', '`huyen`', '`hoi`']
    append_enter_code = {
        "`cham`": "j",
        "`sac`": "s",
        "`huyen`": "f",
        "`hoi`": "r"
    }
    while list(re.finditer(vni_patern, text)):
        telex_character = list(re.finditer(vni_patern, text))[0]
        vni_character = text[telex_character.start():telex_character.end()]
        text = text.replace(vni_character, vni_2_telex[vni_character], 1)

    out_words = []
    # words = re.findall(r"[\w`]+|[.,!?;\d]", text)
    words = re.split(" ", re.sub(r"[\(|\d|\)|\.|,|?|!|;]+", " ", text))
    for word in words:
        for telex_character in telex_characters:
            if telex_character in word:
                word = word.replace(telex_character, f"{append_enter_code[telex_character]}.")
                _words = re.split("\.", word)
                __words = []
                for _word in _words[:-1]:
                    sub_word_enter = get_enter_code(_word)
                    __words.append(bogo.process_sequence(sub_word_enter, skip_non_vietnamese=False))
                __words.append(_words[-1])
                word = ''.join(__words)
                word = word.replace(".", "")
        out_words.append(word)
    new_text = ''
    if len(words):
        new_text = text[:text.find(words[0])]
    for i in range(len(words) - 1):
        start = text.find(words[i])
        end = start + len(words[i]) + text[start + len(words[i]):].find(words[i + 1])
        new_word = text[start:end]
        text = text[end:]
        new_text += new_word.replace(words[i], out_words[i])
    if len(words):
        new_text += text.replace(words[-1], out_words[-1])
    return new_text


if __name__ == '__main__':
    text = "Khu phố 4, Thị trấn Dầu Tiếng, Huyện Dầu Tiếng,tı̉nh Bı̀nh Dương"
    text = "CÔNG TY TNHH MAI LINH NAM ĐIṆ H"
    text = "Đơn vị bán hàng"
    text = '--'
    # text = "Miếngdán hạ dao ngoc an. sốtByeByeFever6's"
    # text =  "Miếngdán hạ sốtByeByeFever6's"
    text = 'SuperCool(Hộp3baox2miếng'
    print(vni_to_telex(text))
