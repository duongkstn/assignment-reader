import math
import os
import json
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
        """
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
        """
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
    print(vis_img, type(vis_img))
    imshow(vis_img, "") 


def gemma_marshal_llm_to_json(output: str) -> str:
    """
    Extract a substring containing valid JSON or array from a string.

    Args:
        output: A string that may contain a valid JSON object or array surrounded by
        extraneous characters or information.

    Returns:
        A string containing a valid JSON object or array.
    """
    output = output.strip().replace("{{", "{").replace("}}", "}")

    left_square = output.find("[")
    left_brace = output.find("{")

    if left_square < left_brace and left_square != -1:
        left = left_square
        right = output.find("]")
    else:
        left = left_brace
        right = output.rfind("}")

    return output[left : right + 1]

from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "")
)
"""
from g4f.client import Client
client = Client()
"""
import time
def translate(sentence, language):
    sentence = sentence.strip()
    if not any(c.isalpha() for c in sentence):
        return sentence
    try:
        for _ in range(1):
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"""translate the sentence, return to format:
                ```
                {{
                    "translated_sentence": ""
                }}
                ```
                Let's translate it to {language}
                Sentence:
                {sentence}""",
                    }
                ],
                model="gpt-3.5-turbo"
            )
            print('o0', sentence)
            output = chat_completion.choices[0].message.content
            if output != "":
                break
            time.sleep(10)
        print('o1', output)
        output = gemma_marshal_llm_to_json(output)
        print('o2', output)
        output = json.loads(output)
        print('o3', output)
        output = output["translated_sentence"]
        print('o4', output)
    except Exception as e:
        print(e)
        print(sentence, language)
        output = sentence
    return output


def map_translate(sentence, language):
    mapping = {
        "english": {'ĐẠI HỌC QUỐC GIA TP.HCM': 'HO CHI MINH CITY NATIONAL UNIVERSITY',
 'CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM': 'SOCIALIST REPUBLIC OF VIETNAM',
 'TRƯỜNG ĐẠI HỌC': 'UNIVERSITY',
 'Độc lập - Tự do - Hạnh phúc': 'Independence - Freedom - Happiness',
 'CÔNG NGHỆ THÔNG TIN': 'INFORMATION TECHNOLOGY',
 'Số: 18/QĐ-ĐHCNTT': 'Number: 18/QD-DHCNTT',
 'Tp.HCM, ngày 05 tháng 01 năm 2024': 'Ho Chi Minh City, January 05, 2024',
 'QUYẾT ĐỊNH': 'DECISION',
 'V/v Điều chỉnh mức thu học phí trình độ đào tạo đại học chính quy': 'Adjust the tuition fee level for regular university education',
 'chương trình chuẩn từ KHÓA 2020 trở về trước': 'standard program from COURSE 2020 and earlier',
 'Năm học 2023-2024': 'Academic year 2023-2024',
 'HIỆU TRƯỞNG TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN': 'PRINCIPAL OF THE UNIVERSITY OF INFORMATION TECHNOLOGY',
 'Căn cứ Quyết định số 134/2006/QĐ-TTg ngày 08/6/2006 của Thủ tướng Chính phủ': 'Based on Decision No. 134/2006/QD-TTg dated June 08, 2006 of the Prime Minister',
 'về việc thành lập Trường Đại học Công nghệ Thông tin thuộc Đại học Quốc gia thành phố': 'regarding the establishment of the University of Information Technology under the National University of the City',
 'Hồ Chí Minh (ĐHQG-HCM);': 'Ho Chi Minh (HCMC University of Science)',
 'Căn cứ Quyết định số 867/QĐ-ĐHQG ngày 17/8/2016 của Giám đốc ĐHQG-HCM': 'Based on Decision No. 867/QD-DHQG dated August 17, 2016 of the Director of HCMC National University',
 'về việc ban hành Quy chế tổ chức và hoạt động của Trường Đại học thành viên và khoa': 'on the issuance of the Charter of Organization and Operation of Member Universities and Faculties',
 'trực thuộc ĐHQG-HCM;': 'affiliated with VNU-HCM',
 'Căn cứ Nghị định số 97/2023/NĐ-CP ngày 31/12/2023 của Thủ tướng Chính phủ': 'Pursuant to Decree No. 97/2023/ND-CP dated December 31, 2023 of the Prime Minister',
 'sửa đổi Nghị định 81/2021/NĐ-CP quy định về cơ chế thu, quản lý học phí đối với cơ sở': 'amend Decree 81/2021/ND-CP regulating the mechanism for collection and management of tuition fees at educational institutions',
 'giáo dục thuộc hệ thống giáo dục quốc dân và chính sách miễn, giảm học phí, hỗ trợ chi': 'education under the national education system and policies of fee exemptions, reductions, and support',
 'phí học tập; giá dịch vụ trong lĩnh vực giáo dục, đào tạo;': 'tuition fees; service prices in the field of education and training;',
 'Căn cứ Quyết định số 1368/QĐ-ĐHQG-ĐH&SĐH ngày 21/11/2008 của Giám đốc': 'Based on Decision No. 1368/QD-DHQG-DH&SDH dated November 21, 2008 of the Director',
 'ĐHQG-HCM về việc Ban hành Quy chế đào tạo Đại học, Cao đẳng theo hệ thống tín chỉ;': 'HCMC National University on issuing regulations on university and college education according to the credit system;',
 'Căn cứ Quyết định số 546/QĐ-ĐHCNTT ngày 30/8/2019 của Hiệu trưởng Trường': 'Based on Decision No. 546/QD-DHCNTT dated August 30, 2019 of the Rector',
 'Đại học Công nghệ Thông tin về việc ban hành Quy chế đào tạo theo học chế tín chỉ cho': 'Information Technology University on the promulgation of the training regulations according to the credit system for',
 'hệ Đại học chính quy của Trường Đại học Công nghệ Thông tin;': 'regular university system of the University of Information Technology;',
 'Căn cứ Nghị quyết số 06/NQ-HĐTĐHCNTT ngày 11/12/2023 của Hội đồng Trường': 'According to Resolution No. 06/NQ-HĐTĐHCNTT dated December 11, 2023 of the University Council',
 'Đại học Công nghệ Thông tin;': 'University of Information Technology;',
 'Theo đề nghị của Trưởng Phòng Kế hoạch - Tài chính,': 'According to the proposal of the Head of Planning and Finance Department,',
 'QUYẾT ĐỊNH:': 'DECISION:',
 'Điều 1. Điều chỉnh mức thu học phí trình độ đào tạo đại học chính quy chương': 'Article 1. Adjusting the tuition fees for regular undergraduate training levels.',
 'trình chuẩn từ KHÓA 2020 trở về trước trong năm học 2023-2024 như sau:': 'present standards from the 2020 cohort and earlier in the academic year 2023-2024 as follows:',
 'STT': 'No.',
 'Nội dung thu': 'Revenue content',
 'Mức thu': 'Revenue level',
 'Khóa áp dụng': 'Applied cohort',
 'I': 'I',
 'Học phí trong học kỳ chính': 'Tuition fee in the main semester',
 '1': '1',
 'Học phí học kỳ chính': 'Main semester tuition fee',
 '14.500.000 đồng/Năm học': '14,500,000 Vietnamese dong/academic year',
 'Khóa 2020': '2020 cohort',
 'Từ Khóa 2019': 'Keyword 2019',
 '2': '2',
 'Học phí học mới': 'New tuition fees',
 '430.000 đồng/Tín chỉ học phí': '430,000 VND per credit tuition fee',
 'trở về trước': 'Go back to the previous state',
 'Học phí học lại, học cải': 'Tuition for retake, remedial study',
 'Từ Khóa 2020': 'Keyword 2020',
 '3': '3',
 'thiện điểm (tất cả các môn': 'improvement points (all subjects)',
 'học)': 'study)',
 'II': 'II',
 'Học phí trong học kỳ hè/Học phí ngoài giờ hành chính': 'Tuition fee during summer semester/After-hours tuition fee',
 'Học phí học mới, học lại,': 'New tuition fee, retake tuition fee,',
 'Đơn giá học phí (Tương ứng': 'Unit price of tuition fee (Corresponding',
 'học cải thiện điểm': 'study for grade improvement',
 'tại khoản 2, 3 - mục I) X 1.5': 'in clause 2, 3 - item I) multiplied by 1.5',
 '- Đối với các môn học có sĩ số sinh viên dưới chuẩn, học phí sẽ được áp dụng': '- For subjects with student enrollment below the standard, tuition fees will be applied',
 'theo quy định riêng của Nhà trường.': 'according to the specific regulations of the University.',
 'Điều 2. Quyết định này thay thế Quyết định số 737/QĐ-ĐHCNTT ngày': 'Article 2. This decision replaces Decision No. 737/QD-DHCNTT dated',
 '24/07/2023. Quyết định có hiệu lực kể từ ngày ký.': '24/07/2023. The decision shall take effect from the date of signing.',
 'Điều 3. Các Ông/Bà Trưởng phòng Kế hoạch -Tài chính, Trưởng các Phòng,': 'Article 3. Mr./Ms. Heads of Planning - Finance Department, Heads of Departments,',
 'Khoa, Bộ môn có liên quan và sinh viên trình độ đào tạo đại học chính quy chương': 'Faculties, relevant Departments, and regular undergraduate students',
 'trình chuẩn từ Khóa 2020 trở về trước chịu trách nhiệm thi hành quyết định này./.': 'who meet the standard requirements from the 2020 intake and earlier are responsible for implementing this decision./.',
 'KT. HIỆU TRƯỞNG': "President's Office",
 'Nơi nhận:': 'Recipient:',
 'Như Điều 3;': 'As stated in Article 3;',
 'PHÓ HIỆU TRƯỞNG': 'Vice President',
 'Lưu: VT, KHTC.': 'Save: VT, KHTC.',
 '(Đã ký)': '(Signed)',
 'Nguyễn Tấn Trần Minh Khang': 'Nguyen Tan Tran Minh Khang'}
    }
    if sentence in mapping[language]:
        return mapping[language][sentence]
    else:
        raise ValueError("FAIL")
