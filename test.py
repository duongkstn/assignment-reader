from reader.converter import Converter
from reader.predict import Reader
from reader.utils import visual_result

reader = Reader()
pdf_path = './Doc2_AI_Translate_VN.pdf'
with open(pdf_path, 'rb') as f:
    file_bytes = f.read()

imgs = Converter().from_bytes(file_bytes)[0]
pages = reader.process_document(file_bytes)
for page, img in zip(pages, imgs):
    ocr_results, bboxes = page
    visual_result(img, bboxes, ocr_results)
