from reader.converter import Converter
from reader.predict import Reader
from reader.utils import visual_result, translate, map_translate
from tqdm import tqdm
reader = Reader()
pdf_path = './Doc1_AI_Translate_VN.pdf'
language = "english"
with open(pdf_path, 'rb') as f:
    file_bytes = f.read()

imgs = Converter().from_bytes(file_bytes)[0]

pages = reader.process_document(file_bytes)

all_sents = []
for page, img in zip(pages, imgs):
    ocr_results, bboxes = page
    # all_sents.extend([x[0] for x in ocr_results])
    new_ocr_results = []
    for x0, x1 in tqdm(ocr_results):
        translated_x0 = map_translate(x0, language)
        all_sents.append((x0, translated_x0))
        new_ocr_results.append((translated_x0, x1))
    visual_result(img, bboxes, new_ocr_results)

print("all_sents = ", all_sents)
