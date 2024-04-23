## PDF Reader


### change log
 - version 1.0.0 : upload repo, sử dụng vietocr và paddle ocr để đọc file pdf, 

### Requirements
paddlepaddle

### Installation
Note: Activate your environment (if necessary)

cd <path_to_root_project> (folder containing setup.py)
```bash
pip install -r requirements.txt
pip install -e .
```

### Instruction
khởi tạo instance Reader, kết quả trả về xem ở hàm process_document

```python
from reader.predict import Reader

reader = Reader()
pdf_path = './Doc2_AI_Translate_VN.pdf'
with open(pdf_path, 'rb') as f:
    file_bytes = f.read()

print(reader.process_document(file_bytes))
```

### Demo
```bash
python test.py
```

### train
https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/docs/tutorials/GETTING_STARTED.md
