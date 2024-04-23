import cv2
import torch
import os
from reader.ocr.vietocr.tool.translate import build_model, translate
from argparse import Namespace

def to_tensor(img):
    # handle numpy array
    if img.ndim == 2:
        img = img[:, :, None]

    img = torch.from_numpy(img.transpose((2, 0, 1)))
    # backward compatibility
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


class Predictor:
    def __init__(self):
        current_file_path = os.path.realpath(__file__)
        current_dir = os.path.dirname(current_file_path)
        current_dir = os.path.abspath(current_dir)
        ocr_config = {
            "image_channel_size": 3,
            "height": 32,
            "vocab": "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ",
            "device": "cuda",
            "gpu": 0,
            "seq_modeling": "seq2seq",
            "transformer": {
                "decoder_embedded": 256,
                "decoder_hidden": 256,
                "dropout": 0.1,
                "encoder_hidden": 256,
                "img_channel": 256
            },
            "predictor": {
                "beamsearch": False
            },
            "quiet": False,
            "backbone": "vgg19_bn",
            "cnn": {
                "hidden": 256,
                "ks": [
                    [
                        2,
                        2
                    ],
                    [
                        2,
                        2
                    ],
                    [
                        2,
                        1
                    ],
                    [
                        2,
                        1
                    ],
                    [
                        2,
                        1
                    ]
                ],
                "pretrained": False,
                "ss": [
                    [
                        2,
                        2
                    ],
                    [
                        2,
                        2
                    ],
                    [
                        2,
                        1
                    ],
                    [
                        2,
                        1
                    ],
                    [
                        2,
                        1
                    ]
                ]
            },
            "weights": os.path.join(current_dir, '../../weights/transformerocr.pt')
        }

        self.config = Namespace(**ocr_config)
        model, vocab = build_model(self.config)
        self.device = self.config.device if torch.cuda.is_available() else 'cpu'

        model.load_state_dict(torch.load(self.config.weights, map_location=torch.device(self.device)))

        self.model = model
        self.vocab = vocab

    def translate_image(self, imgs, debug=False):
        sents, probs = translate(imgs, self.model, debug=debug)
        sents = [self.vocab.decode(s.tolist()) for s in sents]
        probs = probs.tolist()
        return sents, probs

    def process_batch(self, imgs, batch_size=32):
        batch_lines = []
        line_to_id = []
        predictions = [None for _ in range(len(imgs))]
        scores = [None for _ in range(len(imgs))]
        for idx, img in enumerate(imgs):
            w = int(round(32 * img.shape[1] / img.shape[0]))
            img = cv2.resize(img, (w, 32))

            img = to_tensor(img)
            line_to_id.append(idx)
            batch_lines.append(img)
        if len(batch_lines) != 0:
            n_samples = len(batch_lines)
            torch.set_num_threads(torch.get_num_threads())
            ####### sort by width #########
            sorted_batch_lines = zip(batch_lines, range(n_samples))
            sorted_batch_lines = sorted(sorted_batch_lines, key=lambda x: x[0].shape[2])
            preds, accs = ['' for _ in range(n_samples)], [0 for _ in range(n_samples)]
            for sorted_batch_imgs in [sorted_batch_lines[i:i + batch_size] for i in range(0, n_samples, batch_size)]:
                batch_img, index_imgs = zip(*sorted_batch_imgs)
                c = batch_img[0].size(0)
                h = max([t.size(1) for t in batch_img])
                w = max([t.size(2) for t in batch_img])
                batch_img_tensor = torch.zeros(len(batch_img), c, h, w).fill_(1)
                for i, img in enumerate(batch_img):
                    batch_img_tensor[i, :, 0:img.size(1), 0:img.size(2)] = img
                # print(batch_img_tensor.shape)
                if self.device == 'cuda' and torch.cuda.is_available():
                    batch_img_tensor = batch_img_tensor.cuda()
                _preds, _accs = self.translate_image(batch_img_tensor)
                for i, p, a in zip(index_imgs, _preds, _accs):
                    preds[i] = p
                    accs[i] = a
            for i, (pred, acc) in enumerate(zip(preds, accs)):
                predictions[line_to_id[i]] = pred
                scores[line_to_id[i]] = acc
        return predictions, scores
