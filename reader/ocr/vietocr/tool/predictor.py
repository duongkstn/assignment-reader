import numpy as np
import torch

from reader.ocr.vietocr.tool.translate import build_model, translate, translate_beam_search, process_input
from reader.ocr.vietocr.tool.utils import download_weights


class Predictor:
    def __init__(self, config):

        device = config['device']

        model, vocab = build_model(config)

        if config['weights'].startswith('http'):
            weights = download_weights(config['weights'])
        else:
            weights = config['weights']

        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab

    def process_img(self, img, **kwargs):
        img = process_input(img, self.config['dataset']['image_height'],
                            self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])
        img = img[np.newaxis, ...]
        img = img.to(self.config['device'])

        if self.config['predictor']['beamsearch']:
            sent = translate_beam_search(img, self.model)
            s = sent
            prob = None
        else:
            s, prob = translate(img, self.model)
            s = s[0].tolist()
            prob = prob[0]

        s = self.vocab.decode(s)

        return s, prob

    def process_batch(self, imgs, batch_size=32, **kwargs):
        predictions = [None for _ in range(len(imgs))]
        scores = [None for _ in range(len(imgs))]
        batch_lines = [process_input(img, self.config['dataset']['image_height'],
                                     self.config['dataset']['image_min_width'],
                                     2000)
                       for img in imgs]
        batch_lines = sorted(zip(range(len(batch_lines)), batch_lines), key=lambda k: k[1].shape[2])
        for _batch_lines in [batch_lines[i:i + batch_size] for i in range(0, len(batch_lines), batch_size)]:
            c = 3
            h = self.config['dataset']['image_height']
            w = max([t[1].size(2) for t in _batch_lines])
            batch_imgs = torch.zeros(len(_batch_lines), c, h, w).fill_(1)
            for i, img in enumerate(_batch_lines):
                batch_imgs[i, :, :, :img[1].shape[2]] = img[1]
            batch_imgs = batch_imgs.to(self.config['device'])
            sents, probs = translate(batch_imgs, self.model)
            sents = [self.vocab.decode(s.tolist()) for s in sents]
            probs = probs.tolist()
            for idx, (i, _) in enumerate(_batch_lines):
                predictions[i] = sents[idx]
                scores[i] = probs[idx]

        return predictions, scores
