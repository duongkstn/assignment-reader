import io
from xml.dom import minidom

import PyPDF2
import numpy as np
import pdf2image
import visen
import time

from reader.pdfhandle import PDFHandle


class Converter:
    """Converter."""

    def __init__(self):
        """Initialize."""
        super(Converter, self).__init__()

    def from_pdf(
        self,
        pdf,
        page_limit=None,
        dpi=200,
        thread_count=1,
        page_indices=None,
        try_get_content=False
    ):
        """Convert pdf file to image using PyPDF2.

        Arguments:
            pdf: pdf file.
            page_limit: limit number of pages.
            dpi (int): dpi for convert to image.
            thread_count (int): number of threads.
        Returns:
            images (list): list RGB image.
            real_pages: list numpy array (h,w,c).
            content.
        """
        t0 = time.time()
        content = None
        pdfReader = PyPDF2.PdfFileReader(pdf, strict=False)
        if page_indices is not None:
            page_indices = list(page_indices)
            return [self.pdf_page_to_png(pdfReader, i, dpi=dpi) for i in page_indices if
                    i < pdfReader.numPages], pdfReader.numPages, None
        real_pages = pdfReader.numPages
        if page_limit is not None:
            num_pages = min([page_limit, real_pages])
        else:
            num_pages = real_pages
        images = self.crop_and_convert(pdfReader, num_pages, dpi=dpi, thread_count=thread_count)
        t1 = time.time()
        if try_get_content:
            content = PDFHandle.process_pdf_file(pdf, imgs=images, by=self.by)
        t2 = time.time()
        print(f"pdf2image: {t1 - t0}")
        print(f"pdf2content: {t2 - t1}")

        return images, real_pages, content

    def from_bytes(self, file_content, page_limit=1e6, dpi=200, thread_count=1, page_indices=None,
                   try_get_content=False):
        """Convert byte data to image.

        Arguments:
            file_content: bytes.
            page_limit (int): limit number of pages.
            dpi (int): dpi for convert to image.
            thread_count (int): number of threads
        Returns:
            images: list RGB image.
            real_pages: list numpy array (h,w,c).
            content.
        """
        content = None
        pdf = io.BytesIO(file_content)
        pdfReader = PyPDF2.PdfFileReader(pdf, strict=False)
        if page_indices is not None:
            page_indices = list(page_indices)
            return [self.pdf_page_to_png(pdfReader, i, dpi=dpi) for i in page_indices if
                    i < pdfReader.numPages], pdfReader.numPages, content
        real_pages = pdfReader.numPages
        if page_limit is not None:
            num_pages = min([page_limit, real_pages])
        else:
            num_pages = real_pages
        del pdf, pdfReader
        images = pdf2image.convert_from_bytes(
            file_content,
            first_page=1,
            last_page=num_pages,
            dpi=dpi,
            thread_count=thread_count
        )
        images = [np.ascontiguousarray(np.asarray(image)[:, :, ::-1]) for image in images]

        if try_get_content:
            content = PDFHandle.process_pdf_content(file_content, imgs=images)
        return images, real_pages, content

    def from_pdf_old(self, pdf, page_limit=None, dpi=200, page_indices=None):
        """Convert pdf file to image using PyPDF2.

        Arguments:
            pdf: pdf file.
            page_limit: limit number of pages.
            dpi (int): dpi for convert to image.
            page_indices (list): page indices.
        Returns:
            list RGB image, list numpy array (h,w,c).
        """
        pdfReader = PyPDF2.PdfFileReader(pdf, strict=False)
        if page_indices is not None:
            page_indices = list(page_indices)
            return [self.pdf_page_to_png(pdfReader, i, dpi=dpi) for i in page_indices if
                    i < pdfReader.numPages], pdfReader.numPages

        if (page_limit is not None) and (page_limit < pdfReader.numPages):
            png_images = [self.pdf_page_to_png(pdfReader, i, dpi=dpi) for i in range(page_limit)]
            return png_images, pdfReader.numPages
        else:
            png_images = [
                self.pdf_page_to_png(pdfReader, i, dpi=dpi) for i in range(pdfReader.numPages)
            ]
            return png_images, pdfReader.numPages

    def from_pdf_path(self, pdf_path, page_limit=None, dpi=200, page_indices=None):
        """Convert pdf to images.

        Arguments:
            pdf_path: direction to pdf file path.
            page_limit : number of limit page.
            dpi (int): dpi for convert to image.
        Returns:
            tuple (list RGB image,number_images of original pdf).
        """
        pdf_file_path = open(pdf_path, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdf_file_path, strict=False)

        if page_indices is not None:
            page_indices = list(page_indices)
            return [self.pdf_page_to_png(pdfReader, i, dpi=dpi) for i in page_indices if
                    i < pdfReader.numPages], pdfReader.numPages

        if (page_limit is not None) and (page_limit < pdfReader.numPages):
            png_images = [self.pdf_page_to_png(pdfReader, i, dpi=dpi) for i in range(page_limit)]
            return png_images, pdfReader.numPages
        else:
            images = pdf2image.convert_from_path(pdf_path)
            return [np.asarray(image)[:, :, ::-1] for image in images], pdfReader.numPages

    def pdf_page_to_png(self, src_pdf, pagenum=0, dpi=200):
        """Convert pdf file to PNG image.

        Arguments:
            src_pdf:
            pagenum (int): default 0.
            dpi (int): dpi for convert to image, default 200.
        Returns:
            specified PDF page as wand.image.Image png.
        """
        dst_pdf = PyPDF2.PdfFileWriter()
        dst_pdf.addPage(src_pdf.getPage(pagenum))

        r = io.BytesIO()
        dst_pdf.write(r)

        images = pdf2image.convert_from_bytes(r.getvalue(), dpi=dpi)

        return np.asarray(images[0])[:, :, ::-1]

    def crop_and_convert(self, src_pdf, num_pages, dpi=200, thread_count=1):
        """Convert a pdf file to image list with limit specified.

        Arguments:
            src_pdf: pdf file reader.
            num_pages: number of pages.
            dpi: dpi.
            thread_count: number of threads.
        Returns:
            list of images.
        """
        dst_pdf = PyPDF2.PdfFileWriter()
        for pagenum in range(0, num_pages):
            dst_pdf.addPage(src_pdf.getPage(pagenum))
        r = io.BytesIO()
        dst_pdf.write(r)
        # r.seek(0)
        # images = pdf2image.convert_from_bytes(r.read(), dpi=dpi, thread_count=thread_count)
        images = pdf2image.convert_from_bytes(r.getvalue(), dpi=dpi, thread_count=thread_count)
        return [np.ascontiguousarray(np.asarray(image)[:, :, ::-1]) for image in images]


def xml2textlines(xml_path, n_page, hreal, wreal):
    """Convert XML to text lines."""
    pages = []
    mydoc = minidom.parse(xml_path)
    for page in mydoc.getElementsByTagName('page')[:n_page]:
        pages.append(get_text_lines(page, hreal, wreal))
    return pages


def get_text_lines(page, hreal, wreal):
    """Get text lines."""
    ocr_results, lines = [], []
    bbox = list(map(float, page.attributes['bbox'].value.split(',')))
    bbox = list(map(int, bbox))
    h, w = bbox[3], bbox[2]
    hscale = hreal / h
    wscale = wreal / w
    itemlist = page.getElementsByTagName('textline')
    for textline in itemlist:
        characters = textline.getElementsByTagName('text')
        text = ''
        for c in characters:
            try:
                text += c.firstChild.nodeValue
            except:
                text += ' '
        text = visen.clean_tone(text)
        bbox = list(map(float, textline.attributes['bbox'].value.split(',')))
        bbox = list(map(int, bbox))
        bbox[1], bbox[3] = int((h - bbox[3]) * hscale), int((h - bbox[1]) * hscale)
        bbox[0], bbox[2] = int(bbox[0] * wscale), int(bbox[2] * wscale)
        if len(text) < 2 and bbox[3] - bbox[1] < bbox[2] - bbox[0]:
            continue
        ocr_results.append((text, 1.0))
        lines.append(bbox)
    if not ocr_results:
        return [], []
    s = zip(ocr_results, lines)
    s = sorted(s, key=lambda x: (x[1][1], x[1][0]))
    ocr_results, bbox = zip(*s)
    return ocr_results, lines


def get_number_of_papers(file_content):
    """Get number of pages."""
    doc = fitz.Document(stream=file_content, filetype=".pdf")
    page_count = doc.page_count
    doc.close()
    return page_count
