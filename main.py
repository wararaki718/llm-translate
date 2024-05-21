from typing import List

from downloader import PDFDownloader
from extractor import PDFTextExtractor


def show(texts: List[str]) -> None:
    print(f"number of pages: {len(texts)}")
    print("contents:")
    for i, text in enumerate(texts, start=1):
        print(f"  #{i} -> {text[:20]}...{text[-20:]}")
    print()


def main():
    uri = "https://arxiv.org/pdf/2109.10086"

    downloader = PDFDownloader()
    pdf = downloader.download(uri)
    print(f"'{uri}' is downloaded!")
    print()

    extractor = PDFTextExtractor()
    texts = extractor.extract(pdf)
    for i, text in enumerate(texts):
        with open(f"texts/text_{i}.txt", "wt") as f:
            f.write(text)

    show(texts)

    print("DONE")


if __name__ == "__main__":
    main()
