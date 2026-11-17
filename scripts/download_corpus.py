import os, time, urllib.request
import pandas as pd

PAPERS = {
    "doc01_rag_original":      "https://arxiv.org/pdf/2005.11401.pdf",
    "doc02_rag_survey":        "https://arxiv.org/pdf/2312.10997.pdf",
    "doc03_dpr":               "https://arxiv.org/pdf/2004.04906.pdf",
    "doc04_ragas":             "https://arxiv.org/pdf/2309.15217.pdf",
    "doc05_self_rag":          "https://arxiv.org/pdf/2310.11511.pdf",
    "doc06_transformer":       "https://arxiv.org/pdf/1706.03762.pdf",
    "doc07_lora":              "https://arxiv.org/pdf/2106.09685.pdf",
    "doc08_flash_attention":   "https://arxiv.org/pdf/2205.14135.pdf",
    "doc09_rag_survey_huang":  "https://arxiv.org/pdf/2404.10981.pdf",
    "doc10_comprehensive_rag": "https://arxiv.org/pdf/2410.12837.pdf",
}

def download_all(pdf_dir="data/raw_pdfs", proc_dir="data/processed"):
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Research Benchmark Script)")]
    urllib.request.install_opener(opener)

    for doc_id, url in PAPERS.items():
        dest = os.path.join(pdf_dir, f"{doc_id}.pdf")
        if not os.path.exists(dest):
            print(f"Downloading {doc_id}...")
            urllib.request.urlretrieve(url, dest)
            time.sleep(1)

    manifest = pd.DataFrame([
        {"doc_id": k, "url": v, "filename": f"{k}.pdf"} for k, v in PAPERS.items()
    ])
    manifest.to_csv(os.path.join(proc_dir, "corpus_manifest.csv"), index=False)

if __name__ == "__main__":
    download_all()
