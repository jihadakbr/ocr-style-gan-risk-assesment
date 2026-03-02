# OCR + StyleGAN2 — SME Lending Risk Assessment

An automated document processing pipeline for Indonesian SME lending. Applicants submit phone photos of their documents, and the system extracts structured data via OCR, verifies identity with face matching, and computes a credit risk score.

The dataset is fully synthetic — KTP cards use GAN-generated faces, and all documents are degraded to simulate real phone-captured photos of printed paper.

## Demo

[![YouTube Demo](https://img.shields.io/badge/YouTube-Demo-red?logo=youtube)](https://youtu.be/M1L2e7sYLoY)  
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Synthetic_Dataset_Generator-blue?logo=kaggle)](https://www.kaggle.com/code/jihadakbr/ocr-style-gan-dataset-generator)

## How It Works

```
Applicant submits 3 documents (phone photos)
    │
    ├─► KTP + Face photo ──► PaddleOCR extracts fields + DeepFace verifies identity
    │                         └─► Identity score: 0–30 pts
    │
    ├─► Bank Statement ────► PaddleOCR extracts transactions + computes metrics
    │                         └─► Financial score: 0–40 pts
    │
    └─► Business Summary ──► PaddleOCR extracts P&L + DePlot reads revenue chart
                              └─► Business score: 0–30 pts
                                        │
                          credit_score = sum of all 3 (0–100)
                          risk_score   = 100 - credit_score
                                        │
                         ┌──────────────┼──────────────┐
                         │              │              │
                     0–25: LOW      26–50: MED     51–100: HIGH
```

## Project Structure

```
ocr_style_gan/
├── requirements.txt
├── notebooks/
│   └── ocr_style_gan_dataset_generator.ipynb   # Kaggle GPU notebook (data generation)
├── src/
│   ├── pipeline/
│   │   ├── ekyc.py            # Stage 1: KTP OCR + face matching
│   │   ├── financial.py       # Stage 2: Bank statement table OCR
│   │   └── business.py        # Stage 3: Revenue chart (DePlot) + P&L OCR
│   ├── evaluation/
│   │   └── accuracy.py        # Accuracy evaluation against ground truth
│   └── dashboard/
│       └── app.py             # Streamlit risk assessment dashboard
└── data/
    └── raw/                   # Generated applicant bundles (not committed)
```

Each applicant bundle contains:
```
applicant_XXXX/
├── face.jpg                   # GAN-generated face (StyleGAN2-ADA FFHQ)
├── ktp.jpg                    # Synthetic KTP card with degradation
├── bank_statement.jpg         # Degraded bank statement (crumpled paper effect)
├── business_summary.jpg       # Degraded business report with chart + P&L
├── identity.txt               # Ground truth: KTP fields
├── bank_statement_gt.json     # Ground truth: transactions, header, metrics
└── business_summary_gt.json   # Ground truth: P&L, monthly revenue, margins
```

## Setup

Requires Python 3.12+.

```bash
git clone <repo-url>
cd ocr_style_gan
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

PaddleOCR models (~30MB) and DeepFace VGG-Face weights (~580MB) download automatically on first run.

## Usage

### 1. Generate the synthetic dataset (Kaggle notebook)

The dataset is generated on a Kaggle T4 GPU because StyleGAN2 requires GPU inference. The notebook handles the full synthetic data pipeline:

- Generates realistic face photos using **StyleGAN2-ADA** (FFHQ 1024x1024), filtered by **DeepFace** to reject child-looking faces (age < 25) and detect gender
- Renders **KTP cards** with the generated face + 13 identity fields using Pillow and Faker(`id_ID`)
- Renders **bank statements** with transaction tables and **business summaries** with revenue charts (matplotlib) + P&L tables
- Applies **phone-photo degradation** to all documents — crumpled paper texture, uneven lighting, perspective distortion, shadows, sensor noise, and JPEG compression
- Outputs ground truth files (`identity.txt`, `bank_statement_gt.json`, `business_summary_gt.json`) for accuracy evaluation

To run it:

1. Upload `notebooks/ocr_style_gan_dataset_generator.ipynb` to [Kaggle](https://www.kaggle.com/) or use the published notebook linked above
2. Enable **GPU accelerator** (T4) in notebook settings
3. Run all cells — generates 500 applicant bundles by default
4. Download the output zip and extract it:
   ```bash
   unzip applicant_bundles.zip -d data/raw/
   ```

### 2. Run the dashboard

```bash
streamlit run src/dashboard/app.py
```

Open http://localhost:8501, select an applicant, and click **Run OCR Pipeline**. The dashboard shows extracted data from all 3 stages, a risk score, and a human-in-the-loop correction interface for fixing OCR errors.

### 3. Run pipelines individually

```python
from src.pipeline.ekyc import process_ekyc
from src.pipeline.financial import process_bank_statement
from src.pipeline.business import process_business_summary

ekyc = process_ekyc("data/raw/applicant_0001/ktp.jpg", "data/raw/applicant_0001/face.jpg")
bank = process_bank_statement("data/raw/applicant_0001/bank_statement.jpg")
biz = process_business_summary("data/raw/applicant_0001/business_summary.jpg")
```

### 4. Run accuracy evaluation

```bash
# Quick test on first 5 bundles
python -m src.evaluation.accuracy --data-dir data/raw --limit 5

# Full evaluation
python -m src.evaluation.accuracy --data-dir data/raw
```

Runs all 3 OCR pipeline stages on each applicant bundle and compares the output against ground truth files. The evaluation uses different metrics depending on the field type:

| Stage | Fields | Metric |
|---|---|---|
| eKYC (text fields) | NIK, name, address, gender, religion, etc. | Character accuracy via Levenshtein distance (1 - CER), match if >= 90% |
| eKYC (face) | face_verified | Exact boolean match against expected value |
| Financial (header) | bank name, account holder, account number | Character accuracy (Levenshtein), match if >= 90% |
| Financial (metrics) | total credit, total debit, net cash flow | Numeric match within 10% tolerance |
| Financial (count) | transaction count | Exact match within ±3 |
| Business (P&L) | pendapatan, HPP, laba kotor, beban operasional, laba bersih | Numeric match within 10% tolerance |
| Business (margin) | profit margin % | Match within ±3 percentage points |

Reports per-field average scores, match rates, stage averages, and overall accuracy. Detailed per-applicant results are saved as JSON.

## Tech Stack

| Component | Tool |
|---|---|
| Face generation | StyleGAN2-ADA (FFHQ 1024×1024) via Kaggle T4 GPU |
| Face filtering | DeepFace (rejects age < 25, detects gender) |
| OCR engine | PaddleOCR (Indonesian language model) |
| Chart data extraction | Google DePlot (Pix2Struct), ONNX Runtime |
| Face matching | DeepFace (VGG-Face) |
| Document degradation | OpenCV, Pillow (crumpled paper, lighting, perspective, noise) |
| Dashboard | Streamlit |

## Risk Scoring

| Component | Max Points | Criteria |
|---|---|---|
| Identity verification | 30 | Face photo matches KTP card (DeepFace) |
| Financial health | 40 | Has income (+15), positive cash flow (+15), 10+ transactions (+10) |
| Business viability | 30 | Profitable (+15), profit margin >20% (+15) / >10% (+10) / >0% (+5) |

Risk score = 100 - total credit score. Lower risk score = safer to lend.
