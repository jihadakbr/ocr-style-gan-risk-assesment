import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

from rapidfuzz.distance import Levenshtein

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline.ekyc import process_ekyc
from src.pipeline.financial import process_bank_statement
from src.pipeline.business import process_business_summary

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
for name in ["ppocr", "paddle", "matplotlib", "PIL", "tensorflow", "deepface", "optimum"]:
    logging.getLogger(name).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def load_identity_gt(bundle_dir: Path) -> dict:
    """Parse identity.txt into a dict."""
    gt = {}
    path = bundle_dir / "identity.txt"
    if not path.exists():
        return gt
    for line in path.read_text().splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            gt[key.strip()] = value.strip()
    return gt


def load_json_gt(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def normalize(text: str | None) -> str:
    """Normalize text for comparison: uppercase, strip, collapse whitespace."""
    if text is None:
        return ""
    return " ".join(text.upper().split())


def character_accuracy(ocr_text: str | None, gt_text: str | None) -> float:
    """Compute 1 - CER (Character Error Rate). Returns 0.0–1.0."""
    a = normalize(ocr_text)
    b = normalize(gt_text)
    if not b:
        return 1.0 if not a else 0.0
    return max(0.0, 1.0 - Levenshtein.distance(a, b) / len(b))


def numeric_within_tolerance(ocr_val, gt_val, tolerance=0.10) -> bool:
    """Check if OCR numeric value is within tolerance % of ground truth."""
    if ocr_val is None or gt_val is None:
        return False
    if gt_val == 0:
        return ocr_val == 0
    return abs(ocr_val - gt_val) / abs(gt_val) <= tolerance


STAGE1_FIELD_MAP = {
    "nik": "nik",
    "name": "nama",
    "place_of_birth": "tempat_tgl_lahir",
    "gender": "jenis_kelamin",
    "address": "alamat",
    "rt_rw": "rt_rw",
    "kelurahan": "kelurahan",
    "kecamatan": "kecamatan",
    "religion": "agama",
    "marital_status": "status_perkawinan",
    "occupation": "pekerjaan",
    "kewarganegaraan": "kewarganegaraan",
    "berlaku_hingga": "berlaku_hingga",
}


def evaluate_stage1(bundle_dir: Path, ekyc_result: dict, identity_gt: dict) -> dict:
    """Evaluate Stage 1 (eKYC) accuracy."""
    results = {}
    ktp_fields = ekyc_result.get("ktp_fields", {})

    for gt_key, ocr_key in STAGE1_FIELD_MAP.items():
        gt_val = identity_gt.get(gt_key, "")
        ocr_val = ktp_fields.get(ocr_key)

        if gt_key == "place_of_birth" and ocr_val and "," in ocr_val:
            ocr_val = ocr_val.split(",", 1)[0].strip()

        score = character_accuracy(ocr_val, gt_val)
        results[gt_key] = {"score": score, "match": score >= 0.9, "ocr": ocr_val, "gt": gt_val}

    face = ekyc_result.get("face_match", {})
    face_ok = face.get("verified", False)
    face_gt = identity_gt.get("face_match", "True") == "True"
    face_correct = face_ok == face_gt
    results["face_verified"] = {
        "score": 1.0 if face_correct else 0.0,
        "match": face_correct,
        "ocr": str(face_ok),
        "gt": str(face_gt),
    }

    return results


def evaluate_stage2(bundle_dir: Path, financial_result: dict, bank_gt: dict) -> dict:
    """Evaluate Stage 2 (Financial) accuracy."""
    results = {}

    gt_header = bank_gt.get("header", {})
    ocr_header = financial_result.get("header", {})

    for field in ["bank_name", "account_holder", "account_number"]:
        ocr_val = ocr_header.get(field)
        gt_val = gt_header.get(field)

        if field == "bank_name" and ocr_val:
            ocr_val = ocr_val.removeprefix("BANK ").removeprefix("Bank ")

        score = character_accuracy(ocr_val, gt_val)
        results[field] = {
            "score": score,
            "match": score >= 0.9,
            "ocr": ocr_header.get(field),
            "gt": gt_val,
        }

    gt_metrics = bank_gt.get("metrics", {})
    ocr_metrics = financial_result.get("metrics", {})

    for key in ["total_credit", "total_debit", "net_cash_flow"]:
        gt_val = gt_metrics.get(key)
        ocr_val = ocr_metrics.get(key)
        match = numeric_within_tolerance(ocr_val, gt_val, 0.10)
        results[key] = {
            "score": 1.0 if match else 0.0,
            "match": match,
            "ocr": ocr_val,
            "gt": gt_val,
        }

    gt_count = gt_metrics.get("transaction_count", 0)
    ocr_count = ocr_metrics.get("transaction_count", 0)
    count_match = abs(ocr_count - gt_count) <= 3
    results["transaction_count"] = {
        "score": 1.0 if count_match else 0.0,
        "match": count_match,
        "ocr": ocr_count,
        "gt": gt_count,
    }

    return results


def evaluate_stage3(bundle_dir: Path, business_result: dict, business_gt: dict) -> dict:
    """Evaluate Stage 3 (Business) accuracy."""
    results = {}

    gt_pnl = business_gt.get("pnl", {})
    ocr_pnl = business_result.get("pnl", {})

    for key in ["pendapatan", "hpp", "laba_kotor", "beban_operasional", "laba_bersih"]:
        gt_val = gt_pnl.get(key)
        ocr_val = ocr_pnl.get(key)
        match = numeric_within_tolerance(ocr_val, gt_val, 0.10)
        results[key] = {
            "score": 1.0 if match else 0.0,
            "match": match,
            "ocr": ocr_val,
            "gt": gt_val,
        }

    gt_margin = gt_pnl.get("profit_margin_pct")
    ocr_margin = business_result.get("health_metrics", {}).get("profit_margin_pct")
    margin_match = False
    if gt_margin is not None and ocr_margin is not None:
        margin_match = abs(ocr_margin - gt_margin) <= 3.0
    results["profit_margin_pct"] = {
        "score": 1.0 if margin_match else 0.0,
        "match": margin_match,
        "ocr": ocr_margin,
        "gt": gt_margin,
    }

    return results


def print_stage_report(stage_name: str, all_results: list[dict]):
    """Print accuracy summary for one stage."""
    if not all_results:
        print(f"\n{stage_name}")
        print("  No results (ground truth files missing?)")
        return

    fields = list(all_results[0].keys())
    total = len(all_results)
    all_scores = []

    print(f"\n{stage_name}")
    print(f"  {'field':<28s} {'avg score':>10s}  {'match rate':>10s}")
    print(f"  {'-----':<28s} {'---------':>10s}  {'----------':>10s}")
    for field in fields:
        scores = [r[field]["score"] for r in all_results]
        avg_score = sum(scores) / len(scores)
        matches = sum(1 for s in scores if s >= 0.9)
        all_scores.extend(scores)
        print(f"  {field:<28s} {avg_score*100:9.1f}%  {matches:>4d}/{total:<4d}")

    if all_scores:
        stage_avg = sum(all_scores) / len(all_scores) * 100
        print(f"  {'--- stage average ---':<28s} {stage_avg:9.1f}%")

    return sum(all_scores), len(all_scores)


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR pipeline accuracy against ground truth")
    parser.add_argument("--data-dir", default="data/raw", help="Path to applicant bundles")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N bundles (0 = all)")
    parser.add_argument("--output-dir", default="data/evaluation_results", help="Directory for detailed JSON results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    bundle_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("applicant_")])

    if not bundle_dirs:
        print(f"No applicant bundles found in {data_dir}")
        sys.exit(1)

    if args.limit > 0:
        bundle_dirs = bundle_dirs[:args.limit]

    print(f"Evaluating {len(bundle_dirs)} applicant bundles from {data_dir}/\n")

    stage1_results = []
    stage2_results = []
    stage3_results = []
    detailed = []

    for i, bundle_dir in enumerate(bundle_dirs):
        label = bundle_dir.name
        print(f"[{i+1}/{len(bundle_dirs)}] Processing {label}...", end=" ", flush=True)

        identity_gt = load_identity_gt(bundle_dir)
        bank_gt = load_json_gt(bundle_dir / "bank_statement_gt.json")
        business_gt = load_json_gt(bundle_dir / "business_summary_gt.json")

        applicant_detail = {"applicant": label}

        # Stage 1: eKYC
        ktp_path = bundle_dir / "ktp.jpg"
        face_path = bundle_dir / "face.jpg"
        if ktp_path.exists() and face_path.exists() and identity_gt:
            try:
                ekyc_result = process_ekyc(str(ktp_path), str(face_path))
                s1 = evaluate_stage1(bundle_dir, ekyc_result, identity_gt)
                stage1_results.append(s1)
                applicant_detail["stage1"] = s1
            except Exception as e:
                print(f"Stage 1 error: {e}", end=" ")

        # Stage 2: Financial
        bank_path = bundle_dir / "bank_statement.jpg"
        if bank_path.exists() and bank_gt:
            try:
                financial_result = process_bank_statement(str(bank_path))
                s2 = evaluate_stage2(bundle_dir, financial_result, bank_gt)
                stage2_results.append(s2)
                applicant_detail["stage2"] = s2
            except Exception as e:
                print(f"Stage 2 error: {e}", end=" ")

        # Stage 3: Business
        biz_path = bundle_dir / "business_summary.jpg"
        if biz_path.exists() and business_gt:
            try:
                business_result = process_business_summary(str(biz_path))
                s3 = evaluate_stage3(bundle_dir, business_result, business_gt)
                stage3_results.append(s3)
                applicant_detail["stage3"] = s3
            except Exception as e:
                print(f"Stage 3 error: {e}", end=" ")

        detailed.append(applicant_detail)
        print("done")

    print("\n" + "=" * 60)
    print("ACCURACY REPORT")
    print("=" * 60)

    grand_score_sum = 0.0
    grand_count = 0

    r1 = print_stage_report("Stage 1 — eKYC (KTP Fields + Face Match)", stage1_results)
    if r1:
        grand_score_sum += r1[0]
        grand_count += r1[1]

    r2 = print_stage_report("Stage 2 — Financial (Bank Statement)", stage2_results)
    if r2:
        grand_score_sum += r2[0]
        grand_count += r2[1]

    r3 = print_stage_report("Stage 3 — Business (P&L + Metrics)", stage3_results)
    if r3:
        grand_score_sum += r3[0]
        grand_count += r3[1]

    if grand_count > 0:
        print(f"\n{'Overall average accuracy:':<28s} {grand_score_sum / grand_count * 100:5.1f}%")

    # Save detailed results
    wib = timezone(timedelta(hours=7))
    timestamp = datetime.now(wib).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"evaluation_results_{len(bundle_dirs)}applicants_{timestamp}.json"

    # Convert non-serializable values for JSON output
    def _make_serializable(obj):
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        if isinstance(obj, float) and (obj != obj):  # NaN
            return None
        return obj

    output_path.write_text(json.dumps(_make_serializable(detailed), indent=2, default=str))
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
