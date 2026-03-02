import logging
import re
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
from deepface import DeepFace

logger = logging.getLogger(__name__)

# Initialize PaddleOCR once (lazy singleton)
_ocr_instance = None


def _get_ocr(force_new: bool = False) -> PaddleOCR:
    global _ocr_instance
    if _ocr_instance is None or force_new:
        _ocr_instance = PaddleOCR(lang="id")
    return _ocr_instance


def _safe_ocr(image, cls: bool = True):
    """Run OCR with retry on predictor crash (reinitializes the engine)."""
    try:
        return _get_ocr().ocr(image, cls=cls)
    except RuntimeError:
        logger.warning("PaddleOCR predictor crashed, reinitializing...")
        return _get_ocr(force_new=True).ocr(image, cls=cls)


# KTP field patterns — label → regex to match the value after it
KTP_FIELDS = {
    "nik": r"NIK\s*[:\-]?\s*(\d[\d\s]{10,})",
    "nama": r"Nama\s*[:\-]?\s*([A-Z][A-Z\s\.,]+?)(?:\n|Tempat|$)",
    "tempat_tgl_lahir": r"Tempat.*?Lahir\s*[:\-]?\s*(.+?)(?:\n|$)",
    "jenis_kelamin": r"Jenis\s*Kelamin\s*[:\-]?\s*(LAKI[\-\s]?LAKI|PEREMPUAN)",
    "alamat": r"Alamat\s*[:\-]?\s*(.+?)(?:\n|$)",
    "rt_rw": r"RT\s*[/\\|I]?\s*RW\s*[:\-]?\s*(\d{1,3}\s*[/\\|I]\s*\d{1,3})",
    "kelurahan": r"Kel.*?Desa\s*[:\-]?\s*(.+?)(?:\n|$)",
    "kecamatan": r"Kecamatan\s*[:\-]?\s*(.+?)(?:\n|$)",
    "agama": r"Agama\s*[:\-]?\s*(\w+)",
    "status_perkawinan": r"Status.*?Perkawinan\s*[:\-]?\s*(.+?)(?:\n|$)",
    "pekerjaan": r"Pekerjaan\s*[:\-]?\s*(.+?)(?:\n|$)",
    "kewarganegaraan": r"Kewarganegaraan\s*[:\-]?\s*(WN[IA]|\w+)",
    "berlaku_hingga": r"[Bb]er[il]aku.*?[Hh]ingga\s*[:\-]?\s*(.+?)(?:\n|$)",
}


def _ocr_to_text(ocr_result: list) -> str:
    """Flatten PaddleOCR result into text, grouping detections on the same visual row.

    PaddleOCR returns separate text boxes for labels, colons, and values even when
    they sit on the same line visually.  Grouping by Y-coordinate (within 15 px)
    reconstructs "Nama : KURNIA ANDRIANI" as a single line so the downstream
    regex patterns can match label+value together.
    """
    if not ocr_result or not ocr_result[0]:
        return ""

    detections = []
    for line in ocr_result[0]:
        box = line[0]
        text = line[1][0]
        y_center = (box[0][1] + box[2][1]) / 2
        x_center = (box[0][0] + box[2][0]) / 2
        detections.append((x_center, y_center, text))

    detections.sort(key=lambda d: d[1])

    rows: list[list[tuple]] = [[detections[0]]]
    for det in detections[1:]:
        if abs(det[1] - rows[-1][-1][1]) < 15:
            rows[-1].append(det)
        else:
            rows.append([det])

    lines = []
    for row in rows:
        row.sort(key=lambda d: d[0])
        lines.append(" ".join(d[2] for d in row))

    return "\n".join(lines)


def _parse_ktp_fields(raw_text: str) -> dict:
    """Extract structured KTP fields from OCR text using regex."""
    fields = {}
    for field_name, pattern in KTP_FIELDS.items():
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            # Clean up NIK — remove spaces
            if field_name == "nik":
                value = re.sub(r"\s+", "", value)
            fields[field_name] = value
        else:
            fields[field_name] = None

    return fields


def _compute_face_match(face_path: str, ktp_path: str) -> dict:
    """Compare the applicant's face photo vs the face on the KTP card.

    Returns match result with distance and verified flag.
    """
    try:
        result = DeepFace.verify(
            img1_path=face_path,
            img2_path=ktp_path,
            model_name="VGG-Face",
            enforce_detection=False,
        )
        return {
            "verified": result["verified"],
            "distance": round(result["distance"], 4),
            "threshold": result["threshold"],
            "model": "VGG-Face",
        }
    except Exception as e:
        logger.warning("Face match failed: %s", e)
        return {
            "verified": False,
            "distance": None,
            "threshold": None,
            "model": "VGG-Face",
            "error": str(e),
        }


def process_ekyc(ktp_path: str | Path, face_path: str | Path) -> dict:
    """Run the full eKYC pipeline on a KTP card image and face photo.

    Args:
        ktp_path: Path to the KTP card image.
        face_path: Path to the applicant's face photo.

    Returns:
        {
            "ktp_fields": { "nik": "...", "nama": "...", ... },
            "ocr_raw_text": "...",
            "face_match": { "verified": bool, "distance": float, ... },
        }
    """
    ktp_path = str(ktp_path)
    face_path = str(face_path)
    logger.info("Processing eKYC: ktp=%s, face=%s", ktp_path, face_path)

    # 1. Run OCR on the KTP card
    img = cv2.imread(ktp_path)
    ocr_result = _safe_ocr(img, cls=True)
    raw_text = _ocr_to_text(ocr_result)
    ktp_fields = _parse_ktp_fields(raw_text)

    logger.info("KTP OCR extracted %d fields", sum(1 for v in ktp_fields.values() if v))

    # 2. Face matching — compare applicant face with the face on KTP
    face_match = _compute_face_match(face_path, ktp_path)
    logger.info("Face match: verified=%s, distance=%s", face_match["verified"], face_match.get("distance"))

    return {
        "ktp_fields": ktp_fields,
        "ocr_raw_text": raw_text,
        "face_match": face_match,
    }
