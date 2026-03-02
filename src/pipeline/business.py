import logging
import re
from pathlib import Path

from PIL import Image
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

_ocr_instance = None
_deplot_model = None
_deplot_processor = None


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


_DEPLOT_ONNX_DIR = Path(__file__).resolve().parent.parent.parent / ".cache" / "deplot-onnx"


def _get_deplot():
    """Lazy-load Google DePlot model via ONNX Runtime for fast CPU inference.

    On first call, exports the model to ONNX and caches it locally (~70s).
    Subsequent loads are instant (~3s).
    """
    global _deplot_model, _deplot_processor
    if _deplot_model is None:
        from transformers import Pix2StructProcessor
        from optimum.onnxruntime import ORTModelForVision2Seq

        _deplot_processor = Pix2StructProcessor.from_pretrained("google/deplot")

        # Suppress "Could not find decoder_model_merged.onnx" warning from optimum.
        # The separate encoder/decoder files work fine — the merged file is optional.
        _optimum_logger = logging.getLogger("optimum")
        _prev_level = _optimum_logger.level
        _optimum_logger.setLevel(logging.ERROR)
        try:
            if _DEPLOT_ONNX_DIR.exists():
                logger.info("Loading cached DePlot ONNX model...")
                _deplot_model = ORTModelForVision2Seq.from_pretrained(str(_DEPLOT_ONNX_DIR))
            else:
                logger.info("Exporting DePlot to ONNX (one-time, may take ~60s)...")
                _deplot_model = ORTModelForVision2Seq.from_pretrained(
                    "google/deplot", export=True,
                )
                _deplot_model.save_pretrained(str(_DEPLOT_ONNX_DIR))
                logger.info("DePlot ONNX model cached to %s", _DEPLOT_ONNX_DIR)
        finally:
            _optimum_logger.setLevel(_prev_level)

        logger.info("DePlot model loaded (ONNX Runtime).")
    return _deplot_model, _deplot_processor


def _ocr_to_grouped_lines(ocr_result: list) -> list[str]:
    """Group OCR detections by Y-coordinate to reconstruct visual lines.

    PaddleOCR returns separate boxes for labels and values even when they sit
    on the same row visually. Grouping by Y (within 15 px) reconstructs
    "Pendapatan  Rp 392.367.605" as a single line so regex can match.
    """
    if not ocr_result or not ocr_result[0]:
        return []

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
        if abs(det[1] - rows[-1][-1][1]) < 28:
            rows[-1].append(det)
        else:
            rows.append([det])

    lines = []
    for row in rows:
        row.sort(key=lambda d: d[0])
        lines.append(" ".join(d[2] for d in row))

    return lines


def _parse_rupiah(text: str) -> float | None:
    """Parse 'Rp 948.954.476' or 'Rp -265.256.612' into a float."""
    if not text:
        return None
    cleaned = re.sub(r"^Rp\.?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = cleaned.replace("\u2013", "-").replace("\u2014", "-")  # en-dash / em-dash
    cleaned = cleaned.replace(".", "").replace(",", "")
    cleaned = re.sub(r"[^\d\-]", "", cleaned)
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _parse_percentage(text: str) -> float | None:
    """Parse '18.9%' or '-19.5%' into a float."""
    match = re.search(r"(-?[\d,\.]+)\s*%", text)
    if match:
        try:
            return float(match.group(1).replace(",", "."))
        except ValueError:
            return None
    return None


def _extract_header(lines: list[str]) -> dict:
    """Extract business name, type, and owner from header lines."""
    header = {"business_name": None, "business_type": None, "owner": None}
    header_text = " ".join(lines[:8])

    for line in lines[:5]:
        line_upper = line.strip().upper()
        if line_upper and "LAPORAN" not in line_upper and "USAHA" not in line_upper:
            if not header["business_name"] and len(line.strip()) > 3:
                header["business_name"] = line.strip()
                break

    type_match = re.search(r"Jenis\s*Usaha\s*[:\-]?\s*([A-Za-z\s]+?)(?:\s{2,}|\n|Pemilik|$)", header_text, re.IGNORECASE)
    if type_match:
        header["business_type"] = type_match.group(1).strip()

    owner_match = re.search(r"Pemilik\s*[:\-]?\s*([A-Z][A-Za-z\s\.,]+?)(?:\s{2,}|\n|Pendapatan|$)", header_text)
    if owner_match:
        header["owner"] = owner_match.group(1).strip()

    return header


def _extract_pnl(lines: list[str]) -> dict:
    """Extract P&L values from OCR text lines."""
    pnl = {
        "pendapatan": None,
        "hpp": None,
        "laba_kotor": None,
        "beban_operasional": None,
        "laba_bersih": None,
        "profit_margin": None,
    }

    full_text = "\n".join(lines)

    patterns = {
        "pendapatan": r"Pendapatan\s+(Rp[\s\d\.\-]+)",
        "hpp": r"(?:Harga\s*Pokok|HPP)\s*.*?\s+(Rp[\s\d\.\-]+)",
        "laba_kotor": r"Laba\s*Kotor\s+(Rp[\s\d\.\-]+)",
        "beban_operasional": r"Beban\s*Operasional\s+(Rp[\s\d\.\-]+)",
        "laba_bersih": r"Laba\s*Bersih\s+(Rp[\s\d\.\-]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            pnl[key] = _parse_rupiah(match.group(1))

    margin_match = re.search(r"Margin.*?Laba.*?Bersih\s*[:\-]?\s*(-?[\d,\.]+\s*%)", full_text, re.IGNORECASE)
    if margin_match:
        pnl["profit_margin"] = _parse_percentage(margin_match.group(1))

    return pnl


def _crop_chart_region(image_path: str, ocr_result: list) -> Image.Image:
    """Crop the chart region from the business summary image.

    Detects the chart boundaries using OCR text positions:
    - Top boundary: below the "Pemilik" / "Jenis Usaha" header lines
    - Bottom boundary: above the "RINGKASAN LABA RUGI" P&L section header
    """
    img = Image.open(image_path)
    w, h = img.size

    chart_top = int(h * 0.12)
    chart_bottom = int(h * 0.55)

    if ocr_result and ocr_result[0]:
        for line in ocr_result[0]:
            text = line[1][0].strip().upper()
            box = line[0]
            y_top = min(p[1] for p in box)
            y_bottom = max(p[1] for p in box)

            if "RINGKASAN" in text or "LABA RUGI" in text:
                chart_bottom = int(y_top) - 30
            if "PEMILIK" in text:
                chart_top = int(y_bottom) + 20

    chart_top = max(0, chart_top)
    chart_bottom = min(h, chart_bottom)
    if chart_bottom <= chart_top:
        chart_bottom = int(h * 0.55)

    return img.crop((0, chart_top, w, chart_bottom))


def _extract_revenue_with_deplot(chart_image: Image.Image,
                                  ocr_months: list[str] | None = None) -> list[dict]:
    """Use Google DePlot to extract monthly revenue data points from chart image.

    DePlot converts chart images into linearized data tables. Since DePlot may
    misread month labels (e.g., "Jan" → "2001"), we use OCR-detected month names
    and pair them with DePlot's extracted values positionally.

    Returns list of {month, amount} dicts.
    """
    try:
        model, processor = _get_deplot()
    except Exception as e:
        logger.warning("Failed to load DePlot model: %s", e)
        return []

    try:
        # Resize chart to reduce Pix2Struct encoder patches (faster on CPU)
        max_width = 800
        if chart_image.width > max_width:
            ratio = max_width / chart_image.width
            chart_image = chart_image.resize(
                (max_width, int(chart_image.height * ratio)), Image.LANCZOS,
            )

        inputs = processor(
            images=chart_image,
            text="Generate underlying data table of the chart",
            return_tensors="pt",
        )
        predictions = model.generate(**inputs, max_new_tokens=160)
        output = processor.decode(predictions[0], skip_special_tokens=True)
        logger.info("DePlot raw output: %s", output)
    except Exception as e:
        logger.warning("DePlot inference failed: %s", e)
        return []

    # DePlot output: "TITLE | ... <0x0A> Header | Header <0x0A> Label | Value ..."
    lines = output.replace("<0x0A>", "\n").strip().split("\n")

    # Extract all numeric values from DePlot output (skip title and header rows)
    values = []
    for line in lines:
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        label = parts[0].lower()
        if label in ("title", ""):
            continue
        # Skip header rows that describe column names
        if any(kw in label for kw in ("year", "month", "bulan", "juta", "rp")):
            continue
        try:
            value_str = parts[1].replace(",", "").replace(" ", "")
            value = float(value_str)
            values.append(value)
        except (ValueError, IndexError):
            continue

    if not values:
        return []

    # Convert values: chart Y-axis is "Juta Rp" (millions)
    amounts = []
    for v in values:
        if v < 10_000:
            amounts.append(int(v * 1_000_000))
        else:
            amounts.append(int(v))

    # Pair DePlot values with OCR-detected month labels positionally
    if ocr_months and len(ocr_months) == len(amounts):
        return [{"month": m, "amount": a} for m, a in zip(ocr_months, amounts)]

    # If counts don't match, use standard month labels
    month_labels = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                    "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    return [{"month": month_labels[i % 12], "amount": a}
            for i, a in enumerate(amounts)]


def _extract_revenue_from_ocr(ocr_result: list) -> list[dict]:
    """Fallback: extract month labels from chart via OCR (no amounts)."""
    if not ocr_result or not ocr_result[0]:
        return []

    months = []
    month_names = {"jan", "feb", "mar", "apr", "mei", "jun",
                   "jul", "agu", "sep", "okt", "nov", "des"}

    for line in ocr_result[0]:
        text = line[1][0].strip().lower()
        box = line[0]
        x_center = (box[0][0] + box[2][0]) / 2

        if text in month_names:
            months.append({"name": text.capitalize(), "x": x_center})

    months.sort(key=lambda m: m["x"])
    return [{"month": m["name"]} for m in months]


def _compute_health_metrics(pnl: dict, revenue_months: list[dict]) -> dict:
    """Compute business health indicators from extracted data."""
    metrics = {
        "revenue_months_detected": len(revenue_months),
        "has_profit": None,
        "profit_margin_pct": pnl.get("profit_margin"),
    }

    if pnl.get("laba_bersih") is not None:
        metrics["has_profit"] = pnl["laba_bersih"] > 0

    if pnl.get("pendapatan") and pnl.get("laba_bersih") and not pnl.get("profit_margin"):
        margin = (pnl["laba_bersih"] / pnl["pendapatan"]) * 100
        metrics["profit_margin_pct"] = round(margin, 1)

    return metrics


def process_business_summary(business_summary_path: str | Path) -> dict:
    """Run the full business summary OCR pipeline.

    Uses DePlot for chart data extraction and PaddleOCR for P&L text.

    Returns:
        {
            "header": { "business_name": ..., "business_type": ..., "owner": ... },
            "pnl": { "pendapatan": ..., "laba_bersih": ..., ... },
            "revenue_months": [ { "month": "Jan", "amount": 32000000 }, ... ],
            "health_metrics": { "has_profit": bool, "profit_margin_pct": float, ... },
        }
    """
    business_summary_path = str(business_summary_path)
    logger.info("Processing business summary: %s", business_summary_path)

    ocr_result = _safe_ocr(business_summary_path, cls=True)

    lines = _ocr_to_grouped_lines(ocr_result)

    logger.info("OCR detected %d grouped lines", len(lines))

    header = _extract_header(lines)
    pnl = _extract_pnl(lines)

    # Extract chart data: use OCR for month labels + DePlot for values
    ocr_month_data = _extract_revenue_from_ocr(ocr_result)
    ocr_month_names = [m["month"] for m in ocr_month_data]

    chart_image = _crop_chart_region(business_summary_path, ocr_result)
    revenue_months = _extract_revenue_with_deplot(chart_image, ocr_months=ocr_month_names)

    if not revenue_months:
        logger.info("DePlot extraction returned no results, falling back to OCR month detection")
        revenue_months = ocr_month_data

    health_metrics = _compute_health_metrics(pnl, revenue_months)

    logger.info("P&L: pendapatan=%s, laba_bersih=%s, margin=%s%%",
                pnl.get("pendapatan"), pnl.get("laba_bersih"),
                health_metrics.get("profit_margin_pct"))

    return {
        "header": header,
        "pnl": pnl,
        "revenue_months": revenue_months,
        "health_metrics": health_metrics,
    }
