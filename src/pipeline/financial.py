import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

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


def _parse_amount(text: str) -> float | None:
    """Parse Indonesian formatted amount (e.g., '1.500.000' or '1,500,000') to float."""
    if not text or not text.strip():
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^Rp\.?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace(".", "").replace(",", "")
    cleaned = re.sub(r"[^\d\-]", "", cleaned)
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _parse_date(text: str) -> str | None:
    """Parse date text, return as-is if it looks like a date."""
    if not text:
        return None
    match = re.search(r"\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?", text)
    return match.group(0) if match else None


def _estimate_line_spacing(detections: list[tuple]) -> float:
    """Estimate the median vertical spacing between adjacent text lines."""
    if len(detections) < 3:
        return 25.0

    y_values = sorted(set(d[1] for d in detections))
    if len(y_values) < 3:
        return 25.0

    spacings = [y_values[i + 1] - y_values[i] for i in range(len(y_values) - 1)]
    spacings = [s for s in spacings if s > 5]
    if not spacings:
        return 25.0

    spacings.sort()
    return spacings[len(spacings) // 2]


def _group_ocr_into_rows(ocr_result: list) -> list[list[tuple]]:
    """Group OCR detections into table rows based on Y-coordinate proximity.

    Uses adaptive threshold based on detected line spacing to avoid merging
    adjacent rows in large documents.

    Returns list of rows, each row is a list of (x_center, text) tuples sorted by X.
    """
    if not ocr_result or not ocr_result[0]:
        return []

    detections = []
    for line in ocr_result[0]:
        box = line[0]
        text = line[1][0]
        y_center = (box[0][1] + box[2][1]) / 2
        x_center = (box[0][0] + box[2][0]) / 2
        box_height = abs(box[2][1] - box[0][1])
        detections.append((x_center, y_center, text, box_height))

    detections.sort(key=lambda d: d[1])

    # Use half the median line spacing as the grouping threshold,
    # clamped to a reasonable range
    line_spacing = _estimate_line_spacing(detections)
    threshold = max(12, min(line_spacing * 0.55, 40))
    logger.debug("Row grouping threshold: %.1f (line spacing: %.1f)", threshold, line_spacing)

    rows = []
    current_row = [detections[0]]
    for det in detections[1:]:
        row_y_avg = sum(d[1] for d in current_row) / len(current_row)
        if abs(det[1] - row_y_avg) < threshold:
            current_row.append(det)
        else:
            rows.append(sorted(current_row, key=lambda d: d[0]))
            current_row = [det]
    rows.append(sorted(current_row, key=lambda d: d[0]))

    return [[(d[0], d[2]) for d in row] for row in rows]


def _extract_header_info(rows: list[list[tuple]]) -> dict:
    """Extract bank name, account holder, account number, period, address from header rows.

    Parses row by row (instead of joining all text) to prevent regex from
    bleeding across adjacent fields.
    """
    header_info = {
        "bank_name": None, "account_holder": None, "account_number": None,
        "period": None, "address": None,
    }

    # OCR often renders the ":" separator with artifacts like "_:", ".:", etc.
    # We strip leading punctuation/separators from captured values.
    _SEP = r"[:\-;_\.]*"

    def _clean_value(raw: str) -> str:
        return re.sub(r"^[:\-;_\.\s]+", "", raw).strip()

    for row in rows[:15]:
        row_text = " ".join(cell[1] for cell in row)

        # Bank name: "BANK CIMN NIARA"
        # Lookahead stops capture before "REKENING" or "STATEMENT" when rows merge.
        if header_info["bank_name"] is None:
            bank_match = re.search(
                r"BANK\s+([A-Z][A-Z\s]*?)(?=\s+REKENING|\s+BANK\s+STATEMENT|\s*$)",
                row_text, re.IGNORECASE,
            )
            if bank_match:
                name = bank_match.group(1).strip()
                if name:
                    header_info["bank_name"] = ("BANK " + name).upper()

        # Account holder: "Nama Nasabah : NALAR TAMBA"
        if header_info["account_holder"] is None:
            name_match = re.search(
                rf"(?:Nama\s*(?:Nasabah)?|Nasabah)\s*{_SEP}\s*(.+)",
                row_text, re.IGNORECASE,
            )
            if name_match:
                name = _clean_value(name_match.group(1))
                if len(name) >= 2:
                    header_info["account_holder"] = name.upper()

        # Account number: "No. Rekening : 7010379415"
        if header_info["account_number"] is None:
            acc_match = re.search(
                rf"(?:N[o0]\.?\s*Rekening|Rek(?:ening)?\.?)\s*{_SEP}\s*(\d[\d\s\-]{{5,}})",
                row_text, re.IGNORECASE,
            )
            if acc_match:
                header_info["account_number"] = re.sub(r"[\s\-]", "", acc_match.group(1))

        # Period: "Periode : 3 Bulan Terakhir"
        if header_info["period"] is None:
            period_match = re.search(rf"Periode\s*{_SEP}\s*(.+)", row_text, re.IGNORECASE)
            if period_match:
                header_info["period"] = _clean_value(period_match.group(1))

        # Address: "Alamat : JALAN CIWASTRA NO. 6"
        if header_info["address"] is None:
            addr_match = re.search(rf"Alamat\s*{_SEP}\s*(.+)", row_text, re.IGNORECASE)
            if addr_match:
                header_info["address"] = _clean_value(addr_match.group(1))

    return header_info


def _find_column_positions(rows: list[list[tuple]]) -> dict | None:
    """Find the X-positions of the 5 table columns from the header row(s).

    Looks for column headers like Tanggal, Keterangan, Debit, Kredit, Saldo.
    Headers may be split across two adjacent rows (e.g., numeric columns on one
    row and text columns on the next), so we also check merged pairs.
    """
    header_keywords = {
        "tanggal": "tanggal",
        "keterangan": "keterangan",
        "debit": "debit",
        "kredit": "kredit",
        "credit": "kredit",
        "saldo": "saldo",
        "balance": "saldo",
    }

    def _scan_row(row):
        positions = {}
        for x_center, text in row:
            text_lower = text.lower().split("(")[0].strip()
            for keyword, col_name in header_keywords.items():
                if keyword in text_lower:
                    positions[col_name] = x_center
                    break
        return positions

    for i, row in enumerate(rows):
        positions = _scan_row(row)

        if len(positions) >= 3 and "tanggal" in positions:
            # Before returning, check if the next row has additional columns
            # (e.g. "Saldo (Rp)" split onto a separate row)
            if i + 1 < len(rows):
                extra = _scan_row(rows[i + 1])
                for col, x in extra.items():
                    if col not in positions:
                        positions[col] = x

            logger.info("Found column headers at row %d: %s", i, positions)
            header_idx = i
            # If extra columns came from the next row, advance header_row_idx
            if i + 1 < len(rows) and _scan_row(rows[i + 1]):
                header_idx = i + 1
            return {"positions": positions, "header_row_idx": header_idx}

        # Check if headers span two adjacent rows
        if i + 1 < len(rows):
            merged = _scan_row(row)
            merged.update(_scan_row(rows[i + 1]))
            if len(merged) >= 4 and "tanggal" in merged:
                header_idx = i + 1
                # Also check the row AFTER the pair for stragglers (e.g. "Saldo (Rp)")
                if i + 2 < len(rows):
                    extra = _scan_row(rows[i + 2])
                    for col, x in extra.items():
                        if col not in merged:
                            merged[col] = x
                    if extra:
                        header_idx = i + 2
                logger.info("Found column headers across rows %d-%d: %s", i, header_idx, merged)
                return {"positions": merged, "header_row_idx": header_idx}

    return None


def _build_column_ranges(col_positions: dict[str, float]) -> list[tuple[str, float, float]]:
    """Build (col_name, x_min, x_max) ranges from column center positions.

    Each column owns the space from the midpoint to the previous column
    to the midpoint to the next column. This handles long description text
    whose bounding-box center is shifted right of the "Keterangan" header.
    """
    sorted_cols = sorted(col_positions.items(), key=lambda kv: kv[1])
    ranges = []
    for idx, (name, x) in enumerate(sorted_cols):
        if idx == 0:
            x_min = 0
        else:
            x_min = (sorted_cols[idx - 1][1] + x) / 2
        if idx == len(sorted_cols) - 1:
            x_max = float("inf")
        else:
            x_max = (x + sorted_cols[idx + 1][1]) / 2
        ranges.append((name, x_min, x_max))
    return ranges


def _assign_cell_to_column(x_center: float, col_positions: dict[str, float],
                           col_ranges: list[tuple[str, float, float]] = None) -> str | None:
    """Assign a cell to its column based on X-position ranges.

    Uses pre-computed ranges (midpoints between adjacent columns) so that
    wide text boxes (e.g. long descriptions) are correctly assigned even when
    their center is far from the column header center.
    """
    if col_ranges is None:
        col_ranges = _build_column_ranges(col_positions)

    for col_name, x_min, x_max in col_ranges:
        if x_min <= x_center < x_max:
            return col_name
    return None


def _extract_transactions(rows: list[list[tuple]], col_info: dict | None) -> list[dict]:
    """Parse transaction rows using X-position-based column assignment.

    Handles the common OCR pattern where a single transaction's data is split
    across multiple rows (e.g., amounts on one row, date+description on the next).
    Merges consecutive rows into complete transactions.
    """
    transactions = []

    if col_info is None:
        logger.warning("Could not find column headers, falling back to positional parsing")
        return _extract_transactions_positional(rows)

    col_positions = col_info["positions"]
    table_start = col_info["header_row_idx"] + 1
    col_ranges = _build_column_ranges(col_positions)

    def _parse_row(row):
        """Parse a single OCR row into a partial transaction dict."""
        txn = {"date": None, "description": "", "debit": None, "credit": None, "balance": None}
        for x_center, text in row:
            col = _assign_cell_to_column(x_center, col_positions, col_ranges=col_ranges)
            if col is None:
                continue
            if col == "tanggal":
                parsed_date = _parse_date(text)
                if parsed_date:
                    txn["date"] = parsed_date
                elif not txn["description"]:
                    txn["description"] = text
            elif col == "keterangan":
                if txn["description"]:
                    txn["description"] += " " + text
                else:
                    txn["description"] = text
            elif col == "debit":
                amt = _parse_amount(text)
                if amt is not None and amt > 0:
                    txn["debit"] = amt
            elif col == "kredit":
                amt = _parse_amount(text)
                if amt is not None and amt > 0:
                    txn["credit"] = amt
            elif col == "saldo":
                amt = _parse_amount(text)
                if amt is not None:
                    txn["balance"] = amt
        return txn

    def _merge(base, extra):
        """Merge fields from extra into base (fill None slots only)."""
        for key in ("date", "description", "debit", "credit", "balance"):
            if base[key] is None and extra[key] is not None:
                base[key] = extra[key]
            elif key == "description" and extra[key] and not base[key]:
                base[key] = extra[key]

    # In degraded bank statements, OCR often detects amounts on a row
    # ABOVE the date+description row. So we accumulate amount-only rows
    # and merge them INTO the next date row when it appears.
    pending_amounts = None
    current_txn = None

    for row in rows[table_start:]:
        if not row:
            continue
        parsed = _parse_row(row)
        has_date = parsed["date"] is not None
        has_amount = parsed["debit"] is not None or parsed["credit"] is not None or parsed["balance"] is not None

        if has_date:
            # Flush any complete previous transaction
            if current_txn and current_txn["date"]:
                if current_txn["debit"] or current_txn["credit"] or current_txn["balance"]:
                    transactions.append(current_txn)

            # Start new transaction from date row, merging any preceding amounts
            current_txn = parsed
            if pending_amounts:
                _merge(current_txn, pending_amounts)
                pending_amounts = None
        elif has_amount and current_txn and current_txn["date"]:
            # Amount row after a date row — merge if it fills empty slots without conflict
            has_conflict = any(
                current_txn[k] is not None and parsed[k] is not None
                for k in ("debit", "credit", "balance")
            )
            if not has_conflict:
                _merge(current_txn, parsed)
            else:
                # Conflicting amounts belong to the next transaction
                pending_amounts = parsed
        elif has_amount:
            # Amount row before any date — accumulate for the next date row
            if pending_amounts:
                _merge(pending_amounts, parsed)
            else:
                pending_amounts = parsed

    # Flush last transaction
    if current_txn and current_txn["date"]:
        if pending_amounts:
            _merge(current_txn, pending_amounts)
        if current_txn["debit"] or current_txn["credit"] or current_txn["balance"]:
            transactions.append(current_txn)

    return transactions


def _extract_transactions_positional(rows: list[list[tuple]]) -> list[dict]:
    """Fallback: parse transactions by position count (original logic)."""
    transactions = []

    table_start = 0
    for i, row in enumerate(rows):
        row_text = " ".join(cell[1] for cell in row).lower()
        if "tanggal" in row_text or "keterangan" in row_text:
            table_start = i + 1
            break

    for row in rows[table_start:]:
        if len(row) < 2:
            continue

        texts = [cell[1] for cell in row]
        date = _parse_date(texts[0])
        if not date:
            continue

        txn = {"date": date, "description": "", "debit": None, "credit": None, "balance": None}

        if len(texts) > 1:
            txn["description"] = texts[1]

        amounts = []
        for t in texts[2:]:
            amt = _parse_amount(t)
            if amt is not None:
                amounts.append(amt)

        if len(amounts) >= 3:
            txn["debit"] = amounts[0] if amounts[0] > 0 else None
            txn["credit"] = amounts[1] if amounts[1] > 0 else None
            txn["balance"] = amounts[2]
        elif len(amounts) == 2:
            txn["debit"] = amounts[0] if amounts[0] > 0 else None
            txn["balance"] = amounts[1]
        elif len(amounts) == 1:
            txn["balance"] = amounts[0]

        transactions.append(txn)

    return transactions


def _compute_metrics(transactions: list[dict]) -> dict:
    """Compute financial metrics from parsed transactions."""
    if not transactions:
        return {
            "total_credit": 0, "total_debit": 0,
            "avg_monthly_income": 0, "net_cash_flow": 0,
            "transaction_count": 0,
        }

    df = pd.DataFrame(transactions)

    total_credit = df["credit"].dropna().sum()
    total_debit = df["debit"].dropna().sum()
    net_cash_flow = total_credit - total_debit

    # Estimate months from date range
    num_months = 1
    dates_parsed = []
    for txn in transactions:
        if txn.get("date"):
            match = re.search(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", txn["date"])
            if match:
                month = int(match.group(2))
                year = int(match.group(3))
                dates_parsed.append((year, month))
    if dates_parsed:
        unique_months = set(dates_parsed)
        num_months = max(1, len(unique_months))

    avg_monthly_income = total_credit / num_months if num_months > 0 else 0

    return {
        "total_credit": round(total_credit),
        "total_debit": round(total_debit),
        "avg_monthly_income": round(avg_monthly_income),
        "net_cash_flow": round(net_cash_flow),
        "transaction_count": len(transactions),
    }


def process_bank_statement(bank_statement_path: str | Path) -> dict:
    """Run the full bank statement OCR pipeline.

    Returns:
        {
            "header": { "bank_name": ..., "account_holder": ..., "account_number": ... },
            "transactions": [ { "date": ..., "description": ..., ... }, ... ],
            "metrics": { "total_credit": ..., "avg_monthly_income": ..., ... },
            "ocr_line_count": int,
        }
    """
    bank_statement_path = str(bank_statement_path)
    logger.info("Processing bank statement: %s", bank_statement_path)

    ocr_result = _safe_ocr(bank_statement_path, cls=True)

    rows = _group_ocr_into_rows(ocr_result)
    logger.info("OCR detected %d text groups across %d rows",
                sum(len(r) for r in rows), len(rows))

    header = _extract_header_info(rows)
    col_info = _find_column_positions(rows)
    transactions = _extract_transactions(rows, col_info)
    metrics = _compute_metrics(transactions)

    logger.info("Parsed %d transactions, avg monthly income: %s",
                len(transactions), f"Rp {metrics['avg_monthly_income']:,.0f}")

    return {
        "header": header,
        "transactions": transactions,
        "metrics": metrics,
        "ocr_line_count": sum(len(r) for r in rows),
    }
