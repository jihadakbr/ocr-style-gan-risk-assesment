import json
import logging
import os
import sys
import threading
import time
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Suppress noisy logs before imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline.ekyc import process_ekyc
from src.pipeline.financial import process_bank_statement
from src.pipeline.business import process_business_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
for name in ["ppocr", "paddle", "matplotlib", "PIL", "tensorflow", "deepface"]:
    logging.getLogger(name).setLevel(logging.ERROR)

# Page config
st.set_page_config(page_title="SME Lending - Risk Assessment", layout="wide")
st.title("SME Lending — Risk Assessment Dashboard")


def _fmt_rupiah(value) -> str:
    if value is None:
        return "—"
    return f"Rp {value:,.0f}".replace(",", ".")


def _parse_rupiah_input(text: str) -> float | None:
    """Parse user-entered rupiah value back to float."""
    if not text or text.strip() in ("—", ""):
        return None
    cleaned = text.strip()
    cleaned = cleaned.replace("Rp", "").replace("rp", "").strip()
    cleaned = cleaned.replace(".", "").replace(",", "")
    cleaned = "".join(c for c in cleaned if c.isdigit() or c == "-")
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _compute_risk_score(ekyc: dict, financial: dict, business: dict) -> dict:
    """Compute a weighted credit score from all 3 stages.

    Credit Score: 0–100 where higher = better applicant.
    Risk is the inverse: lower credit score = higher risk.
    """
    score = 0
    breakdown = {}

    # 1. Face match (30 points)
    face = ekyc.get("face_match", {})
    if face.get("verified"):
        face_score = 30
    elif face.get("distance") is not None:
        face_score = max(0, int(30 * (1 - face["distance"])))
    else:
        face_score = 0
    score += face_score
    breakdown["identity_verification"] = face_score

    # 2. Financial health (40 points)
    metrics = financial.get("metrics", {})
    fin_score = 0
    if metrics.get("avg_monthly_income", 0) > 0:
        fin_score += 15
    if metrics.get("net_cash_flow", 0) > 0:
        fin_score += 15
    if metrics.get("transaction_count", 0) >= 10:
        fin_score += 10
    score += fin_score
    breakdown["financial_health"] = fin_score

    # 3. Business viability (30 points)
    biz_score = 0
    health = business.get("health_metrics", {})
    if health.get("has_profit"):
        biz_score += 15
    margin = health.get("profit_margin_pct")
    if margin is not None:
        if margin > 20:
            biz_score += 15
        elif margin > 10:
            biz_score += 10
        elif margin > 0:
            biz_score += 5
    score += biz_score
    breakdown["business_viability"] = biz_score

    # Risk score is inverse of credit score: 0 = no risk, 100 = maximum risk
    risk_score = 100 - score

    if risk_score <= 25:
        level = "LOW RISK"
    elif risk_score <= 50:
        level = "MEDIUM RISK"
    else:
        level = "HIGH RISK"

    return {"score": risk_score, "credit_score": score, "level": level, "breakdown": breakdown}


def _get_applicant_dirs(data_dir: str) -> list[Path]:
    """List all applicant bundle directories."""
    root = Path(data_dir)
    if not root.exists():
        return []
    return sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("applicant_")])


_WIB = timezone(timedelta(hours=7))


def _build_save_output(applicant_name, ekyc_res, financial_res, business_res, risk_res, corrected=False):
    """Build the JSON output dict for saving results."""
    return {
        "applicant": applicant_name,
        "corrected": corrected,
        "timestamp": datetime.now(_WIB).isoformat(),
        "risk": risk_res,
        "ekyc": ekyc_res,
        "financial": {
            "header": financial_res.get("header"),
            "metrics": financial_res.get("metrics"),
            "transaction_count": len(financial_res.get("transactions", [])),
        },
        "business": {
            "header": business_res.get("header"),
            "pnl": business_res.get("pnl"),
            "health_metrics": business_res.get("health_metrics"),
        },
    }


# Sidebar
st.sidebar.header("Settings")
data_dir = st.sidebar.text_input("Data directory", value="data/raw")
applicant_dirs = _get_applicant_dirs(data_dir)

if not applicant_dirs:
    st.warning(f"No applicant bundles found in `{data_dir}`. Generate data first by running "
               f"the Kaggle notebook (`notebooks/dataset_generator.ipynb`), then extract the "
               f"output zip into `{data_dir}/`.")
    st.stop()

st.sidebar.markdown("---")

st.sidebar.subheader("Single Applicant")
selected = st.sidebar.selectbox(
    "Select applicant",
    applicant_dirs,
    format_func=lambda d: d.name,
)
run_button = st.sidebar.button("Run OCR Pipeline", type="primary", width="stretch")

st.sidebar.markdown("---")

st.sidebar.subheader("Batch Processing")
select_all = st.sidebar.checkbox("Select all applicants", value=False)
batch_selection = st.sidebar.multiselect(
    "Select applicants to batch process",
    applicant_dirs,
    default=applicant_dirs if select_all else [],
    format_func=lambda d: d.name,
)
batch_button = st.sidebar.button("Process & Save Selected", type="primary", width="stretch", disabled=not batch_selection)

st.markdown(
    """<style>
    span[data-baseweb="tag"] {
        background-color: #0068c9 !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

def _run_with_timer(label, func, *args):
    """Run *func* in a background thread while showing a live stopwatch."""
    placeholder = st.empty()
    result = [None]
    error = [None]

    def _worker():
        try:
            result[0] = func(*args)
        except Exception as exc:
            error[0] = exc

    thread = threading.Thread(target=_worker)
    thread.start()
    start = time.time()
    while thread.is_alive():
        elapsed = int(time.time() - start)
        placeholder.info(f"⏳ {label} ({elapsed}s)")
        time.sleep(0.5)
    thread.join()

    elapsed = int(time.time() - start)
    placeholder.success(f"✅ {label} ({elapsed}s)")

    if error[0]:
        raise error[0]
    return result[0]


if batch_button:
    output_dir = Path("data/applicant_result")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = st.progress(0, text="Starting batch processing...")
    total = len(batch_selection)
    errors = []

    for idx, app_dir in enumerate(batch_selection):
        label = app_dir.name
        progress.progress((idx) / total, text=f"Processing {label} ({idx + 1}/{total})...")
        try:
            ekyc_res = process_ekyc(str(app_dir / "ktp.jpg"), str(app_dir / "face.jpg"))
            financial_res = process_bank_statement(str(app_dir / "bank_statement.jpg"))
            business_res = process_business_summary(str(app_dir / "business_summary.jpg"))
            risk_res = _compute_risk_score(ekyc_res, financial_res, business_res)

            output = _build_save_output(label, ekyc_res, financial_res, business_res, risk_res)
            wib = timezone(timedelta(hours=7))
            ts = datetime.now(wib).strftime("%Y%m%d_%H%M%S")
            (output_dir / f"{label}_{ts}.json").write_text(json.dumps(output, indent=2, default=str))
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    progress.progress(1.0, text="Batch processing complete!")
    st.success(f"Processed and saved {total - len(errors)}/{total} applicants to `{output_dir}/`")
    if errors:
        st.warning("Errors:\n" + "\n".join(errors))

if run_button:
    ekyc_result = _run_with_timer(
        "Running eKYC (KTP OCR + face matching)...",
        process_ekyc, str(selected / "ktp.jpg"), str(selected / "face.jpg"),
    )
    financial_result = _run_with_timer(
        "Running bank statement OCR...",
        process_bank_statement, str(selected / "bank_statement.jpg"),
    )
    business_result = _run_with_timer(
        "Running business summary OCR...",
        process_business_summary, str(selected / "business_summary.jpg"),
    )

    risk = _compute_risk_score(ekyc_result, financial_result, business_result)

    st.session_state["ekyc"] = ekyc_result
    st.session_state["financial"] = financial_result
    st.session_state["business"] = business_result
    st.session_state["risk"] = risk
    st.session_state["applicant"] = selected.name
    st.session_state.pop("corrected", None)

# Clear stale results when the user switches to a different applicant
if st.session_state.get("applicant") != selected.name:
    for key in ("ekyc", "financial", "business", "risk", "applicant", "corrected"):
        st.session_state.pop(key, None)

if "risk" not in st.session_state:
    if not batch_button:
        st.info("Select an applicant and click **Run OCR Pipeline** to begin.")
    st.stop()

ekyc_result = st.session_state["ekyc"]
financial_result = st.session_state["financial"]
business_result = st.session_state["business"]
risk = st.session_state["risk"]

# Risk Score Header
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Risk Score", f"{risk['score']}/100")
col2.metric("Risk Level", risk["level"])
col3.metric("Identity", f"{risk['breakdown']['identity_verification']}/30")
col4.metric("Financial", f"{risk['breakdown']['financial_health']}/40")
col5.metric("Business", f"{risk['breakdown']['business_viability']}/30")

st.caption("LOW RISK: 0–25  \nMEDIUM RISK: 26–50  \nHIGH RISK: 51–100")

if st.session_state.get("corrected"):
    st.success("Showing results after human correction. Risk score has been recalculated.")

st.markdown("---")

# Stage 1: eKYC
st.header("Stage 1 — eKYC (Identity Verification)")
_s1 = risk["breakdown"]["identity_verification"]
st.caption(f"Score: **{_s1}/30** (interval: 0–30)")
col_img, col_data = st.columns([1, 2])

with col_img:
    face_path = selected / "face.jpg"
    ktp_path = selected / "ktp.jpg"
    if face_path.exists():
        st.image(str(face_path), caption="Applicant Face", width="stretch")
    if ktp_path.exists():
        st.image(str(ktp_path), caption="KTP Card", width="stretch")

with col_data:
    st.subheader("Extracted KTP Fields")
    fields = ekyc_result.get("ktp_fields", {})

    field_data = []
    for key, value in fields.items():
        field_data.append({
            "Field": key.replace("_", " ").title(),
            "OCR Value": value or "—",
        })

    st.dataframe(pd.DataFrame(field_data), width="stretch", hide_index=True)

    st.subheader("Face Match")
    face = ekyc_result.get("face_match", {})
    fcol1, fcol2, fcol3 = st.columns(3)
    fcol1.metric("Verified", "Yes" if face.get("verified") else "No")
    fcol2.metric("Distance", f"{face.get('distance', '—')}")
    fcol3.metric("Threshold", f"{face.get('threshold', '—')}")

st.subheader("Scoring Breakdown")
_face = ekyc_result.get("face_match", {})
_face_verified = _face.get("verified", False)
_face_dist = _face.get("distance")
if _face_verified:
    _f_pts, _f_status, _f_detail = 30, "Passed", "Face verified — full points"
elif _face_dist is not None:
    _f_pts = max(0, int(30 * (1 - _face_dist)))
    _f_status, _f_detail = "Partial", f"Not verified, distance {_face_dist:.4f} — 30 x (1 - {_face_dist:.4f}) = {_f_pts}"
else:
    _f_pts, _f_status, _f_detail = 0, "Failed", "Face matching failed"
st.dataframe(pd.DataFrame([
    {"Check": "Face Match", "Condition": "Face photo matches KTP card", "Max": 30, "Awarded": _f_pts, "Status": _f_status, "Detail": _f_detail},
]), width="stretch", hide_index=True)
st.caption(f"Total: **{_f_pts}/30**")

st.markdown("---")

# Stage 2: Financial
st.header("Stage 2 — Financial Verification")
_s2 = risk["breakdown"]["financial_health"]
st.caption(f"Score: **{_s2}/40** (interval: 0–40)")
col_img2, col_data2 = st.columns([1, 2])

with col_img2:
    bank_path = selected / "bank_statement.jpg"
    if bank_path.exists():
        st.image(str(bank_path), caption="Bank Statement", width="stretch")

with col_data2:
    header = financial_result.get("header", {})
    st.subheader(f"{header.get('bank_name', 'Bank')} — {header.get('account_holder', '—')}")
    st.caption(
        f"No. Rekening: {header.get('account_number', '—')}"
        f" · Periode: {header.get('period', '—')}"
        f" · Alamat: {header.get('address', '—')}"
    )

    metrics = financial_result.get("metrics", {})
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Pendapatan Bulanan Rata-rata", _fmt_rupiah(metrics.get("avg_monthly_income")))
    mcol2.metric("Total Kredit (Rp)", _fmt_rupiah(metrics.get("total_credit")))
    mcol3.metric("Total Debit (Rp)", _fmt_rupiah(metrics.get("total_debit")))
    mcol4.metric("Arus Kas Bersih", _fmt_rupiah(metrics.get("net_cash_flow")))

    st.subheader("Transaksi")
    txns = financial_result.get("transactions", [])
    if txns:
        df = pd.DataFrame(txns)
        col_rename = {
            "date": "Tanggal",
            "description": "Keterangan",
            "debit": "Debit (Rp)",
            "credit": "Kredit (Rp)",
            "balance": "Saldo (Rp)",
        }
        df = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})
        for col in ["Debit (Rp)", "Kredit (Rp)", "Saldo (Rp)"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: f"{v:,.0f}".replace(",", ".") if pd.notna(v) else "")
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption(f"Menampilkan {len(txns)} transaksi")
    else:
        st.warning("Tidak ada transaksi yang terdeteksi")

st.subheader("Scoring Breakdown")
_fm = financial_result.get("metrics", {})
_income_ok = _fm.get("avg_monthly_income", 0) > 0
_cash_ok = _fm.get("net_cash_flow", 0) > 0
_active_ok = _fm.get("transaction_count", 0) >= 10
_income_pts = 15 if _income_ok else 0
_cash_pts = 15 if _cash_ok else 0
_active_pts = 10 if _active_ok else 0
st.dataframe(pd.DataFrame([
    {
        "Check": "Has Income",
        "Condition": "Avg monthly income > 0",
        "Max": 15,
        "Awarded": _income_pts,
        "Status": "Passed" if _income_ok else "Failed",
        "Detail": _fmt_rupiah(_fm.get("avg_monthly_income")),
    },
    {
        "Check": "Positive Cash Flow",
        "Condition": "Net cash flow > 0",
        "Max": 15,
        "Awarded": _cash_pts,
        "Status": "Passed" if _cash_ok else "Failed",
        "Detail": _fmt_rupiah(_fm.get("net_cash_flow")),
    },
    {
        "Check": "Active Account",
        "Condition": "Transaction count >= 10",
        "Max": 10,
        "Awarded": _active_pts,
        "Status": "Passed" if _active_ok else "Failed",
        "Detail": f"{_fm.get('transaction_count', 0)} transactions",
    },
]), width="stretch", hide_index=True)
st.caption(f"Total: **{_income_pts + _cash_pts + _active_pts}/40**")

st.markdown("---")

# Stage 3: Business Health
st.header("Stage 3 — Business Health")
_s3 = risk["breakdown"]["business_viability"]
st.caption(f"Score: **{_s3}/30** (interval: 0–30)")

col_img3, col_data3 = st.columns([1, 2])

with col_img3:
    biz_path = selected / "business_summary.jpg"
    if biz_path.exists():
        st.image(str(biz_path), caption="Business Summary", width="stretch")

with col_data3:
    biz_header = business_result.get("header", {})
    st.subheader(biz_header.get("business_name", "—"))
    st.caption(f"Type: {biz_header.get('business_type', '—')} | Owner: {biz_header.get('owner', '—')}")

    # Monthly revenue extracted from chart (via DePlot or OCR)
    revenue_months = business_result.get("revenue_months", [])
    if revenue_months:
        has_amounts = any("amount" in m for m in revenue_months)
        if has_amounts:
            st.subheader("Pendapatan Bulanan (dari Chart)")
            rev_df = pd.DataFrame(revenue_months)
            rev_df.columns = ["Bulan", "Pendapatan (Rp)"]
            rev_df["Pendapatan (Rp)"] = rev_df["Pendapatan (Rp)"].apply(
                lambda v: f"Rp {v:,.0f}".replace(",", "."))
            st.dataframe(rev_df, width="stretch", hide_index=True)
        else:
            st.subheader("Detected Months from Chart")
            st.caption("OCR could only detect month labels, not revenue values.")
            st.write(", ".join(m.get("month", m.get("name", "")) for m in revenue_months))

    pnl = business_result.get("pnl", {})

    st.subheader("Profit & Loss")
    pnl_items = [
        ("Pendapatan (Revenue)", "pendapatan"),
        ("HPP (COGS)", "hpp"),
        ("Laba Kotor (Gross Profit)", "laba_kotor"),
        ("Beban Operasional (OpEx)", "beban_operasional"),
        ("Laba Bersih (Net Profit)", "laba_bersih"),
    ]
    pnl_data = []
    for label, key in pnl_items:
        pnl_data.append({"Item": label, "Value": _fmt_rupiah(pnl.get(key))})
    st.dataframe(pd.DataFrame(pnl_data), width="stretch", hide_index=True)

    health = business_result.get("health_metrics", {})
    hcol1, hcol2 = st.columns(2)
    hcol1.metric("Profitable", "Yes" if health.get("has_profit") else "No")
    hcol2.metric("Profit Margin", f"{health.get('profit_margin_pct', '—')}%")

st.subheader("Scoring Breakdown")
_bh = business_result.get("health_metrics", {})
_profitable = _bh.get("has_profit", False)
_margin = _bh.get("profit_margin_pct")
_profit_pts = 15 if _profitable else 0
if _margin is not None and _margin > 20:
    _margin_pts, _margin_detail = 15, f"{_margin}% (> 20%)"
elif _margin is not None and _margin > 10:
    _margin_pts, _margin_detail = 10, f"{_margin}% (> 10%)"
elif _margin is not None and _margin > 0:
    _margin_pts, _margin_detail = 5, f"{_margin}% (> 0%)"
else:
    _margin_pts = 0
    _margin_detail = f"{_margin}%" if _margin is not None else "N/A"
st.dataframe(pd.DataFrame([
    {
        "Check": "Business Profitable",
        "Condition": "Net profit (Laba Bersih) > 0",
        "Max": 15,
        "Awarded": _profit_pts,
        "Status": "Passed" if _profitable else "Failed",
        "Detail": _fmt_rupiah(business_result.get("pnl", {}).get("laba_bersih")),
    },
    {
        "Check": "Profit Margin",
        "Condition": "> 20% = 15 pts, > 10% = 10 pts, > 0% = 5 pts",
        "Max": 15,
        "Awarded": _margin_pts,
        "Status": "Passed" if _margin_pts > 0 else "Failed",
        "Detail": _margin_detail,
    },
]), width="stretch", hide_index=True)
st.caption(f"Total: **{_profit_pts + _margin_pts}/30**")

st.markdown("---")

# Human-in-the-Loop Correction Section
st.header("Human-in-the-Loop Correction")
st.caption(
    "If OCR missed or garbled text, correct the values below. "
    "Click **Recalculate Risk** to update the risk score with your corrections."
)

with st.expander("Edit KTP Fields", expanded=False):
    edited_ktp = {}
    ktp_fields = ekyc_result.get("ktp_fields", {})
    ktp_cols = st.columns(2)
    for i, (key, value) in enumerate(ktp_fields.items()):
        with ktp_cols[i % 2]:
            edited_ktp[key] = st.text_input(
                key.replace("_", " ").title(),
                value=value or "",
                key=f"ktp_{key}",
            )

with st.expander("Edit Financial Metrics", expanded=False):
    fin_metrics = financial_result.get("metrics", {})
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        edit_avg_income = st.text_input(
            "Pendapatan Bulanan Rata-rata",
            value=str(int(fin_metrics.get("avg_monthly_income", 0))),
            key="fin_avg_income",
        )
        edit_total_credit = st.text_input(
            "Total Kredit (Rp)",
            value=str(int(fin_metrics.get("total_credit", 0))),
            key="fin_total_credit",
        )
    with fcol2:
        edit_total_debit = st.text_input(
            "Total Debit (Rp)",
            value=str(int(fin_metrics.get("total_debit", 0))),
            key="fin_total_debit",
        )
        edit_net_cash = st.text_input(
            "Arus Kas Bersih",
            value=str(int(fin_metrics.get("net_cash_flow", 0))),
            key="fin_net_cash",
        )
    edit_txn_count = st.number_input(
        "Jumlah Transaksi",
        value=fin_metrics.get("transaction_count", 0),
        min_value=0,
        key="fin_txn_count",
    )

with st.expander("Edit Business P&L", expanded=False):
    biz_pnl = business_result.get("pnl", {})
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        edit_pendapatan = st.text_input(
            "Pendapatan (Revenue)",
            value=str(int(biz_pnl.get("pendapatan") or 0)),
            key="biz_pendapatan",
        )
        edit_hpp = st.text_input(
            "HPP (COGS)",
            value=str(int(biz_pnl.get("hpp") or 0)),
            key="biz_hpp",
        )
        edit_laba_kotor = st.text_input(
            "Laba Kotor",
            value=str(int(biz_pnl.get("laba_kotor") or 0)),
            key="biz_laba_kotor",
        )
    with pcol2:
        edit_beban_op = st.text_input(
            "Beban Operasional",
            value=str(int(biz_pnl.get("beban_operasional") or 0)),
            key="biz_beban_op",
        )
        edit_laba_bersih = st.text_input(
            "Laba Bersih",
            value=str(int(biz_pnl.get("laba_bersih") or 0)),
            key="biz_laba_bersih",
        )
        edit_margin = st.text_input(
            "Margin Laba Bersih (%)",
            value=str(business_result.get("health_metrics", {}).get("profit_margin_pct") or ""),
            key="biz_margin",
        )

# Recalculate Risk button
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if st.button("Recalculate Risk", type="primary", width="stretch"):
        # Build corrected ekyc
        corrected_ekyc = {**ekyc_result, "ktp_fields": edited_ktp}

        # Build corrected financial metrics
        corrected_metrics = {
            "avg_monthly_income": _parse_rupiah_input(edit_avg_income) or 0,
            "total_credit": _parse_rupiah_input(edit_total_credit) or 0,
            "total_debit": _parse_rupiah_input(edit_total_debit) or 0,
            "net_cash_flow": _parse_rupiah_input(edit_net_cash) or 0,
            "transaction_count": edit_txn_count,
        }
        corrected_financial = {**financial_result, "metrics": corrected_metrics}

        # Build corrected business
        pendapatan_val = _parse_rupiah_input(edit_pendapatan)
        laba_bersih_val = _parse_rupiah_input(edit_laba_bersih)
        margin_val = None
        if edit_margin and edit_margin.strip():
            try:
                margin_val = float(edit_margin.strip().replace("%", "").replace(",", "."))
            except ValueError:
                pass
        if margin_val is None and pendapatan_val and laba_bersih_val and pendapatan_val > 0:
            margin_val = round((laba_bersih_val / pendapatan_val) * 100, 1)

        corrected_pnl = {
            "pendapatan": pendapatan_val,
            "hpp": _parse_rupiah_input(edit_hpp),
            "laba_kotor": _parse_rupiah_input(edit_laba_kotor),
            "beban_operasional": _parse_rupiah_input(edit_beban_op),
            "laba_bersih": laba_bersih_val,
            "profit_margin": margin_val,
        }
        corrected_health = {
            "revenue_months_detected": business_result.get("health_metrics", {}).get("revenue_months_detected", 0),
            "has_profit": laba_bersih_val > 0 if laba_bersih_val is not None else None,
            "profit_margin_pct": margin_val,
        }
        corrected_business = {
            **business_result,
            "pnl": corrected_pnl,
            "health_metrics": corrected_health,
        }

        new_risk = _compute_risk_score(corrected_ekyc, corrected_financial, corrected_business)

        st.session_state["ekyc"] = corrected_ekyc
        st.session_state["financial"] = corrected_financial
        st.session_state["business"] = corrected_business
        st.session_state["risk"] = new_risk
        st.session_state["corrected"] = True
        st.rerun()

with btn_col2:
    if st.button("Save Results", width="stretch"):
        output = _build_save_output(
            selected.name, ekyc_result, financial_result, business_result, risk,
            corrected=bool(st.session_state.get("corrected")),
        )
        output_dir = Path("data/applicant_result")
        output_dir.mkdir(parents=True, exist_ok=True)
        wib = timezone(timedelta(hours=7))
        ts = datetime.now(wib).strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{selected.name}_{ts}.json"
        output_path.write_text(json.dumps(output, indent=2, default=str))
        st.success(f"Results saved to `{output_path}`")
