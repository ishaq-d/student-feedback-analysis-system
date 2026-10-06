from pathlib import Path
import io
import math
import os
import uuid
from datetime import datetime

import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    jsonify
)

from werkzeug.utils import secure_filename

from services import (
    VALID_CATEGORIES,
    VALID_SENTIMENTS,
    load_metadata,
    validate_csv,
    predict_feedback,
    apply_filters
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "student-feedback-dev-key"
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

DATA_PATH = (
    BASE_DIR
    / "data"
    / "student_feedback_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the currently active analysed dataset.
    """
    if not DATA_PATH.exists():
        return pd.DataFrame(
            columns=[
                "ID",
                "Feedback Text",
                "Category",
                "Sentiment",
                "Date"
            ]
        )

    return pd.read_csv(DATA_PATH)


# ============================================================
# DASHBOARD DATA
# ============================================================

def get_dashboard_data(df=None):
    if df is None:
        df = load_data()

    total = len(df)

    sentiment_counts = (
        df["Sentiment"].value_counts().to_dict()
        if total
        else {}
    )

    percentages = {
        sentiment: round(
            (sentiment_counts.get(sentiment, 0) / total) * 100, 1
        )
        if total
        else 0
        for sentiment in VALID_SENTIMENTS
    }

    category_counts = (
        df.groupby(["Category", "Sentiment"])
        .size()
        .unstack(fill_value=0)
        if total
        else pd.DataFrame()
    )

    category_data = []

    for category in VALID_CATEGORIES:
        if not category_counts.empty and category in category_counts.index:
            row = category_counts.loc[category]
            pos = int(row.get("Positive", 0)) if "Positive" in row else 0
            neu = int(row.get("Neutral", 0)) if "Neutral" in row else 0
            neg = int(row.get("Negative", 0)) if "Negative" in row else 0
        else:
            pos, neu, neg = 0, 0, 0

        category_data.append({
            "category": category,
            "Positive": pos,
            "Neutral": neu,
            "Negative": neg,
        })

    return {
        "total": total,
        "sentiment_counts": sentiment_counts,
        "percentages": percentages,
        "category_data": category_data
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():
    df = load_data()
    metadata = load_metadata()
    data = get_dashboard_data(df)

    recent = (
        df.sort_values(
            "ID",
            ascending=False
        )
        .head(10)
        .to_dict("records")
        if len(df)
        else []
    )

    return render_template(
        "dashboard.html",
        data=data,
        recent=recent,
        metadata=metadata,
        categories=VALID_CATEGORIES,
        active_page="dashboard"
    )


# ============================================================
# UPLOAD AND ANALYSE DATA
# ============================================================

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")

        ok, result = validate_csv(file)

        if not ok:
            flash(
                result,
                "danger"
            )
            return redirect(
                url_for("upload")
            )

        # ----------------------------------------------------
        # Preserve original uploaded file
        # ----------------------------------------------------

        upload_id = uuid.uuid4().hex[:12]

        safe_name = secure_filename(
            file.filename
        )

        upload_path = (
            BASE_DIR
            / "uploads"
            / f"{upload_id}_{safe_name}"
        )

        upload_path.parent.mkdir(parents=True, exist_ok=True)
        file.save(upload_path)

        # ----------------------------------------------------
        # Predict sentiment
        # ----------------------------------------------------

        try:
            predictions, probabilities = predict_feedback(
                result["Feedback Text"]
            )
        except Exception as exc:
            upload_path.unlink(
                missing_ok=True
            )
            flash(
                f"Unable to analyse the uploaded feedback: {exc}",
                "danger"
            )
            return redirect(
                url_for("upload")
            )

        # ----------------------------------------------------
        # Add predictions
        # ----------------------------------------------------

        result["Sentiment"] = predictions

        if probabilities is not None:
            result["Prediction Confidence"] = [
                round(float(probability), 4)
                for probability in probabilities
            ]

        # ----------------------------------------------------
        # Save analysed dataset
        # ----------------------------------------------------

        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(
            DATA_PATH,
            index=False
        )

        flash(
            (
                f"Dataset uploaded and analysed successfully: "
                f"{len(result)} records. "
                "Sentiment was predicted by the trained model."
            ),
            "success"
        )

        return redirect(
            url_for("analysis")
        )

    return render_template(
        "upload.html",
        categories=VALID_CATEGORIES,
        active_page="upload"
    )


# ============================================================
# VIEW ANALYSIS
# ============================================================

@app.route("/analysis")
def analysis():
    df = load_data()

    category = request.args.get(
        "category",
        "All Categories"
    )

    sentiment = request.args.get(
        "sentiment",
        "All Sentiments"
    )

    search = request.args.get(
        "search",
        ""
    )

    filtered = apply_filters(
        df,
        category,
        sentiment,
        search
    )

    try:
        page = max(
            1,
            int(
                request.args.get(
                    "page",
                    1
                )
            )
        )
    except ValueError:
        page = 1

    per_page = 15

    total_pages = max(
        1,
        math.ceil(
            len(filtered) / per_page
        )
    )

    page = min(
        page,
        total_pages
    )

    start = (
        page - 1
    ) * per_page

    rows = (
        filtered
        .iloc[
            start:start + per_page
        ]
        .to_dict("records")
    )

    metadata = load_metadata()

    return render_template(
        "analysis.html",
        rows=rows,
        total=len(filtered),
        page=page,
        total_pages=total_pages,
        category=category,
        sentiment=sentiment,
        search=search,
        categories=VALID_CATEGORIES,
        sentiments=VALID_SENTIMENTS,
        metadata=metadata,
        active_page="analysis"
    )


# ============================================================
# REPORTS PAGE
# ============================================================

@app.route("/reports")
def reports():
    df = load_data()
    data = get_dashboard_data(df)
    metadata = load_metadata()

    return render_template(
        "reports.html",
        data=data,
        metadata=metadata,
        categories=VALID_CATEGORIES,
        active_page="reports"
    )


# ============================================================
# DOWNLOAD PDF REPORT
# ============================================================

@app.route("/download-report")
def download_report():
    """Generate and download the current analysis as a PDF report."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        )
    except ImportError:
        flash(
            "ReportLab is not installed. Please run: pip install reportlab",
            "danger"
        )
        return redirect(url_for("reports"))

    df = load_data()
    metadata = load_metadata()

    if df.empty:
        flash(
            "There is no analysed feedback available to generate a report.",
            "warning"
        )
        return redirect(url_for("reports"))

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
        fontSize=18, leading=22, spaceAfter=8
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], alignment=TA_CENTER,
        fontSize=10, leading=14, spaceAfter=15
    )
    heading_style = ParagraphStyle(
        "ReportHeading", parent=styles["Heading2"],
        fontSize=13, leading=16, spaceBefore=12, spaceAfter=8
    )
    normal_style = ParagraphStyle(
        "ReportNormal", parent=styles["Normal"], fontSize=9, leading=12
    )
    small_style = ParagraphStyle(
        "ReportSmall", parent=styles["Normal"], fontSize=7, leading=9
    )
    table_header_style = ParagraphStyle(
        "TableHeader", parent=styles["Normal"], fontSize=8,
        leading=10, textColor=colors.white
    )

    elements = [
        Paragraph("STUDENT FEEDBACK ANALYSIS SYSTEM", title_style),
        Paragraph("Sentiment Analysis Report", subtitle_style),
        Paragraph(
            "Generated: " + datetime.now().strftime("%d %B %Y, %H:%M:%S"),
            normal_style
        ),
        Spacer(1, 12)
    ]

    # 1. Analysis Summary
    elements.append(Paragraph("1. Analysis Summary", heading_style))
    total_records = len(df)
    sentiment_series = df["Sentiment"].astype(str).str.strip().str.lower()
    positive_count = int((sentiment_series == "positive").sum())
    neutral_count = int((sentiment_series == "neutral").sum())
    negative_count = int((sentiment_series == "negative").sum())

    def pct(count, total):
        return (count / total * 100) if total else 0.0

    summary_data = [
        [Paragraph("<b>Metric</b>", table_header_style),
         Paragraph("<b>Value</b>", table_header_style)],
        ["Total Feedback", str(total_records)],
        ["Positive Feedback", str(positive_count)],
        ["Neutral Feedback", str(neutral_count)],
        ["Negative Feedback", str(negative_count)]
    ]
    summary_table = Table(summary_data, colWidths=[250, 150])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F3F4F6")]),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))
    elements.append(summary_table)

    # 2. Sentiment Distribution
    elements.append(Paragraph("2. Sentiment Distribution", heading_style))
    sentiment_data = [
        [Paragraph("<b>Sentiment</b>", table_header_style),
         Paragraph("<b>Count</b>", table_header_style),
         Paragraph("<b>Percentage</b>", table_header_style)],
        ["Positive", str(positive_count),
         f"{pct(positive_count, total_records):.1f}%"],
        ["Neutral", str(neutral_count),
         f"{pct(neutral_count, total_records):.1f}%"],
        ["Negative", str(negative_count),
         f"{pct(negative_count, total_records):.1f}%"]
    ]
    sentiment_table = Table(sentiment_data, colWidths=[180, 100, 120])
    sentiment_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F3F4F6")])
    ]))
    elements.append(sentiment_table)

    # 3. Model Performance
    elements.append(Paragraph("3. Model Performance", heading_style))
    metrics = metadata.get("metrics", {})
    accuracy = float(metrics.get("accuracy", 0))
    precision = float(metrics.get("precision", 0))
    recall = float(metrics.get("recall", 0))
    f1_score = float(metrics.get("f1", 0))
    cv_accuracy = float(metrics.get("cv_mean_accuracy", 0))

    model_data = [
        [Paragraph("<b>Metric</b>", table_header_style),
         Paragraph("<b>Value</b>", table_header_style)],
        ["Accuracy", f"{accuracy * 100:.2f}%"],
        ["Precision", f"{precision * 100:.2f}%"],
        ["Recall", f"{recall * 100:.2f}%"],
        ["F1-score", f"{f1_score * 100:.2f}%"],
        ["5-Fold Cross-Validation Accuracy", f"{cv_accuracy * 100:.2f}%"]
    ]
    model_table = Table(model_data, colWidths=[280, 120])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F3F4F6")])
    ]))
    elements.append(model_table)

    # 4. Category Comparison
    elements.append(
        Paragraph("4. Category Sentiment Comparison", heading_style)
    )
    category_rows = [[
        Paragraph("<b>Category</b>", table_header_style),
        Paragraph("<b>Positive</b>", table_header_style),
        Paragraph("<b>Neutral</b>", table_header_style),
        Paragraph("<b>Negative</b>", table_header_style)
    ]]

    for category in VALID_CATEGORIES:
        category_df = df[df["Category"] == category]
        cs = category_df["Sentiment"].astype(str).str.strip().str.lower()
        category_rows.append([
            category,
            str(int((cs == "positive").sum())),
            str(int((cs == "neutral").sum())),
            str(int((cs == "negative").sum()))
        ])

    category_table = Table(
        category_rows, colWidths=[220, 70, 70, 70]
    )
    category_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F3F4F6")])
    ]))
    elements.append(category_table)

    # 5. Detailed Feedback Results
    elements.append(PageBreak())
    elements.append(Paragraph("5. Detailed Feedback Results", heading_style))
    feedback_rows = [[
        Paragraph("<b>ID</b>", table_header_style),
        Paragraph("<b>Feedback Text</b>", table_header_style),
        Paragraph("<b>Category</b>", table_header_style),
        Paragraph("<b>Sentiment</b>", table_header_style),
        Paragraph("<b>Date</b>", table_header_style)
    ]]

    for _, row in df.iterrows():
        feedback_text = str(row.get("Feedback Text", ""))
        if len(feedback_text) > 150:
            feedback_text = feedback_text[:147] + "..."

        feedback_rows.append([
            str(row.get("ID", "")),
            Paragraph(feedback_text, small_style),
            Paragraph(str(row.get("Category", "")), small_style),
            Paragraph(str(row.get("Sentiment", "")), small_style),
            str(row.get("Date", ""))
        ])

    feedback_table = Table(
        feedback_rows,
        colWidths=[35, 205, 105, 65, 70],
        repeatRows=1
    )
    feedback_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F8FAFC")])
    ]))
    elements.append(feedback_table)
    elements.append(Spacer(1, 15))
    elements.append(
        Paragraph(
            "Generated by the Student Feedback Analysis System.",
            normal_style
        )
    )

    document.build(elements)
    buffer.seek(0)

    res = send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="student_feedback_analysis_report.pdf"
    )
    res.headers["Content-Type"] = "application/pdf"
    res.headers["Content-Disposition"] = (
        'attachment; filename="student_feedback_analysis_report.pdf"'
    )
    res.headers["Cache-Control"] = "no-store"
    return res


# ============================================================
# SETTINGS
# ============================================================

@app.route("/settings")
def settings():
    metadata = load_metadata()

    return render_template(
        "settings.html",
        metadata=metadata,
        categories=VALID_CATEGORIES,
        active_page="settings"
    )


# ============================================================
# DASHBOARD API
# ============================================================

@app.route("/api/dashboard")
def api_dashboard():
    return jsonify(
        get_dashboard_data()
    )


# ============================================================
# FILE TOO LARGE ERROR HANDLER
# ============================================================

@app.errorhandler(413)
def too_large(_):
    flash(
        "The uploaded file exceeds the 10 MB limit.",
        "danger"
    )
    return redirect(
        url_for("upload")
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True
    )