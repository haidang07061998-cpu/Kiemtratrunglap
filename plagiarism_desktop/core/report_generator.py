import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# FIX: Đăng ký font Unicode hỗ trợ tiếng Việt — đa nền tảng
_FONT_NAME = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"
_FONT_CANDIDATES = [
    ("ArialUnicode", "ArialUnicode-Bold", [
        "C:\\Windows\\Fonts\\ARIALUNI.TTF",
    ]),
    ("Arial", "Arial-Bold", [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\ARIALN.TTF",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    ]),
    ("Tahoma", "Tahoma-Bold", [
        "C:\\Windows\\Fonts\\tahoma.ttf",
        "/usr/share/fonts/truetype/tahoma.ttf",
    ]),
    ("TimesNewRoman", "TimesNewRoman-Bold", [
        "C:\\Windows\\Fonts\\times.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    ]),
]
_registered = False
for fname, fbold, paths in _FONT_CANDIDATES:
    for p in paths:
        if os.path.exists(p):
            try:
                pdfmetrics.registerFont(TTFont(fname, p))
                _FONT_NAME = fname
                bold_p = (
                    p.replace("ARIALUNI.TTF", "arialbd.ttf")
                     .replace("arial.ttf", "arialbd.ttf")
                     .replace("ARIALN.TTF", "ARIALNB.TTF")
                     .replace("tahoma.ttf", "tahomabd.ttf")
                     .replace("times.ttf", "timesbd.ttf")
                     .replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
                )
                if os.path.exists(bold_p):
                    try:
                        pdfmetrics.registerFont(TTFont(fbold, bold_p))
                        _FONT_BOLD = fbold
                    except Exception:
                        _FONT_BOLD = fname
                else:
                    _FONT_BOLD = fname
                _registered = True
                break
            except Exception:
                continue
    if _registered:
        break


def _make_style(name, parent, **kwargs):
    if 'fontName' not in kwargs:
        kwargs['fontName'] = _FONT_NAME
    return ParagraphStyle(name, parent=parent, **kwargs)


def generate_pairwise_report(file_a_name, file_b_name, result, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    # Override all styles to use Unicode font
    for s in styles.byName.values():
        s.fontName = _FONT_NAME
    title_style = _make_style("Title2", styles["Title"],
                              fontSize=18, alignment=TA_CENTER,
                              spaceAfter=12)
    heading_style = _make_style("Heading2Custom", styles["Heading2"],
                                fontName=_FONT_BOLD)
    normal_style = _make_style("NormalCustom", styles["Normal"])
    elements = []

    elements.append(Paragraph("BÁO CÁO KẾT QUẢ KIỂM TRA ĐẠO VĂN", title_style))
    elements.append(Spacer(1, 10*mm))

    info = [
        ["File kiểm tra A:", file_a_name],
        ["File kiểm tra B:", file_b_name],
        ["Thời gian:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    info_table = Table(info, colWidths=[50*mm, 120*mm])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), _FONT_BOLD),
        ("FONTNAME", (1, 0), (1, -1), _FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 8*mm))

    s1 = result.get("score_1", 0)
    s2 = result.get("score_2", 0)
    s3 = result.get("score_3", 0)
    coverage_a = result.get("coverage_a", 0)
    coverage_b = result.get("coverage_b", 0)

    # Overall score = Coverage×0.70 + Shingling×0.30 (same as web UI)
    overall = min((coverage_a * 0.70) + (s2 * 0.30), 1.0)
    overall_pct = round(overall * 100)
    if overall_pct >= 30: verdict = "Đạo văn nghiêm trọng"
    elif overall_pct >= 15: verdict = "Có dấu hiệu đạo văn rõ"
    elif overall_pct >= 5: verdict = "Có dấu hiệu cần kiểm tra"
    else: verdict = "Không phát hiện đạo văn"

    elements.append(Paragraph(
        f'<b>Similarity Index: {overall_pct}%</b>', styles["Normal"]))
    elements.append(Paragraph(f'<b>Đánh giá: {verdict}</b>', styles["Normal"]))
    elements.append(Spacer(1, 6*mm))

    # Formula breakdown
    cov_part = coverage_a * 0.70 * 100
    shingle_part = s2 * 0.30 * 100
    formula_lines = [
        f'Công thức = Coverage×0.70 + Shingling×0.30',
        f'  Coverage ({coverage_a*100:.1f}%) × 0.70 = {cov_part:.1f}%',
        f'  Shingling ({s2*100:.1f}%) × 0.30 = {shingle_part:.1f}%',
        f'  Tổng = {overall_pct}%',
    ]
    for line in formula_lines:
        elements.append(Paragraph(line, styles["Normal"]))
    elements.append(Spacer(1, 6*mm))

    # BERT warning if applicable
    if s3 >= 0.90 and coverage_a < 0.10:
        bert_warn = "\u26A0\ufe0f BERT %.1f%% \u2014 c\u1EA5u tr\u00FAc \u00FD t\u01B0\u1EDFng r\u1EA5t t\u01B0\u01A1ng \u0111\u1ED3ng. Kh\u00F4ng t\u00EDnh v\u00E0o \u0111i\u1EC3m t\u1ED5ng." % (s3 * 100)
        elements.append(Paragraph(
            '<font color="orange"><b>%s</b></font>' % bert_warn,
            styles["Normal"]))
        elements.append(Spacer(1, 6*mm))

    details = [
        ["Thành phần", "Điểm số"],
        ["TF-IDF (Cú pháp)", f"{s1*100:.1f}%"],
        ["Shingling (Cấu trúc)", f"{s2*100:.1f}%"],
        ["Semantic (Ngữ nghĩa)", f"{s3*100:.1f}%"],
        ["Coverage A (% File A bị trùng)", f"{coverage_a*100:.1f}%"],
        ["Coverage B (% File B bị trùng)", f"{coverage_b*100:.1f}%"],
    ]
    dt = Table(details, colWidths=[120*mm, 50*mm])
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), _FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), _FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(dt)
    elements.append(Spacer(1, 10*mm))

    matches = result.get("matches", [])
    if matches:
        susp_matches = [m for m in matches if not (m.get("is_common_knowledge", False) or m["type"] == "common_knowledge")]
        ck_matches = [m for m in matches if m.get("is_common_knowledge", False) or m["type"] == "common_knowledge"]

        def add_match_section(title, match_list, color_fn):
            if not match_list:
                return
            elements.append(Paragraph("<b>%s</b>" % title, styles["Heading2"]))
            elements.append(Spacer(1, 3*mm))
            for i, m in enumerate(match_list[:15], 1):
                num = m.get("num_sentences", 1)
                sent_range = "(c\u00E2u %d-%d)" % (m['orig_idx_start']+1, m['orig_idx_end']+1) if num > 1 else "(c\u00E2u %d)" % (m['orig_idx_start']+1)
                color = color_fn(m)
                elements.append(Paragraph(
                    '%d. <font color="%s"><b>[%s]</b></font> %d c\u00E2u li\u1EC1n k\u1EC1 \u2014 \u0110\u1ED9 t\u01B0\u01A1ng \u0111\u1ED3ng: %.1f%% %s' % (
                        i, color, m["type"].upper(), num, m["score"] * 100, sent_range),
                    styles["Normal"]))
                elements.append(Paragraph(
                    '&nbsp;&nbsp;<b>File A:</b> %s' % m["orig_text"],
                    styles["Normal"]))
                elements.append(Paragraph(
                    '&nbsp;&nbsp;<b>File B:</b> %s' % m["susp_text"],
                    styles["Normal"]))
                elements.append(Spacer(1, 2*mm))
            elements.append(Spacer(1, 4*mm))

        add_match_section(
            "Doan nghi ngo dao van (PARAPHRASE / COPY):",
            susp_matches, lambda m: "red")
        add_match_section(
            "Kien thuc chung (COMMON KNOWLEDGE):",
            ck_matches, lambda m: "gray")

    doc.build(elements)
    return output_path


def generate_library_report(filename, results, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    for s in styles.byName.values():
        s.fontName = _FONT_NAME
    title_style = _make_style("LibTitle2", styles["Title"],
                              fontSize=18, alignment=TA_CENTER,
                              spaceAfter=12)
    normal_style = _make_style("LibNormal", styles["Normal"])
    elements = []

    elements.append(Paragraph("BÁO CÁO KẾT QUẢ KIỂM TRA ĐẠO VĂN", title_style))
    elements.append(Spacer(1, 6*mm))
    elements.append(Paragraph(
        f"File kiểm tra: <b>{filename}</b>", normal_style))
    elements.append(Paragraph(
        f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 8*mm))

    avg_score = results.get("avg_score", 0)
    level = results.get("level", "Không xác định")

    # Overall score = max coverage*0.70 + shingling*0.30 (approximate)
    top = results.get("top_sources", [])
    top_s1 = top[0].get("s1", 0) if top else 0
    overall_pct = round(avg_score * 100)
    if overall_pct >= 30: verdict = "Đạo văn nghiêm trọng"
    elif overall_pct >= 15: verdict = "Có dấu hiệu đạo văn rõ"
    elif overall_pct >= 5: verdict = "Có dấu hiệu cần kiểm tra"
    else: verdict = "Không phát hiện đạo văn"

    elements.append(Paragraph(
        f'<b>Similarity Index: {overall_pct}%</b>', normal_style))
    elements.append(Paragraph(f'<b>Đánh giá: {verdict}</b>', normal_style))
    elements.append(Spacer(1, 8*mm))

    top_sources = results.get("top_sources", [])
    if top_sources:
        table_data = [["#", "Tên file nguồn", "Tỷ lệ trùng", "Coverage A", "Coverage B", "Mức độ"]]
        for i, src in enumerate(top_sources, 1):
            cov_a = src.get("coverage_a", 0)
            cov_b = src.get("coverage_b", 0)
            table_data.append([
                str(i),
                src["filename"],
                f'{src["score"]*100:.1f}%',
                f'{cov_a*100:.1f}%',
                f'{cov_b*100:.1f}%',
                src.get("level", "")
            ])
        t = Table(table_data, colWidths=[12*mm, 50*mm, 25*mm, 25*mm, 25*mm, 33*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), _FONT_BOLD),
            ("FONTNAME", (0, 1), (-1, -1), _FONT_NAME),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(t)

    doc.build(elements)
    return output_path
