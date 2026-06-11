import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER


def generate_pairwise_report(file_a_name, file_b_name, result, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                 fontSize=18, alignment=TA_CENTER,
                                 spaceAfter=12)
    elements = []

    elements.append(Paragraph("BÁO CÁO KẾT QUẢ KIỂM TRA ĐẠO VĂN", title_style))
    elements.append(Spacer(1, 10*mm))

    info = [
        ["File kiểm tra A:", file_a_name],
        ["File kiểm tra B:", file_b_name],
        ["Thời gian:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    info_table = Table(info, colWidths=[120*mm, 80*mm])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 8*mm))

    final_score = result.get("final_score", 0)
    level = result.get("level", "Không xác định")
    elements.append(Paragraph(
        f'<b>Tỷ lệ trùng lặp tổng: {final_score*100:.1f}%</b>', styles["Normal"]))
    elements.append(Paragraph(f'<b>Đánh giá: {level}</b>', styles["Normal"]))
    elements.append(Spacer(1, 8*mm))

    s1 = result.get("score_1", 0)
    s2 = result.get("score_2", 0)
    s3 = result.get("score_3", 0)
    details = [
        ["Thành phần", "Điểm số"],
        ["TF-IDF (Cú pháp)", f"{s1*100:.1f}%"],
        ["Shingling (Cấu trúc)", f"{s2*100:.1f}%"],
        ["Semantic (Ngữ nghĩa)", f"{s3*100:.1f}%"],
        ["Ensemble (Tổng hợp)", f"{final_score*100:.1f}%"],
    ]
    dt = Table(details, colWidths=[120*mm, 80*mm])
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(dt)
    elements.append(Spacer(1, 10*mm))

    matches = result.get("matches", [])
    if matches:
        elements.append(Paragraph("<b>Các đoạn trùng phát hiện (gộp từ câu liền kề):</b>", styles["Heading2"]))
        elements.append(Spacer(1, 4*mm))
        for i, m in enumerate(matches[:15], 1):
            num = m.get("num_sentences", 1)
            is_ck = m.get("is_common_knowledge", False)
            sent_range = f"(câu {m['orig_idx_start']+1}-{m['orig_idx_end']+1})" if num > 1 else f"(câu {m['orig_idx_start']+1})"
            color = "gray" if is_ck else "red"
            ck_label = " [Common Knowledge]" if is_ck else ""
            elements.append(Paragraph(
                f'{i}. <font color="{color}"><b>[{m["type"].upper()}{ck_label}]</b></font> '
                f'{num} câu liền kề — Độ tương đồng: {m["score"]*100:.1f}% {sent_range}',
                styles["Normal"]))
            elements.append(Paragraph(
                f'&nbsp;&nbsp;<b>File A:</b> {m["orig_text"]}',
                styles["Normal"]))
            elements.append(Paragraph(
                f'&nbsp;&nbsp;<b>File B:</b> {m["susp_text"]}',
                styles["Normal"]))
            elements.append(Spacer(1, 3*mm))

    doc.build(elements)
    return output_path


def generate_library_report(filename, results, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                 fontSize=18, alignment=TA_CENTER,
                                 spaceAfter=12)
    elements = []

    elements.append(Paragraph("BÁO CÁO KẾT QUẢ KIỂM TRA ĐẠO VĂN", title_style))
    elements.append(Spacer(1, 6*mm))
    elements.append(Paragraph(
        f"File kiểm tra: <b>{filename}</b>", styles["Normal"]))
    elements.append(Paragraph(
        f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 8*mm))

    avg_score = results.get("avg_score", 0)
    level = results.get("level", "Không xác định")
    elements.append(Paragraph(
        f'<b>Tỷ lệ trùng lặp trung bình: {avg_score*100:.1f}%</b>', styles["Normal"]))
    elements.append(Paragraph(f'<b>Đánh giá: {level}</b>', styles["Normal"]))
    elements.append(Spacer(1, 8*mm))

    top_sources = results.get("top_sources", [])
    if top_sources:
        table_data = [["#", "Tên file nguồn", "Tỷ lệ trùng", "Mức độ"]]
        for i, src in enumerate(top_sources, 1):
            table_data.append([
                str(i),
                src["filename"],
                f'{src["score"]*100:.1f}%',
                src.get("level", "")
            ])
        t = Table(table_data, colWidths=[15*mm, 80*mm, 40*mm, 65*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(t)

    doc.build(elements)
    return output_path
