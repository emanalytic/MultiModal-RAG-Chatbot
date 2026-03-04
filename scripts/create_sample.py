"""
Generate a sample PDF for testing the parser.
Creates a multi-page PDF with headings, paragraphs, a table, and an image.
"""

import fitz

import os

def create_sample_pdf(output_path: str = None):
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "data", "samples", "sample.pdf")
    doc = fitz.open()

    # Page 1: Title + Introduction
    page = doc.new_page(width=612, height=792)

    # Title
    page.insert_text(
        (72, 80),
        "Research Report: Analysis of AI Trends",
        fontsize=22,
        fontname="helv",
        color=(0.1, 0.1, 0.4),
    )

    # Section heading
    page.insert_text(
        (72, 130),
        "1. Introduction",
        fontsize=16,
        fontname="helv",
        color=(0, 0, 0),
    )

    # Paragraph
    intro_text = (
        "This report provides an in-depth analysis of current trends in artificial "
        "intelligence research. We examine key developments in natural language processing, "
        "computer vision, and reinforcement learning over the past five years. The findings "
        "are based on a comprehensive survey of 500 peer-reviewed publications and "
        "industry reports from major technology companies."
    )
    rect = fitz.Rect(72, 155, 540, 280)
    page.insert_textbox(rect, intro_text, fontsize=11, fontname="helv")

    # Another heading
    page.insert_text(
        (72, 310),
        "2. Methodology",
        fontsize=16,
        fontname="helv",
        color=(0, 0, 0),
    )

    method_text = (
        "We conducted a systematic literature review following the PRISMA guidelines. "
        "Databases searched included IEEE Xplore, ACM Digital Library, and arXiv. "
        "Keywords used were: deep learning, transformer architecture, generative AI, "
        "and large language models. Papers published between 2020 and 2025 were included."
    )
    rect2 = fitz.Rect(72, 335, 540, 450)
    page.insert_textbox(rect2, method_text, fontsize=11, fontname="helv")

    # Page 2: Results with a table
    page2 = doc.new_page(width=612, height=792)

    page2.insert_text(
        (72, 80),
        "3. Results",
        fontsize=16,
        fontname="helv",
        color=(0, 0, 0),
    )

    results_text = (
        "The analysis revealed significant growth in publications related to "
        "large language models, with a 340% increase from 2020 to 2024. "
        "Table 1 summarizes the key metrics across different AI subfields."
    )
    rect3 = fitz.Rect(72, 105, 540, 180)
    page2.insert_textbox(rect3, results_text, fontsize=11, fontname="helv")

    # Draw a simple table
    table_top = 200
    col_widths = [180, 100, 100, 100]
    row_height = 25
    headers = ["AI Subfield", "Papers", "Growth %", "Impact"]
    rows = [
        ["Natural Language Processing", "187", "340%", "High"],
        ["Computer Vision", "142", "180%", "High"],
        ["Reinforcement Learning", "89", "95%", "Medium"],
        ["Generative AI", "82", "520%", "Very High"],
    ]

    x_start = 72
    for row_idx, row_data in enumerate([headers] + rows):
        y = table_top + row_idx * row_height
        x = x_start
        for col_idx, cell in enumerate(row_data):
            rect_cell = fitz.Rect(x, y, x + col_widths[col_idx], y + row_height)
            page2.draw_rect(rect_cell, color=(0.3, 0.3, 0.3), width=0.5)
            page2.insert_textbox(
                rect_cell,
                cell,
                fontsize=10,
                fontname="helv",
                align=1 if col_idx > 0 else 0,
            )
            x += col_widths[col_idx]

    # Caption under table
    page2.insert_text(
        (72, table_top + 6 * row_height + 5),
        "Table 1: Summary of AI research metrics by subfield (2020–2024)",
        fontsize=9,
        fontname="helv",
        color=(0.4, 0.4, 0.4),
    )

    # Page 3: Conclusion
    page3 = doc.new_page(width=612, height=792)

    page3.insert_text(
        (72, 80),
        "4. Conclusion",
        fontsize=16,
        fontname="helv",
        color=(0, 0, 0),
    )

    conclusion_text = (
        "The rapid advancement of AI technologies presents both opportunities and "
        "challenges for the research community. Large language models have emerged as "
        "the dominant force in NLP research, while generative AI shows the fastest "
        "growth rate. Future research should focus on improving model efficiency, "
        "reducing computational costs, and addressing ethical considerations. "
        "Interdisciplinary collaboration will be essential for responsible AI development."
    )
    rect4 = fitz.Rect(72, 105, 540, 250)
    page3.insert_textbox(rect4, conclusion_text, fontsize=11, fontname="helv")

    # Draw a simple colored rectangle as a "figure" placeholder
    fig_rect = fitz.Rect(120, 280, 480, 450)
    page3.draw_rect(fig_rect, color=(0.2, 0.4, 0.8), fill=(0.85, 0.9, 0.95), width=1)
    page3.insert_textbox(
        fitz.Rect(150, 340, 450, 400),
        "[ Chart: AI Publication Trends 2020-2024 ]",
        fontsize=14,
        fontname="helv",
        color=(0.3, 0.3, 0.6),
        align=1,
    )

    page3.insert_text(
        (120, 470),
        "Figure 1: Growth of AI publications across subfields over five years",
        fontsize=9,
        fontname="helv",
        color=(0.4, 0.4, 0.4),
    )

    doc.save(output_path)
    doc.close()
    print(f"Created sample PDF: {output_path}")


if __name__ == "__main__":
    create_sample_pdf()
