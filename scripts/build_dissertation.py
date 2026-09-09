"""Build the USW-compliant dissertation DOCX from the markdown chapter files.

Format rules implemented (MSc Project Handbook):
- Times New Roman, 12pt body, 1.5 line spacing, justified
- H1 16pt bold, H2 14pt bold, H3 12pt bold italic
- Page numbers in footer (roman for front matter, arabic from Chapter 1)
- Figures numbered per chapter, caption BELOW; tables numbered per chapter, caption ABOVE
- Title sheet, declaration, abstract, acknowledgements, TOC field, lists of figures/tables
"""

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
DISS = ROOT / "dissertation"
OUT = DISS / "USW_MSc_Dissertation_Ridita_Adikta.docx"

FONT = "Times New Roman"
BODY_SIZE = Pt(12)

CHAPTER_FILES = [
    "00_front.md",
    "01_introduction.md",
    "02_literature.md",
    "03_methodology.md",
    "04_results.md",
    "05_explainability.md",
    "06_discussion.md",
    "07_conclusions.md",
    "08_references.md",
    "09_appendices.md",
]


# ---------------------------------------------------------------- utilities

def set_run_font(run, size=BODY_SIZE, bold=False, italic=False):
    run.font.name = FONT
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rFonts.set(qn(attr), FONT)


def style_paragraph(p, size=BODY_SIZE, bold=False, italic=False,
                    align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=Pt(6),
                    first_indent=None):
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_after = space_after
    pf.space_before = Pt(0)
    if first_indent is not None:
        pf.first_line_indent = first_indent
    for run in p.runs:
        set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def add_para(doc, text, **kw):
    p = doc.add_paragraph(text)
    return style_paragraph(p, **kw)


def add_field(paragraph, field_code):
    """Insert a Word field (e.g. PAGE, TOC) into a paragraph."""
    run = paragraph.add_run()
    set_run_font(run)
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = field_code
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    run._element.append(fld1)
    run._element.append(instr)
    run._element.append(fld2)
    return run


def add_page_number_footer(section, fmt="decimal"):
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.text = ""
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(p, " PAGE ")
    # page number format
    sectPr = section._sectPr
    pgNumType = sectPr.find(qn("w:pgNumType"))
    if pgNumType is None:
        pgNumType = OxmlElement("w:pgNumType")
        sectPr.append(pgNumType)
    pgNumType.set(qn("w:fmt"), fmt)


def add_heading(doc, text, level):
    sizes = {1: Pt(16), 2: Pt(14), 3: Pt(12)}
    p = doc.add_paragraph()
    p.style = doc.styles[f"Heading {level}"]
    run = p.add_run(text)
    set_run_font(run, size=sizes[level], bold=True, italic=(level == 3))
    run.font.color.rgb = RGBColor(0, 0, 0)
    pf = p.paragraph_format
    pf.space_before = Pt(12)
    pf.space_after = Pt(6)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    return p


def parse_md_table(lines):
    """Parse markdown table lines into list of row lists."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        rows.append(cells)
    return rows


def add_table(doc, caption, rows, table_num):
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Table {table_num}: {caption}")
    set_run_font(r, bold=True)
    cap.paragraph_format.space_before = Pt(8)
    cap.paragraph_format.space_after = Pt(4)
    cap.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    n_cols = max(len(rw) for rw in rows)
    tbl = doc.add_table(rows=len(rows), cols=n_cols)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j in range(n_cols):
            cell = tbl.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.paragraph_format.space_after = Pt(2)
            text = row[j] if j < len(row) else ""
            run = p.add_run(text)
            set_run_font(run, size=Pt(10), bold=(i == 0))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tbl


def add_figure(doc, path, caption, fig_num):
    img = ROOT / path
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(img), width=Inches(5.8))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Figure {fig_num}: {caption}")
    set_run_font(r, bold=True)
    cap.paragraph_format.space_after = Pt(8)
    cap.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


# ---------------------------------------------------------------- parsing

def parse_chapters():
    """Parse all chapter files into a stream of blocks."""
    blocks = []
    for fname in CHAPTER_FILES:
        text = (DISS / fname).read_text(encoding="utf-8")
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            s = line.strip()
            if not s:
                i += 1
                continue
            if s.startswith("#TITLE# "):
                blocks.append(("title", s[8:].strip()))
            elif s == "#META#":
                meta = {}
                i += 1
                while i < len(lines) and lines[i].strip() and ":" in lines[i]:
                    k, v = lines[i].split(":", 1)
                    meta[k.strip()] = v.strip()
                    i += 1
                blocks.append(("meta", meta))
                continue
            elif s == "#DECLARATION#":
                body = []
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].startswith("#"):
                    body.append(lines[i].strip())
                    i += 1
                blocks.append(("declaration", " ".join(body)))
                continue
            elif s == "#ABSTRACT#":
                body = []
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].startswith("#"):
                    body.append(lines[i].strip())
                    i += 1
                blocks.append(("abstract", " ".join(body)))
                continue
            elif s == "#ACKNOWLEDGEMENTS#":
                body = []
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].startswith("#"):
                    body.append(lines[i].strip())
                    i += 1
                blocks.append(("acknowledgements", " ".join(body)))
                continue
            elif s.startswith("#H1# "):
                blocks.append(("h1", s[5:].strip()))
            elif s.startswith("#H1NR# "):
                blocks.append(("h1nr", s[7:].strip()))
            elif s.startswith("##H2## "):
                blocks.append(("h2", s[7:].strip()))
            elif s.startswith("###H3### "):
                blocks.append(("h3", s[9:].strip()))
            elif s.startswith("#TABLE# "):
                caption = s[8:].strip()
                tbl_lines = []
                i += 1
                while i < len(lines) and lines[i].strip().startswith("|"):
                    tbl_lines.append(lines[i])
                    i += 1
                blocks.append(("table", (caption, parse_md_table(tbl_lines))))
                continue
            elif s.startswith("#FIGURE# "):
                path = s[9:].strip()
                caption = ""
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("#CAPTION# "):
                    caption = lines[i + 1].strip()[10:].strip()
                    i += 1
                blocks.append(("figure", (path, caption)))
            elif s.startswith("#QUOTE# "):
                blocks.append(("quote", s[8:].strip()))
            elif s == "#LIST#":
                items = []
                i += 1
                while i < len(lines) and lines[i].strip().startswith("- "):
                    items.append(lines[i].strip()[2:].strip())
                    i += 1
                blocks.append(("list", items))
                continue
            else:
                blocks.append(("para", s))
            i += 1
    return blocks


# ---------------------------------------------------------------- build

def build():
    blocks = parse_chapters()
    doc = Document()

    # base styles
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = BODY_SIZE

    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)

    title = next(v for k, v in blocks if k == "title")
    meta = next(v for k, v in blocks if k == "meta")

    # ---------------- Title sheet
    for _ in range(3):
        doc.add_paragraph()
    p = add_para(doc, "UNIVERSITY OF SOUTH WALES",
                 align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=Pt(16))
    add_para(doc, "Faculty of Computing, Engineering and Science",
             align=WD_ALIGN_PARAGRAPH.CENTER, size=Pt(13))
    for _ in range(3):
        doc.add_paragraph()
    add_para(doc, title, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=Pt(15))
    for _ in range(2):
        doc.add_paragraph()
    add_para(doc, "by", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, meta["name"], align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=Pt(13))
    add_para(doc, f"Enrolment number: {meta['enrolment']}",
             align=WD_ALIGN_PARAGRAPH.CENTER)
    for _ in range(2):
        doc.add_paragraph()
    add_para(doc,
             "A project submitted in partial fulfilment of the requirements for the degree of",
             align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, f"Master of Science, {meta['scheme']}",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
    for _ in range(2):
        doc.add_paragraph()
    add_para(doc, f"Supervisors: {meta['supervisor1']} and {meta['supervisor2']}",
             align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, f"Academic year: {meta['year']}", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "September 2026", align=WD_ALIGN_PARAGRAPH.CENTER)

    # ---------------- Declaration of originality
    doc.add_page_break()
    add_heading(doc, "Declaration of Originality", 1)
    decl = next(v for k, v in blocks if k == "declaration")
    add_para(doc, decl)
    doc.add_paragraph()
    add_para(doc, "Signed: ............................................",
             align=WD_ALIGN_PARAGRAPH.LEFT)
    doc.add_paragraph()
    add_para(doc, "Date: ............................................",
             align=WD_ALIGN_PARAGRAPH.LEFT)

    # ---------------- Abstract
    doc.add_page_break()
    add_heading(doc, "Abstract", 1)
    abstract = next(v for k, v in blocks if k == "abstract")
    add_para(doc, abstract)

    # ---------------- Acknowledgements
    doc.add_page_break()
    add_heading(doc, "Acknowledgements", 1)
    ack = next(v for k, v in blocks if k == "acknowledgements")
    add_para(doc, ack)

    # ---------------- Table of contents (field — update in Word)
    doc.add_page_break()
    add_heading(doc, "Table of Contents", 1)
    toc_p = doc.add_paragraph()
    add_field(toc_p, ' TOC \\o "1-3" \\h \\z \\u ')
    add_para(doc, "(In Word: right-click the table of contents and choose "
                  "\u201cUpdate Field\u201d to populate page numbers.)",
             italic=True, size=Pt(10))

    # ---------------- Lists of figures and tables (built from parse)
    chapter_no = 0
    fig_counters = {}
    tab_counters = {}
    fig_list = []
    tab_list = []
    current = None
    for kind, val in blocks:
        if kind == "h1":
            m = re.match(r"^(\d+)\s", val)
            current = int(m.group(1)) if m else None
        elif kind == "h1nr":
            current = val.split()[1].rstrip(":") if val.startswith("Appendix") else None
            if val == "References":
                current = None
        elif kind == "figure" and current is not None:
            fig_counters[current] = fig_counters.get(current, 0) + 1
            fig_list.append((f"{current}.{fig_counters[current]}", val[1]))
        elif kind == "table" and current is not None:
            tab_counters[current] = tab_counters.get(current, 0) + 1
            tab_list.append((f"{current}.{tab_counters[current]}", val[0]))

    doc.add_page_break()
    add_heading(doc, "List of Figures", 1)
    for num, cap in fig_list:
        add_para(doc, f"Figure {num}: {cap}", align=WD_ALIGN_PARAGRAPH.LEFT,
                 space_after=Pt(3))
    doc.add_page_break()
    add_heading(doc, "List of Tables", 1)
    for num, cap in tab_list:
        add_para(doc, f"Table {num}: {cap}", align=WD_ALIGN_PARAGRAPH.LEFT,
                 space_after=Pt(3))

    # ---------------- Main body: new section, arabic page numbers from 1
    new_section = doc.add_section(WD_SECTION.NEW_PAGE)
    new_section.top_margin = Cm(2.54)
    new_section.bottom_margin = Cm(2.54)
    new_section.left_margin = Cm(2.54)
    new_section.right_margin = Cm(2.54)

    front_section = doc.sections[0]
    add_page_number_footer(front_section, fmt="lowerRoman")
    add_page_number_footer(new_section, fmt="decimal")
    pgNumType = new_section._sectPr.find(qn("w:pgNumType"))
    pgNumType.set(qn("w:start"), "1")

    chapter_no = 0
    fig_counters = {}
    tab_counters = {}
    skip_kinds = {"title", "meta", "declaration", "abstract", "acknowledgements"}
    word_count = 0

    def count_words(text):
        return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\u2019\-]*", text))

    for kind, val in blocks:
        if kind in skip_kinds:
            continue
        if kind == "h1":
            m = re.match(r"^(\d+)\s", val)
            chapter_no = int(m.group(1)) if m else 0
            doc.add_page_break()
            add_heading(doc, val, 1)
            word_count += count_words(val)
        elif kind == "h1nr":
            doc.add_page_break()
            add_heading(doc, val, 1)
            chapter_no = val.split()[1].rstrip(":") if val.startswith("Appendix") else None
            word_count += count_words(val)
        elif kind == "h2":
            add_heading(doc, val, 2)
            word_count += count_words(val)
        elif kind == "h3":
            add_heading(doc, val, 3)
            word_count += count_words(val)
        elif kind == "para":
            add_para(doc, val)
            word_count += count_words(val)
        elif kind == "quote":
            p = add_para(doc, val, italic=True,
                         align=WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.left_indent = Cm(1.27)
            p.paragraph_format.right_indent = Cm(1.27)
            word_count += count_words(val)
        elif kind == "list":
            for item in val:
                p = doc.add_paragraph(style="List Bullet")
                run = p.add_run(item)
                set_run_font(run)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
                p.paragraph_format.space_after = Pt(3)
                word_count += count_words(item)
        elif kind == "table":
            caption, rows = val
            key = chapter_no if chapter_no is not None else "A"
            tab_counters[key] = tab_counters.get(key, 0) + 1
            add_table(doc, caption, rows, f"{key}.{tab_counters[key]}")
            for row in rows:
                for cell in row:
                    word_count += count_words(cell)
        elif kind == "figure":
            path, caption = val
            key = chapter_no if chapter_no is not None else "A"
            fig_counters[key] = fig_counters.get(key, 0) + 1
            add_figure(doc, path, caption, f"{key}.{fig_counters[key]}")
            word_count += count_words(caption)

    try:
        doc.save(OUT)
        saved = OUT
    except PermissionError:
        saved = OUT.with_name(OUT.stem + "_updated.docx")
        doc.save(saved)
        print(f"NOTE: {OUT.name} was locked (open in Word?) - saved to {saved.name}")
    print(f"Saved: {saved}")
    print(f"Approx word count (body incl. tables/captions): {word_count}")
    print(f"Figures: {sum(fig_counters.values())}, Tables: {sum(tab_counters.values())}")


if __name__ == "__main__":
    build()
