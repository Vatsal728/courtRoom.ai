"""
document_factory.py - Unified Document Engine for courtRoom.ai.
Generates printable, advocate-grade PDFs (via ReportLab) and editable DOCX files (via python-docx)
with full structural parity (tables, statutory citations, and receiving counterfoils)
for all core Indian legal tools:
1. Legal Demand Notice (Landlord-tenant, consumer, money recovery, contract)
2. RTI Application (Form A, Section 6(1) & 6(3) RTI Act 2005)
3. Cyber Fraud & UPI Scam Emergency (Bank account freeze + Cyber cell complaint)
4. Police Complaint to SHO (Section 173 BNSS 2023 & BNS 2023 with police receiving counterfoil)
5. Unpaid Salary & Labour Dues (HR Demand Letter + Labour Commissioner Claim under Code on Wages 2019)
6. Case Strategy & Legal Opinion Memo (Roadmap, forum analysis, compensation estimate)
"""
import os
import re
import sys
import io
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from pathlib import Path

# ReportLab imports
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

# python-docx imports
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from src.paths import resolve
from src.groq_drafter import normalize_tool_params

# Ensure console utf-8
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


def _ensure_output_dir() -> Path:
    out_dir = resolve("output")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ── DYNAMIC STATUTORY INFERENCE HELPERS ─────────────────────────────────────
def infer_bns_offence_sections(facts_text: str) -> str:
    """Dynamically determine applicable Bharatiya Nyaya Sanhita (BNS 2023) sections from facts."""
    t = (facts_text or "").lower()
    matched: List[str] = []

    rules = [
        (["snatch", "snatched", "snatching", "chain snatch"], "Section 304 BNS 2023 (Snatching)"),
        (["theft", "stole", "stolen", "steal", "pickpocket", "shoplift"], "Section 303(2) BNS 2023 (Theft)"),
        (["robbery", "robbed", "mugged", "gunpoint", "knifepoint"], "Section 309 BNS 2023 (Robbery)"),
        (["extort", "extortion", "ransom", "blackmail"], "Section 308 BNS 2023 (Extortion)"),
        (["cheat", "cheated", "cheating", "scam", "fraud", "defraud", "duped"], "Section 318(4) BNS 2023 (Cheating)"),
        (["impersonat", "fake executive", "personation"], "Section 319(2) BNS 2023 (Cheating by Personation)"),
        (["breach of trust", "entrusted", "misappropriat", "embezzle"], "Section 316(2) BNS 2023 (Criminal Breach of Trust)"),
        (["threat", "threaten", "intimidat", "dire consequences"], "Section 351 BNS 2023 (Criminal Intimidation)"),
        (["assault", "beat", "beaten", "slap", "hit", "punch", "attack", "hurt", "injur"], "Section 115(2) BNS 2023 (Voluntarily Causing Hurt)"),
        (["stalk", "stalking", "followed"], "Section 78 BNS 2023 (Stalking)"),
        (["trespass", "broke into", "house-trespass"], "Section 329 BNS 2023 (Criminal and House-Trespass)"),
        (["defam", "slander", "libel"], "Section 356 BNS 2023 (Defamation)"),
        (["damage", "vandal", "mischief"], "Section 324 BNS 2023 (Mischief)"),
    ]
    for keywords, citation in rules:
        if any(k in t for k in keywords):
            matched.append(citation)

    if not matched:
        matched.append("applicable cognizable provisions of the Bharatiya Nyaya Sanhita (BNS), 2023")
    return ", ".join(matched)


def infer_notice_statutory_sections(issue_type: str, facts_text: str, explicit_section: str = "") -> str:
    """Dynamically determine statutory provisions for a civil/commercial/rent Legal Notice."""
    if explicit_section and explicit_section.strip():
        return explicit_section.strip()
    combined = f"{issue_type or ''} {facts_text or ''}".lower()
    if any(k in combined for k in ("rent", "tenant", "landlord", "lease", "evict", "deposit", "premises", "flat", "vacat")):
        return (
            "Section 106 and Section 108 of the Transfer of Property Act, 1882 read with "
            "Section 73 of the Indian Contract Act, 1872"
        )
    if any(k in combined for k in ("cheque", "dishonour", "dishonor", "bounced", "bounce")):
        return (
            "Section 138 of the Negotiable Instruments Act, 1881 read with "
            "Section 318(4) of the Bharatiya Nyaya Sanhita (BNS), 2023"
        )
    if any(k in combined for k in ("defective", "consumer", "warranty", "replacement", "seller", "ecommerce", "service deficiency")):
        return (
            "Sections 2(11), 2(47), and 35 of the Consumer Protection Act, 2019 read with "
            "the Sale of Goods Act, 1930"
        )
    if any(k in combined for k in ("salary", "wage", "fnf", "gratuity", "employer", "employment")):
        return (
            "Sections 17 and 18 of the Code on Wages, 2019 read with "
            "Section 15 of the Payment of Wages Act, 1936"
        )
    return (
        "Sections 73 and 74 of the Indian Contract Act, 1872 and Order XXXVII of the "
        "Code of Civil Procedure, 1908"
    )


# ── XML / HTML SANITIZATION HELPERS ─────────────────────────────────────────
def _rl_safe(text: Any) -> str:
    """Escape raw '&' and unsupported '<'/'>' for ReportLab Paragraph XML parser
    while preserving <b>, </b>, <i>, </i>, <u>, </u>, and <br/> tags."""
    s = str(text or "")
    # Normalize <br> and <strong>
    s = re.sub(r"<\s*br\s*/?\s*>", "<br/>", s, flags=re.I)
    s = re.sub(r"<\s*strong\s*>", "<b>", s, flags=re.I)
    s = re.sub(r"<\s*/\s*strong\s*>", "</b>", s, flags=re.I)
    s = re.sub(r"<\s*em\s*>", "<i>", s, flags=re.I)
    s = re.sub(r"<\s*/\s*em\s*>", "</i>", s, flags=re.I)
    # Escape bare ampersands
    s = re.sub(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", s)
    # Protect allowed ReportLab tags temporarily
    allowed = ("<b>", "</b>", "<i>", "</i>", "<u>", "</u>", "<br/>")
    placeholders = {}
    for idx, tag in enumerate(allowed):
        tok = f"__RLTAG_{idx}__"
        placeholders[tok] = tag
        s = s.replace(tag, tok)
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    for tok, tag in placeholders.items():
        s = s.replace(tok, tag)
    return s


# ── REPORTLAB STYLING HELPERS ───────────────────────────────────────────────
def _get_pdf_styles():
    base = getSampleStyleSheet()

    primary_color = colors.HexColor("#1a365d")
    secondary_color = colors.HexColor("#2b6cb0")
    dark_neutral = colors.HexColor("#1a202c")
    light_neutral = colors.HexColor("#4a5568")

    styles = {
        "title": ParagraphStyle(
            "DocTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=primary_color,
            alignment=1,  # Center
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "DocSubtitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12.5,
            textColor=secondary_color,
            alignment=1,
            spaceAfter=10,
        ),
        "heading": ParagraphStyle(
            "DocHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=primary_color,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "DocBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=dark_neutral,
            spaceAfter=5,
        ),
        "body_bold": ParagraphStyle(
            "DocBodyBold",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=13.5,
            textColor=dark_neutral,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "DocSmall",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=light_neutral,
            spaceAfter=4,
        ),
        "box_text": ParagraphStyle(
            "DocBoxText",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=dark_neutral,
        ),
    }
    return styles


# ── DOCX STYLING HELPERS ───────────────────────────────────────────────────
def _init_docx_doc(title: str, subtitle: str = "") -> docx.Document:
    doc = docx.Document()

    # Set standard margins (0.75 inch)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Document Header Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(2)
    run_title = p_title.add_run(title.upper())
    run_title.font.name = "Arial"
    run_title.font.size = Pt(14)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(26, 54, 93)

    if subtitle:
        p_sub = doc.add_paragraph()
        p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_sub.paragraph_format.space_after = Pt(10)
        run_sub = p_sub.add_run(subtitle)
        run_sub.font.name = "Arial"
        run_sub.font.size = Pt(9.5)
        run_sub.font.bold = True
        run_sub.font.color.rgb = RGBColor(43, 108, 176)

    return doc


def _add_docx_para(
    doc: docx.Document,
    text: str,
    bold: bool = False,
    italic: bool = False,
    space_after: int = 5,
    font_size: float = 10.0,
):
    """Add a paragraph to a python-docx Document, converting inline <b>...</b> and <br/>
    into native bold runs and line breaks, and stripping any stray HTML tags."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15

    normalized = str(text or "")
    normalized = re.sub(r"<\s*br\s*/?\s*>", "\n", normalized, flags=re.I)
    normalized = re.sub(r"<\s*strong\s*>", "<b>", normalized, flags=re.I)
    normalized = re.sub(r"<\s*/\s*strong\s*>", "</b>", normalized, flags=re.I)
    # Remove any other HTML tags besides <b> and </b>
    normalized = re.sub(r"<(?!/?b\b)[^>]+>", "", normalized)

    tokens = re.split(r"(<b>|</b>)", normalized)
    in_bold = bold
    for tok in tokens:
        if tok == "<b>":
            in_bold = True
            continue
        if tok == "</b>":
            in_bold = bold
            continue
        if not tok:
            continue
        run = p.add_run(tok)
        run.font.name = "Arial"
        run.font.size = Pt(font_size)
        run.bold = in_bold
        run.italic = italic
    return p


def _add_docx_kv_table(doc: docx.Document, rows: List[Tuple[str, str]]) -> None:
    """Add a clean 2-column key-value table in DOCX matching the PDF table."""
    if not rows:
        return
    table = doc.add_table(rows=len(rows), cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (k, v) in enumerate(rows):
        c0 = table.cell(idx, 0)
        c1 = table.cell(idx, 1)
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(2)
        r0 = p0.add_run(re.sub(r"<[^>]+>", "", str(k)))
        r0.font.name = "Arial"
        r0.font.size = Pt(9.5)
        r0.bold = True

        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(2)
        r1 = p1.add_run(re.sub(r"<[^>]+>", "", str(v)))
        r1.font.name = "Arial"
        r1.font.size = Pt(9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def _add_docx_counterfoil(doc: docx.Document, title: str, lines: List[str]) -> None:
    """Add a bordered receiving counterfoil box at the bottom of the DOCX."""
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    cell = table.cell(0, 0)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r_title = p.add_run(title + "\n")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(8.5)
    r_title.bold = True
    for line in lines:
        r_line = p.add_run(line + "\n")
        r_line.font.name = "Arial"
        r_line.font.size = Pt(8.5)


# ── 1. LEGAL NOTICE (PDF & DOCX) ───────────────────────────────────────────
def build_legal_notice(data: Dict[str, Any], base_filename: str) -> Dict[str, str]:
    data = normalize_tool_params("legal_notice", data)
    out_dir = _ensure_output_dir()
    pdf_path = str(out_dir / f"{base_filename}.pdf")
    docx_path = str(out_dir / f"{base_filename}.docx")

    sender = data.get("sender_name") or "Aggrieved Party"
    sender_addr = data.get("sender_address") or "Address Not Provided"
    sender_email = data.get("sender_email") or ""
    sender_phone = data.get("sender_phone") or ""

    recipient = data.get("recipient_name") or "Recipient"
    recipient_addr = data.get("recipient_address") or "Address Not Provided"
    recipient_email = data.get("recipient_email") or ""

    issue_type = data.get("issue_type") or "Legal Dispute & Recovery"
    facts = data.get("issue_description") or "Dispute facts"
    sections = infer_notice_statutory_sections(issue_type, facts, data.get("applicable_section", ""))
    amount = data.get("demand_amount") or ""
    date_str = datetime.now().strftime("%d-%m-%Y")
    deadline_date = (datetime.now() + timedelta(days=15)).strftime("%d-%m-%Y")

    # 1. Build PDF
    pdf_doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = _get_pdf_styles()
    story = []

    story.append(Paragraph("FORMAL LEGAL NOTICE", styles["title"]))
    story.append(
        Paragraph(
            "REGISTERED SPEED POST WITH ACKNOWLEDGEMENT DUE (RPAD) / E-MAIL",
            styles["subtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a365d"), spaceAfter=12))

    parties_data = [
        [
            Paragraph(
                _rl_safe(
                    f"<b>FROM:</b><br/><b>{sender}</b><br/>{sender_addr}"
                    + (f"<br/>Email: {sender_email}" if sender_email else "")
                    + (f"<br/>Phone: {sender_phone}" if sender_phone else "")
                ),
                styles["box_text"],
            ),
            Paragraph(
                _rl_safe(
                    f"<b>TO:</b><br/><b>{recipient}</b><br/>{recipient_addr}"
                    + (f"<br/>Email: {recipient_email}" if recipient_email else "")
                ),
                styles["box_text"],
            ),
        ],
        [
            Paragraph(_rl_safe(f"<b>DATE:</b> {date_str}"), styles["box_text"]),
            Paragraph(_rl_safe(f"<b>PEREMPTORY COMPLIANCE PERIOD:</b> 15 Days (By {deadline_date})"), styles["box_text"]),
        ],
    ]
    t_parties = Table(parties_data, colWidths=[3.4 * inch, 3.4 * inch])
    t_parties.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_parties)
    story.append(Spacer(1, 12))

    subj_text = f"<b>SUBJECT: LEGAL NOTICE FOR {issue_type.upper()} UNDER {sections.upper()}</b>" + (
        f" <b>DEMANDING RS. {amount}/-</b>" if amount else ""
    )
    story.append(Paragraph(_rl_safe(subj_text), styles["heading"]))
    story.append(Spacer(1, 6))

    custom_paras = data.get("custom_body_paras")
    if custom_paras and isinstance(custom_paras, list) and len(custom_paras) > 0:
        body_paras = [
            f"Dear {recipient},",
            f"Under instructions and on behalf of my client/myself, <b>{sender}</b>, I hereby serve upon you this formal statutory legal notice under <b>{sections}</b>:",
        ] + [str(p) for p in custom_paras]
    else:
        body_paras = [
            f"Dear {recipient},",
            f"Under instructions and on behalf of my client/myself, <b>{sender}</b>, I hereby serve upon you this formal legal demand notice regarding the following facts and statutory breaches:",
            f"<b>1. FACTS AND CHRONOLOGY OF OCCURRENCE:</b><br/>{facts}",
            f"<b>2. STATUTORY VIOLATION & APPLICABLE LAW:</b><br/>Your aforesaid acts, omissions, and failure to discharge your lawful obligations constitute an actionable breach and default under <b>{sections}</b>.",
        ]
        if amount:
            body_paras.append(
                f"<b>3. LIQUIDATED MONETARY DEMAND:</b><br/>You are hereby called upon to pay/refund the legitimate liquidated sum of <b>Rs. {amount}/-</b> together with interest @ 18% per annum directly to the sender within <b>15 (fifteen) days</b> from the receipt of this notice."
            )
        else:
            body_paras.append(
                "<b>3. RELIEF DEMANDED:</b><br/>You are hereby called upon to immediately redress the said grievance and comply with your legal obligations within <b>15 (fifteen) days</b> from the receipt of this notice."
            )

        body_paras.append(
            f"<b>4. CONSEQUENCE OF DEFAULT (DEADLINE: {deadline_date}):</b><br/>Please take peremptory notice that if you fail, neglect, or refuse to comply with the requisitions of this notice on or before <b>{deadline_date}</b>, appropriate civil, consumer, and criminal proceedings shall be instituted against you before the Court of competent jurisdiction at your sole risk, cost, and consequence."
        )

    for p in body_paras:
        story.append(Paragraph(_rl_safe(p), styles["body"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 14))
    sign_data = [
        [
            Paragraph(_rl_safe(f"<b>PLACE:</b> ___________________<br/><b>DATE:</b> {date_str}"), styles["box_text"]),
            Paragraph(
                _rl_safe(
                    f"<b>YOURS FAITHFULLY,</b><br/><br/>___________________________<br/><b>{sender}</b><br/>(Sender / Authorized Signatory)"
                ),
                styles["box_text"],
            ),
        ]
    ]
    t_sign = Table(sign_data, colWidths=[3.4 * inch, 3.4 * inch])
    t_sign.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t_sign)
    pdf_doc.build(story)

    # 2. Build DOCX with full table parity
    doc = _init_docx_doc("FORMAL LEGAL NOTICE", "REGISTERED SPEED POST WITH ACKNOWLEDGEMENT DUE (RPAD) / E-MAIL")
    _add_docx_kv_table(
        doc,
        [
            ("FROM (SENDER):", f"{sender}, {sender_addr}" + (f" | Email: {sender_email}" if sender_email else "")),
            ("TO (RECIPIENT):", f"{recipient}, {recipient_addr}" + (f" | Email: {recipient_email}" if recipient_email else "")),
            ("DATE OF NOTICE:", date_str),
            ("APPLICABLE STATUTE:", sections),
            ("COMPLIANCE WINDOW:", f"15 Days (On or before {deadline_date})"),
        ],
    )
    _add_docx_para(
        doc,
        f"SUBJECT: LEGAL NOTICE FOR {issue_type.upper()} UNDER {sections.upper()}"
        + (f" (DEMAND: RS. {amount}/-)" if amount else ""),
        bold=True,
        space_after=8,
    )
    for p in body_paras:
        _add_docx_para(doc, p)
    _add_docx_para(
        doc,
        f"\nPLACE: ___________________\nDATE: {date_str}\n\nYOURS FAITHFULLY,\n\n___________________________\n{sender}\n(Sender / Authorized Signatory)",
        bold=True,
        space_after=12,
    )
    doc.save(docx_path)

    return {"pdf_path": pdf_path, "docx_path": docx_path}


# ── 2. RTI APPLICATION (FORM A, SECTION 6 RTI ACT 2005) ────────────────────
def build_rti_application(data: Dict[str, Any], base_filename: str) -> Dict[str, str]:
    out_dir = _ensure_output_dir()
    pdf_path = str(out_dir / f"{base_filename}.pdf")
    docx_path = str(out_dir / f"{base_filename}.docx")

    applicant = data.get("applicant_name") or "Applicant Name"
    applicant_addr = data.get("applicant_address") or "Applicant Address Not Provided"
    applicant_phone = data.get("applicant_phone") or ""
    applicant_email = data.get("applicant_email") or ""

    pio_office = data.get("pio_office") or "The Public Authority / Department"
    pio_addr = data.get("pio_address") or "Office of the Public Information Officer"
    info_sought = data.get("information_sought") or "Details of public records"
    fee_mode = data.get("application_fee") or "Enclosed Rs. 10/- (Rupees Ten only) via Indian Postal Order (IPO) / Demand Draft / Court Fee Stamp."
    date_str = datetime.now().strftime("%d-%m-%Y")

    # 1. Build PDF
    pdf_doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = _get_pdf_styles()
    story = []

    story.append(Paragraph("FORM - A", styles["title"]))
    story.append(
        Paragraph(
            "APPLICATION FOR SEEKING INFORMATION UNDER SECTIONS 6(1) & 6(3) OF THE RTI ACT, 2005",
            styles["subtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a365d"), spaceAfter=10))

    to_text = f"<b>TO:</b><br/>The Public Information Officer (PIO),<br/><b>{pio_office}</b>,<br/>{pio_addr}."
    t_to = Table([[Paragraph(_rl_safe(to_text), styles["body"])]], colWidths=[6.8 * inch])
    t_to.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_to)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>1. PARTICULARS OF THE APPLICANT:</b>", styles["heading"]))
    app_details = [
        [Paragraph("<b>(a) Full Name:</b>", styles["box_text"]), Paragraph(_rl_safe(applicant), styles["box_text"])],
        [Paragraph("<b>(b) Permanent Address:</b>", styles["box_text"]), Paragraph(_rl_safe(applicant_addr), styles["box_text"])],
        [Paragraph("<b>(c) Contact Phone / Mobile:</b>", styles["box_text"]), Paragraph(_rl_safe(applicant_phone or "Not Provided"), styles["box_text"])],
        [Paragraph("<b>(d) E-mail Address:</b>", styles["box_text"]), Paragraph(_rl_safe(applicant_email or "Not Provided"), styles["box_text"])],
        [Paragraph("<b>(e) Citizenship Status:</b>", styles["box_text"]), Paragraph("Citizen of India (Under Section 3, RTI Act 2005)", styles["box_text"])],
    ]
    t_app = Table(app_details, colWidths=[2.3 * inch, 4.5 * inch])
    t_app.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_app)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>2. SPECIFIC INFORMATION SOUGHT (SECTION 6(1) & 6(3)):</b>", styles["heading"]))
    info_box = [
        [
            Paragraph(
                _rl_safe(
                    f"The applicant requests certified copies / inspection of the following public records under Section 6(1) of the Right to Information Act, 2005 (and if any part relates to another public authority, requests transfer within 5 days under Section 6(3)):<br/><br/><b>{info_sought}</b>"
                ),
                styles["body"],
            )
        ]
    ]
    t_info = Table(info_box, colWidths=[6.8 * inch])
    t_info.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_info)
    story.append(Spacer(1, 8))

    fee_data = [
        [
            Paragraph(_rl_safe(f"<b>3. APPLICATION FEE:</b><br/>{fee_mode}"), styles["box_text"]),
            Paragraph(
                "<b>4. PREFERRED MODE OF DELIVERY:</b><br/>[X] Registered / Speed Post<br/>[  ] By E-mail / Electronic Copy",
                styles["box_text"],
            ),
        ]
    ]
    t_fee = Table(fee_data, colWidths=[3.4 * inch, 3.4 * inch])
    t_fee.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t_fee)
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "<b>DECLARATION:</b> I hereby declare that I am a citizen of India and that the information sought falls within the ambit of Section 6 of the Right to Information Act, 2005 and does not fall within the exemptions under Sections 8 or 9 of the Act.",
            styles["small"],
        )
    )
    story.append(Spacer(1, 12))

    sign_data = [
        [
            Paragraph(_rl_safe(f"<b>PLACE:</b> ___________________<br/><b>DATE:</b> {date_str}"), styles["box_text"]),
            Paragraph(
                _rl_safe(f"___________________________<br/><b>SIGNATURE OF APPLICANT</b><br/>({applicant})"),
                styles["box_text"],
            ),
        ]
    ]
    t_sign = Table(sign_data, colWidths=[3.4 * inch, 3.4 * inch])
    story.append(t_sign)

    story.append(Spacer(1, 15))
    ack_data = [
        [
            Paragraph(
                _rl_safe(
                    f"<b>FOR OFFICE USE ONLY (ACKNOWLEDGEMENT SLIP - RTI ACT 2005)</b><br/>"
                    f"Received RTI Application from {applicant} on Date: ____________ Ref/Diary No: ____________________<br/>"
                    f"Designation / Seal of Receiving Officer: ________________________________________________"
                ),
                styles["small"],
            )
        ]
    ]
    t_ack = Table(ack_data, colWidths=[6.8 * inch])
    t_ack.setStyle(
        TableStyle(
            [
                ("LINEBEFORE", (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t_ack)
    pdf_doc.build(story)

    # 2. Build DOCX with full table and counterfoil parity
    doc = _init_docx_doc("FORM - A", "APPLICATION FOR SEEKING INFORMATION UNDER SECTIONS 6(1) & 6(3) OF THE RTI ACT, 2005")
    _add_docx_para(doc, f"DATE: {date_str}", bold=True)
    _add_docx_para(doc, f"TO:\nThe Public Information Officer (PIO),\n{pio_office},\n{pio_addr}.", bold=True)
    _add_docx_kv_table(
        doc,
        [
            ("(a) Full Name of Applicant:", applicant),
            ("(b) Permanent Address:", applicant_addr),
            ("(c) Contact Phone / Mobile:", applicant_phone or "Not Provided"),
            ("(d) E-mail Address:", applicant_email or "Not Provided"),
            ("(e) Citizenship:", "Citizen of India (Section 3, RTI Act 2005)"),
        ],
    )
    _add_docx_para(doc, f"2. SPECIFIC INFORMATION SOUGHT (SECTION 6(1) & 6(3) RTI ACT 2005):\n{info_sought}", bold=True)
    _add_docx_para(doc, f"3. APPLICATION FEE: {fee_mode}")
    _add_docx_para(
        doc,
        "DECLARATION: I am a citizen of India and the requested information is not exempt under Section 8 or 9 of the Right to Information Act, 2005.",
    )
    _add_docx_para(doc, f"\nPLACE: ___________________   DATE: {date_str}\nSIGNATURE OF APPLICANT: ___________________________ ({applicant})", bold=True)
    _add_docx_counterfoil(
        doc,
        "FOR OFFICE USE ONLY (ACKNOWLEDGEMENT SLIP - RTI ACT 2005)",
        [
            f"Received RTI Application from {applicant} on Date: ____________ Ref/Diary No: ____________________",
            "Designation / Seal of Receiving Officer: ________________________________________________",
        ],
    )
    doc.save(docx_path)

    return {"pdf_path": pdf_path, "docx_path": docx_path}


# ── 3. CYBER FRAUD & UPI SCAM EMERGENCY LETTER (PDF & DOCX) ────────────────
def build_cyber_fraud_letter(data: Dict[str, Any], base_filename: str) -> Dict[str, str]:
    data = normalize_tool_params("cyber_fraud", data)
    out_dir = _ensure_output_dir()
    pdf_path = str(out_dir / f"{base_filename}.pdf")
    docx_path = str(out_dir / f"{base_filename}.docx")

    victim_name = data.get("victim_name") or "Aggrieved Account Holder"
    victim_phone = data.get("victim_phone") or "Mobile Number"
    victim_email = data.get("victim_email") or ""
    victim_addr = data.get("victim_address") or "Address Not Provided"

    bank_name = data.get("victim_bank_name") or "Bank Branch Manager"
    branch_raw = (data.get("victim_bank_branch") or "Local Branch").strip()
    branch_display = branch_raw if "branch" in branch_raw.lower() else f"{branch_raw} Branch"
    account_no = data.get("account_number") or "Account Number"

    amount = data.get("amount_lost") or "0"
    txn_id = data.get("transaction_id_utr") or "UTR/Txn Reference"
    beneficiary = data.get("beneficiary_upi_vpa_or_account") or "Fraudulent Beneficiary Details"
    txn_time = data.get("transaction_date_time") or datetime.now().strftime("%d-%m-%Y %H:%M")
    fraud_type = data.get("fraud_type") or "UPI / Cyber Financial Fraud"
    facts = data.get("facts_summary") or "Unauthorized fraudulent electronic banking transaction"
    date_str = datetime.now().strftime("%d-%m-%Y")

    # 1. Build PDF
    pdf_doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = _get_pdf_styles()
    story = []

    story.append(Paragraph("EMERGENCY CYBER FRAUD NOTICE & ACCOUNT FREEZE REQUISITION", styles["title"]))
    story.append(
        Paragraph(
            "UNDER RBI CIRCULAR DBR.No.Leg.BC.78/09.07.005/2017-18, SEC 43/66D IT ACT 2000 & SEC 318(4) BNS 2023",
            styles["subtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dc2626"), spaceAfter=10))

    to_text = (
        f"<b>TO:</b><br/>The Branch Manager / Nodal Cyber Nodal Officer,<br/>"
        f"<b>{bank_name}</b>, {branch_display}.<br/>"
        f"<b>COPY TO:</b> Nodal Cyber Crime Cell & National Cyber Crime Reporting Portal (Helpline 1930)."
    )
    t_to = Table([[Paragraph(_rl_safe(to_text), styles["body"])]], colWidths=[6.8 * inch])
    t_to.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#f87171")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_to)
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            _rl_safe(
                f"<b>SUBJECT: URGENT COMPLAINT REGARDING UNAUTHORIZED FRAUDULENT DEBIT OF RS. {amount}/- AND REQUISITION FOR IMMEDIATE LIEN / BENEFICIARY ACCOUNT FREEZE</b>"
            ),
            styles["heading"],
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>1. FRAUDULENT TRANSACTION PARTICULARS:</b>", styles["heading"]))
    table_data = [
        [Paragraph("<b>Victim / Account Holder:</b>", styles["box_text"]), Paragraph(_rl_safe(f"{victim_name} (Ph: {victim_phone})"), styles["box_text"])],
        [Paragraph("<b>Debited Bank Account:</b>", styles["box_text"]), Paragraph(_rl_safe(f"{account_no} ({bank_name}, {branch_display})"), styles["box_text"])],
        [Paragraph("<b>Total Defrauded Amount:</b>", styles["box_text"]), Paragraph(_rl_safe(f"<b>Rs. {amount}/-</b>"), styles["box_text"])],
        [Paragraph("<b>Transaction ID / UTR Reference:</b>", styles["box_text"]), Paragraph(_rl_safe(f"<b>{txn_id}</b>"), styles["box_text"])],
        [Paragraph("<b>Date & Timestamp:</b>", styles["box_text"]), Paragraph(_rl_safe(txn_time), styles["box_text"])],
        [Paragraph("<b>Beneficiary VPA / Account:</b>", styles["box_text"]), Paragraph(_rl_safe(f"<b>{beneficiary}</b>"), styles["box_text"])],
        [Paragraph("<b>Modus Operandi / Fraud Type:</b>", styles["box_text"]), Paragraph(_rl_safe(fraud_type), styles["box_text"])],
    ]
    t_details = Table(table_data, colWidths=[2.5 * inch, 4.3 * inch])
    t_details.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#fee2e2")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_details)
    story.append(Spacer(1, 8))

    custom_paras = data.get("custom_body_paras")
    if custom_paras and isinstance(custom_paras, list) and len(custom_paras) > 0:
        paras = [str(p) for p in custom_paras]
    else:
        paras = [
            f"<b>2. CHRONOLOGICAL SUMMARY OF THE OCCURRENCE:</b><br/>{facts}",
            "<b>3. STATUTORY VIOLATION & RBI ZERO LIABILITY MANDATE:</b><br/>"
            "The aforesaid unauthorized debit constitutes a cognizable cyber-financial offence under <b>Section 43 and Section 66D of the Information Technology Act, 2000</b> read with <b>Section 318(4) of the Bharatiya Nyaya Sanhita (BNS), 2023</b>. Furthermore, under <b>Reserve Bank of India (RBI) Circular No. DBR.No.Leg.BC.78/09.07.005/2017-18</b> (<i>Customer Protection – Limiting Liability of Customers in Unauthorised Electronic Banking Transactions</i>), a customer enjoys <b>Zero Liability</b> when unauthorized electronic banking transactions are reported within 3 working days, and the bank is statutorily obligated to flag the transaction, raise an immediate NPCI chargeback, and coordinate a lien on the beneficiary account.",
            f"<b>4. URGENT PRAYER / REQUISITION TO BANK:</b><br/>"
            f"a) Immediately mark a hold/lien and freeze the defrauded sum of <b>Rs. {amount}/-</b> in the beneficiary account / VPA: <b>{beneficiary}</b> (Ref: <b>{txn_id}</b>).<br/>"
            f"b) Initiate an immediate NPCI UPI/IMPS/NEFT chargeback and process reversal / shadow credit in accordance with RBI Zero Liability guidelines.<br/>"
            f"c) Preserve and furnish complete electronic transaction logs, IP traces, and beneficiary KYC particulars certified under <b>Section 63 of the Bharatiya Sakshya Adhiniyam (BSA), 2023</b> for police investigation.",
        ]
    for p in paras:
        story.append(Paragraph(_rl_safe(p), styles["body"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 10))
    sign_data = [
        [
            Paragraph(_rl_safe(f"<b>PLACE:</b> ___________________<br/><b>DATE:</b> {date_str}"), styles["box_text"]),
            Paragraph(
                _rl_safe(f"___________________________<br/><b>SIGNATURE OF ACCOUNT HOLDER</b><br/>({victim_name})"),
                styles["box_text"],
            ),
        ]
    ]
    t_sign = Table(sign_data, colWidths=[3.4 * inch, 3.4 * inch])
    story.append(t_sign)

    story.append(Spacer(1, 10))
    ack_data = [
        [
            Paragraph(
                _rl_safe(
                    f"<b>BANK BRANCH RECEIVING ACKNOWLEDGEMENT (MANDATORY COUNTERFOIL)</b><br/>"
                    f"Received representation for cyber fraud account freeze regarding UTR: {txn_id} for Rs. {amount}/- (Beneficiary: {beneficiary})<br/>"
                    f"Bank & Branch: {bank_name}, {branch_display} | Date & Time Received: ____________ | Stamp & Sign: ________________"
                ),
                styles["small"],
            )
        ]
    ]
    t_ack = Table(ack_data, colWidths=[6.8 * inch])
    t_ack.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#ef4444")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff1f2")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t_ack)
    pdf_doc.build(story)

    # 2. Build DOCX with full table and counterfoil parity
    doc = _init_docx_doc(
        "EMERGENCY CYBER FRAUD NOTICE & ACCOUNT FREEZE REQUISITION",
        "UNDER RBI CIRCULAR DBR.No.Leg.BC.78/09.07.005/2017-18, SEC 43/66D IT ACT 2000 & SEC 318(4) BNS 2023",
    )
    _add_docx_para(doc, f"DATE: {date_str}", bold=True)
    _add_docx_para(
        doc,
        f"TO:\nThe Branch Manager / Nodal Cyber Officer,\n{bank_name}, {branch_display}.\nCOPY TO: National Cyber Crime Reporting Portal (1930).",
        bold=True,
    )
    _add_docx_para(
        doc,
        f"SUBJECT: URGENT COMPLAINT REGARDING UNAUTHORIZED FRAUDULENT DEBIT OF RS. {amount}/- AND REQUISITION FOR IMMEDIATE LIEN / BENEFICIARY ACCOUNT FREEZE",
        bold=True,
        space_after=8,
    )
    _add_docx_kv_table(
        doc,
        [
            ("Victim / Account Holder:", f"{victim_name} (Ph: {victim_phone})"),
            ("Debited Bank Account:", f"{account_no} ({bank_name}, {branch_display})"),
            ("Total Defrauded Amount:", f"Rs. {amount}/-"),
            ("Transaction ID / UTR:", txn_id),
            ("Date & Timestamp:", txn_time),
            ("Beneficiary VPA / Account:", beneficiary),
            ("Modus Operandi:", fraud_type),
        ],
    )
    for p in paras:
        _add_docx_para(doc, p)
    _add_docx_para(doc, f"\nPLACE: ___________________   DATE: {date_str}\nSIGNATURE OF ACCOUNT HOLDER: ___________________________ ({victim_name})", bold=True)
    _add_docx_counterfoil(
        doc,
        "BANK BRANCH RECEIVING ACKNOWLEDGEMENT (MANDATORY COUNTERFOIL)",
        [
            f"Received representation for cyber fraud account freeze regarding UTR: {txn_id} for Rs. {amount}/- (Beneficiary: {beneficiary})",
            f"Bank & Branch: {bank_name}, {branch_display} | Date & Time Received: ____________ | Stamp & Sign: ________________",
        ],
    )
    doc.save(docx_path)

    return {"pdf_path": pdf_path, "docx_path": docx_path}


# ── 4. POLICE COMPLAINT TO SHO (PDF & DOCX) ────────────────────────────────
def build_police_complaint(data: Dict[str, Any], base_filename: str) -> Dict[str, str]:
    data = normalize_tool_params("police_complaint", data)
    out_dir = _ensure_output_dir()
    pdf_path = str(out_dir / f"{base_filename}.pdf")
    docx_path = str(out_dir / f"{base_filename}.docx")

    complainant = data.get("complainant_name") or "Complainant Name"
    complainant_addr = data.get("complainant_address") or "Address Not Provided"
    complainant_phone = data.get("complainant_phone") or "Mobile Number"

    station = data.get("police_station_name") or "Local Police Station"
    incident_time = data.get("incident_date_time") or datetime.now().strftime("%d-%m-%Y")
    incident_loc = data.get("incident_location") or "Location of Incident"
    accused = data.get("accused_details") or "Unknown Persons / Opposite Party"
    facts = data.get("incident_facts") or "Description of the criminal incident"
    evidence = data.get("evidence_available") or "Photographs, CCTV footage, mobile phone records (with Section 63 BSA 2023 certificate)"
    bns_sections = data.get("applicable_section") or infer_bns_offence_sections(facts)
    date_str = datetime.now().strftime("%d-%m-%Y")

    # 1. Build PDF
    pdf_doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = _get_pdf_styles()
    story = []

    story.append(Paragraph("FORMAL WRITTEN POLICE COMPLAINT", styles["title"]))
    story.append(
        Paragraph(
            "SUBMITTED UNDER SECTION 173 OF BHARATIYA NAGARIK SURAKSHA SANHITA (BNSS 2023)",
            styles["subtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a365d"), spaceAfter=10))

    to_text = f"<b>TO:</b><br/>The Station House Officer (SHO),<br/><b>{station}</b>."
    t_to = Table([[Paragraph(_rl_safe(to_text), styles["body"])]], colWidths=[6.8 * inch])
    t_to.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_to)
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            _rl_safe(
                f"<b>SUBJECT: WRITTEN COMPLAINT UNDER SECTION 173 BNSS 2023 FOR REGISTRATION OF FIR UNDER {bns_sections.upper()}</b>"
            ),
            styles["heading"],
        )
    )
    story.append(Spacer(1, 6))

    table_data = [
        [Paragraph("<b>Complainant Name & Contact:</b>", styles["box_text"]), Paragraph(_rl_safe(f"{complainant} (Ph: {complainant_phone})"), styles["box_text"])],
        [Paragraph("<b>Complainant Address:</b>", styles["box_text"]), Paragraph(_rl_safe(complainant_addr), styles["box_text"])],
        [Paragraph("<b>Date & Time of Occurrence:</b>", styles["box_text"]), Paragraph(_rl_safe(incident_time), styles["box_text"])],
        [Paragraph("<b>Place of Occurrence:</b>", styles["box_text"]), Paragraph(_rl_safe(incident_loc), styles["box_text"])],
        [Paragraph("<b>Name & Details of Accused:</b>", styles["box_text"]), Paragraph(_rl_safe(f"<b>{accused}</b>"), styles["box_text"])],
        [Paragraph("<b>Operative Statutory Sections:</b>", styles["box_text"]), Paragraph(_rl_safe(f"<b>{bns_sections} read with Section 173 BNSS 2023</b>"), styles["box_text"])],
    ]
    t_info = Table(table_data, colWidths=[2.5 * inch, 4.3 * inch])
    t_info.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_info)
    story.append(Spacer(1, 8))

    custom_paras = data.get("custom_body_paras")
    if custom_paras and isinstance(custom_paras, list) and len(custom_paras) > 0:
        paras = ["Respected Sir/Madam,"] + [str(p) for p in custom_paras]
    else:
        paras = [
            "Respected Sir/Madam,",
            "I, the undersigned complainant, most respectfully bring the following cognizable criminal occurrence to your immediate notice for prompt statutory investigation under the Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023:",
            f"<b>1. STATEMENT OF FACTS & CHRONOLOGY:</b><br/>{facts}",
            f"<b>2. STATUTORY OFFENCES COMMITTED (BNS 2023):</b><br/>The aforesaid unlawful acts of the accused clearly disclose the commission of serious cognizable offences punishable under <b>{bns_sections}</b>.",
            f"<b>3. PARTICULARS OF EVIDENCE ENCLOSED (BSA 2023):</b><br/>The following corroborative material and electronic records (admissible under <b>Section 63 of the Bharatiya Sakshya Adhiniyam, 2023</b>) are available in support of this complaint: {evidence}.",
            f"<b>4. PRAYER:</b><br/>In light of the aforesaid facts disclosing commission of cognizable offences under <b>{bns_sections}</b>, it is most respectfully prayed that:<br/>"
            "a) This written information be registered forthwith as a First Information Report (FIR) under <b>Section 173 of the Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023</b> and a free copy of the FIR be furnished to the complainant.<br/>"
            "b) Immediate investigation be initiated, nearby CCTV footage be seized and preserved, stolen/defrauded property be recovered, and the culprits be apprehended in accordance with law.<br/>"
            "c) Adequate protection to the life, liberty, and property of the complainant and family be ensured.",
        ]
    for p in paras:
        story.append(Paragraph(_rl_safe(p), styles["body"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 10))
    sign_data = [
        [
            Paragraph(_rl_safe(f"<b>PLACE:</b> ___________________<br/><b>DATE:</b> {date_str}"), styles["box_text"]),
            Paragraph(
                _rl_safe(f"___________________________<br/><b>SIGNATURE OF COMPLAINANT</b><br/>({complainant})"),
                styles["box_text"],
            ),
        ]
    ]
    t_sign = Table(sign_data, colWidths=[3.4 * inch, 3.4 * inch])
    story.append(t_sign)

    story.append(Spacer(1, 12))
    ack_data = [
        [
            Paragraph(
                _rl_safe(
                    f"<b>POLICE STATION RECEIVING COUNTERFOIL (MANDATORY ACKNOWLEDGEMENT COPY - SEC 173 BNSS 2023)</b><br/>"
                    f"Received Written Complaint from: {complainant} regarding incident at {incident_loc} ({bns_sections})<br/>"
                    f"Police Station: {station} | Date & Time Received: ____________ | GD/DD/FIR Entry No: ________________<br/>"
                    f"Receiving Officer Name & Belt No: ________________________________ | Police Station Seal: ________________"
                ),
                styles["small"],
            )
        ]
    ]
    t_ack = Table(ack_data, colWidths=[6.8 * inch])
    t_ack.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#1e293b")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t_ack)
    pdf_doc.build(story)

    # 2. Build DOCX with full table and counterfoil parity
    doc = _init_docx_doc(
        "FORMAL WRITTEN POLICE COMPLAINT",
        "SUBMITTED UNDER SECTION 173 OF BHARATIYA NAGARIK SURAKSHA SANHITA (BNSS 2023)",
    )
    _add_docx_para(doc, f"DATE: {date_str}", bold=True)
    _add_docx_para(doc, f"TO:\nThe Station House Officer (SHO),\n{station}.", bold=True)
    _add_docx_para(
        doc,
        f"SUBJECT: WRITTEN COMPLAINT UNDER SECTION 173 BNSS 2023 FOR REGISTRATION OF FIR UNDER {bns_sections.upper()}",
        bold=True,
        space_after=8,
    )
    _add_docx_kv_table(
        doc,
        [
            ("Complainant Name & Contact:", f"{complainant} (Ph: {complainant_phone})"),
            ("Complainant Address:", complainant_addr),
            ("Date & Time of Occurrence:", incident_time),
            ("Place of Occurrence:", incident_loc),
            ("Name & Details of Accused:", accused),
            ("Operative Statutory Sections:", f"{bns_sections} read with Section 173 BNSS 2023"),
        ],
    )
    for p in paras:
        _add_docx_para(doc, p)
    _add_docx_para(
        doc,
        f"\nPLACE: ___________________   DATE: {date_str}\nSIGNATURE OF COMPLAINANT: ___________________________ ({complainant})",
        bold=True,
    )
    _add_docx_counterfoil(
        doc,
        "POLICE STATION RECEIVING COUNTERFOIL (MANDATORY ACKNOWLEDGEMENT COPY - SEC 173 BNSS 2023)",
        [
            f"Received Written Complaint from: {complainant} regarding incident at {incident_loc} ({bns_sections})",
            f"Police Station: {station} | Date & Time Received: ____________ | GD/DD/FIR Entry No: ________________",
            "Receiving Officer Name & Belt No: ________________________________ | Police Station Seal: ________________",
        ],
    )
    doc.save(docx_path)

    return {"pdf_path": pdf_path, "docx_path": docx_path}


# ── 5. UNPAID SALARY & LABOUR DUES (PDF & DOCX) ─────────────────────────────
def build_labour_dues(data: Dict[str, Any], base_filename: str) -> Dict[str, str]:
    data = normalize_tool_params("labour_dues", data)
    out_dir = _ensure_output_dir()
    pdf_path = str(out_dir / f"{base_filename}.pdf")
    docx_path = str(out_dir / f"{base_filename}.docx")

    emp_name = data.get("employee_name") or "Aggrieved Employee"
    emp_desig = data.get("designation") or "Employee"
    emp_id = data.get("employee_id") or "N/A"
    emp_phone = data.get("employee_phone") or "Mobile Number"

    company = data.get("company_name") or "Company Name"
    company_addr = data.get("office_address") or "Registered Office Address"
    hr_email = data.get("hr_email") or ""

    total_dues = data.get("total_dues_amount") or "0"
    breakdown = data.get("breakdown") or "Pending monthly salary, unpaid leave encashment, and Full & Final (FnF) settlement"
    lwd = data.get("last_working_day") or "Recent date"
    date_str = datetime.now().strftime("%d-%m-%Y")
    deadline_date = (datetime.now() + timedelta(days=7)).strftime("%d-%m-%Y")

    has_gratuity = "gratuity" in breakdown.lower()
    statute_subtitle = (
        "UNDER SECTIONS 17 & 18 OF CODE ON WAGES 2019, SECTION 15 PAYMENT OF WAGES ACT 1936"
        + (" & PAYMENT OF GRATUITY ACT 1972" if has_gratuity else " & SHOPS & ESTABLISHMENTS ACT")
    )

    # 1. Build PDF
    pdf_doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = _get_pdf_styles()
    story = []

    story.append(Paragraph("LEGAL DEMAND NOTICE FOR UNPAID SALARY & FNF DUES", styles["title"]))
    story.append(Paragraph(_rl_safe(statute_subtitle), styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a365d"), spaceAfter=10))

    to_text = (
        f"<b>TO:</b><br/>The Management / Board of Directors & HR Department,<br/>"
        f"<b>{company}</b>,<br/>{company_addr}."
        + (f"<br/>Email: {hr_email}" if hr_email else "")
    )
    t_to = Table([[Paragraph(_rl_safe(to_text), styles["body"])]], colWidths=[6.8 * inch])
    t_to.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_to)
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            _rl_safe(
                f"<b>SUBJECT: FORMAL DEMAND NOTICE FOR IMMEDIATE DISBURSEMENT OF WITHHELD SALARY AND FNF SETTLEMENT OF RS. {total_dues}/-</b>"
            ),
            styles["heading"],
        )
    )
    story.append(Spacer(1, 6))

    table_data = [
        [Paragraph("<b>Employee Name & Designation:</b>", styles["box_text"]), Paragraph(_rl_safe(f"{emp_name} ({emp_desig}, Emp ID: {emp_id})"), styles["box_text"])],
        [Paragraph("<b>Resignation / Last Working Day:</b>", styles["box_text"]), Paragraph(_rl_safe(lwd), styles["box_text"])],
        [Paragraph("<b>Total Outstanding Dues:</b>", styles["box_text"]), Paragraph(_rl_safe(f"<b>Rs. {total_dues}/-</b>"), styles["box_text"])],
        [Paragraph("<b>Itemized Breakdown of Dues:</b>", styles["box_text"]), Paragraph(_rl_safe(breakdown), styles["box_text"])],
        [Paragraph("<b>Peremptory Cure Window:</b>", styles["box_text"]), Paragraph(_rl_safe(f"<b>7 Days (On or before {deadline_date})</b>"), styles["box_text"])],
    ]
    t_dues = Table(table_data, colWidths=[2.5 * inch, 4.3 * inch])
    t_dues.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#ecfdf5")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_dues)
    story.append(Spacer(1, 8))

    custom_paras = data.get("custom_body_paras")
    if custom_paras and isinstance(custom_paras, list) and len(custom_paras) > 0:
        paras = ["Dear Management,"] + [str(p) for p in custom_paras]
    else:
        gratuity_clause = (
            " and <b>Section 4 read with Section 7(3A) of the Payment of Gratuity Act, 1972</b>"
            if has_gratuity
            else ""
        )
        paras = [
            "Dear Management,",
            f"<b>1. EMPLOYMENT & CESSATION OF SERVICE:</b><br/>I was employed with <b>{company}</b> as {emp_desig}. My employment concluded upon my resignation / relief on <b>{lwd}</b>.",
            f"<b>2. STATUTORY MANDATE (CODE ON WAGES 2019 & PAYMENT OF WAGES ACT 1936):</b><br/>Under <b>Section 17(2) and Section 18 of the Code on Wages, 2019</b>, where an employee resigns or is removed or dismissed from service, all earned wages must be paid within <b>2 (two) working days</b> of such resignation or removal. Further, unauthorized withholding of earned wages is expressly prohibited under <b>Section 15 of the Payment of Wages Act, 1936</b>{gratuity_clause}.",
            f"<b>3. UNLAWFUL WITHHOLDING OF EARNED DUES:</b><br/>In direct breach of the mandatory provisions of the Code on Wages, 2019 and the Payment of Wages Act, 1936, {company} has unlawfully withheld my legitimate dues totaling <b>Rs. {total_dues}/-</b> ({breakdown}).",
            f"<b>4. 7-DAY PEREMPTORY DEMAND:</b><br/>You are hereby called upon to disburse and remit the entire sum of <b>Rs. {total_dues}/-</b> together with interest @ 18% p.a. into my registered salary bank account within <b>7 (seven) days</b> from receipt of this notice (on or before <b>{deadline_date}</b>), along with my relieving letter and Form 16.",
            "<b>5. ESCALATION TO LABOUR COMMISSIONER:</b><br/>Take notice that upon failure to settle the dues within 7 days, I shall file a formal recovery petition before the <b>Authority under the Code on Wages, 2019 / District Labour Commissioner</b> under Section 45 of the Code on Wages, 2019 and Section 15 of the Payment of Wages Act, 1936 seeking the principal sum along with statutory penalty compensation up to <b>ten times (10x)</b> the withheld amount, holding the Company and its Directors jointly and severally liable.",
        ]
    for p in paras:
        story.append(Paragraph(_rl_safe(p), styles["body"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 14))
    sign_data = [
        [
            Paragraph(_rl_safe(f"<b>PLACE:</b> ___________________<br/><b>DATE:</b> {date_str}"), styles["box_text"]),
            Paragraph(
                _rl_safe(f"___________________________<br/><b>{emp_name}</b><br/>Phone: {emp_phone}"),
                styles["box_text"],
            ),
        ]
    ]
    t_sign = Table(sign_data, colWidths=[3.4 * inch, 3.4 * inch])
    story.append(t_sign)
    pdf_doc.build(story)

    # 2. Build DOCX with full table parity
    doc = _init_docx_doc("LEGAL DEMAND NOTICE FOR UNPAID SALARY & FNF DUES", statute_subtitle)
    _add_docx_para(doc, f"DATE: {date_str}", bold=True)
    _add_docx_para(
        doc,
        f"TO:\nThe Management / Board of Directors & HR Department,\n{company},\n{company_addr}."
        + (f"\nEmail: {hr_email}" if hr_email else ""),
        bold=True,
    )
    _add_docx_para(
        doc,
        f"SUBJECT: FORMAL DEMAND NOTICE FOR IMMEDIATE DISBURSEMENT OF WITHHELD SALARY AND FNF SETTLEMENT OF RS. {total_dues}/-",
        bold=True,
        space_after=8,
    )
    _add_docx_kv_table(
        doc,
        [
            ("Employee Name & Designation:", f"{emp_name} ({emp_desig}, Emp ID: {emp_id})"),
            ("Resignation / Last Working Day:", lwd),
            ("Total Outstanding Dues:", f"Rs. {total_dues}/-"),
            ("Itemized Breakdown of Dues:", breakdown),
            ("Peremptory Cure Window:", f"7 Days (On or before {deadline_date})"),
        ],
    )
    for p in paras:
        _add_docx_para(doc, p)
    _add_docx_para(doc, f"\nPLACE: ___________________   DATE: {date_str}\n\nYOURS FAITHFULLY,\n___________________________\n{emp_name}\nPhone: {emp_phone}", bold=True)
    doc.save(docx_path)

    return {"pdf_path": pdf_path, "docx_path": docx_path}


# ── 6. CASE STRATEGY & LEGAL OPINION MEMO (PDF & DOCX) ─────────────────────
def build_case_strategy(data: Dict[str, Any], base_filename: str) -> Dict[str, str]:
    out_dir = _ensure_output_dir()
    pdf_path = str(out_dir / f"{base_filename}.pdf")
    docx_path = str(out_dir / f"{base_filename}.docx")

    summary = data.get("summary") or "Case strategy overview"
    domain = (data.get("domain") or "General Legal").upper()
    comp = data.get("compensation_estimate") or {}
    min_amt = comp.get("min_amount", 0)
    max_amt = comp.get("max_amount", 0)
    basis = comp.get("basis") or "Statutory computation"
    forums = data.get("forums") or ["Civil Court of competent jurisdiction"]
    stages = data.get("stages") or []
    raw_ev = data.get("evidence_checklist") or []
    if isinstance(raw_ev, dict):
        evidence = raw_ev.get("items") or []
    elif isinstance(raw_ev, list):
        evidence = raw_ev
    else:
        evidence = []
    date_str = datetime.now().strftime("%d-%m-%Y")

    # 1. Build PDF
    pdf_doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = _get_pdf_styles()
    story = []

    story.append(Paragraph("LEGAL OPINION & CASE STRATEGY MEMO", styles["title"]))
    story.append(
        Paragraph(
            _rl_safe(f"DOMAIN: {domain} | PREPARED FOR LITIGATION PLANNING & DISPUTE RESOLUTION"),
            styles["subtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a365d"), spaceAfter=10))

    story.append(Paragraph("<b>1. EXECUTIVE CASE BRIEF:</b>", styles["heading"]))
    t_sum = Table(
        [[Paragraph(_rl_safe(f"<b>Facts & Situation Analysis:</b><br/>{summary}"), styles["body"])]],
        colWidths=[6.8 * inch],
    )
    t_sum.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_sum)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>2. QUANTIFIED COMPENSATION ESTIMATE & FORUM JURISDICTION:</b>", styles["heading"]))
    comp_data = [
        [
            Paragraph("<b>Estimated Recoverable Range:</b>", styles["box_text"]),
            Paragraph(_rl_safe(f"<b>Rs. {min_amt:,.0f} - Rs. {max_amt:,.0f} (INR)</b>"), styles["box_text"]),
        ],
        [
            Paragraph("<b>Calculation Basis & Precedents:</b>", styles["box_text"]),
            Paragraph(_rl_safe(basis), styles["box_text"]),
        ],
        [
            Paragraph("<b>Recommended Judicial Forums:</b>", styles["box_text"]),
            Paragraph(_rl_safe("<br/>".join(f"• {f}" for f in forums)), styles["box_text"]),
        ],
    ]
    t_comp = Table(comp_data, colWidths=[2.5 * inch, 4.3 * inch])
    t_comp.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0fdf4")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t_comp)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>3. STEP-BY-STEP LITIGATION ROADMAP:</b>", styles["heading"]))
    if stages:
        for idx, stage in enumerate(stages, 1):
            s_name = stage.get("name", f"Stage {idx}")
            s_desc = stage.get("description", "")
            s_time = stage.get("timeline", "Standard timeline")
            story.append(
                Paragraph(_rl_safe(f"<b>Step {idx}: {s_name}</b> ({s_time})<br/>{s_desc}"), styles["body"])
            )
            story.append(Spacer(1, 3))
    else:
        story.append(
            Paragraph(
                "1. Issue a peremptory statutory legal notice giving 15 days compliance.<br/>"
                "2. Compile documentary proof & transaction logs.<br/>"
                "3. File complaint/petition before the designated judicial forum within limitation.",
                styles["body"],
            )
        )

    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>4. EVIDENTIARY CHECKLIST (UNDER BSA 2023):</b>", styles["heading"]))
    ev_items = (
        evidence
        if evidence
        else [
            "Contract / Agreement / Lease deed in original or certified copy",
            "Bank transaction statements, receipts, and payment proofs",
            "Written communications (Email threads, WhatsApp chat exports with Section 63 BSA 2023 certificate)",
            "Police complaint copy / acknowledgment receipt (if applicable)",
        ]
    )
    ev_text = "<br/>".join(f"[ ]  {_rl_safe(item)}" for item in ev_items[:10])
    story.append(Paragraph(ev_text, styles["body"]))

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "<b>LEGAL DISCLAIMER:</b> This strategy memo is generated for advisory and planning purposes. It does not constitute formal attorney-client representation. Statutory limitation periods and court procedures must be confirmed by an Advocate-on-Record before filing.",
            styles["small"],
        )
    )
    pdf_doc.build(story)

    # 2. Build DOCX with full table and roadmap parity
    doc = _init_docx_doc(
        "LEGAL OPINION & CASE STRATEGY MEMO",
        f"DOMAIN: {domain} | LITIGATION PLANNING & FORUM JURISDICTION",
    )
    _add_docx_para(doc, f"DATE: {date_str}", bold=True)
    _add_docx_para(doc, f"1. EXECUTIVE CASE BRIEF:\n{summary}", space_after=6)
    _add_docx_kv_table(
        doc,
        [
            ("Estimated Recoverable Range:", f"Rs. {min_amt:,.0f} - Rs. {max_amt:,.0f} (INR)"),
            ("Calculation Basis:", basis),
            ("Recommended Judicial Forums:", "\n".join(f"• {f}" for f in forums)),
        ],
    )
    _add_docx_para(doc, "3. STEP-BY-STEP LITIGATION ROADMAP:", bold=True)
    if stages:
        for idx, stage in enumerate(stages, 1):
            s_name = stage.get("name", f"Stage {idx}")
            s_desc = stage.get("description", "")
            s_time = stage.get("timeline", "Standard timeline")
            _add_docx_para(doc, f"Step {idx}: {s_name} ({s_time}) — {s_desc}")
    else:
        _add_docx_para(
            doc,
            "Step 1: Issue a peremptory statutory legal notice giving 15 days compliance.\n"
            "Step 2: Compile documentary proof & transaction logs.\n"
            "Step 3: File complaint/petition before the designated judicial forum within limitation.",
        )
    _add_docx_para(doc, "\n4. EVIDENTIARY CHECKLIST (UNDER BSA 2023):", bold=True)
    for item in ev_items[:10]:
        _add_docx_para(doc, f"[ ] {item}", space_after=2)
    _add_docx_para(
        doc,
        "\nLEGAL DISCLAIMER: This strategy memo is generated for advisory and planning purposes. It does not constitute formal attorney-client representation.",
        italic=True,
        font_size=8.5,
    )
    doc.save(docx_path)

    return {"pdf_path": pdf_path, "docx_path": docx_path}
