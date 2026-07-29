import sys
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime, timedelta
from typing import Dict
import os

# Set console output encoding to UTF-8 to prevent UnicodeEncodeError on Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

class LegalNoticeAgent:
    """Generate formal legal notice PDF"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
    
    def generate_notice(self, case_data: Dict, output_path: str = "output/legal_notice.pdf") -> str:
        """Generate legal notice PDF"""
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        elements = []
        
        # Header
        header_style = ParagraphStyle(
            'Header',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1a3a6b'),
            spaceAfter=10
        )
        elements.append(Paragraph("LEGAL NOTICE", header_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Sender info
        sender_info = f"""
        <b>From:</b> {case_data.get('sender_name', 'Your Name')}<br/>
        {case_data.get('sender_address', 'Your Address')}<br/>
        <b>To:</b> {case_data.get('recipient_name', 'Recipient Name')}<br/>
        {case_data.get('recipient_address', 'Recipient Address')}<br/>
        <b>Date:</b> {datetime.now().strftime('%d-%m-%Y')}
        """
        elements.append(Paragraph(sender_info, self.styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Subject
        subject_text = f"<b>Subject: Legal Notice for {case_data.get('issue_type', 'Legal Issue')}</b>"
        elements.append(Paragraph(subject_text, self.styles['Heading2']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Body
        body_text = f"""
        Dear {case_data.get('recipient_name', 'Sir/Madam')},<br/><br/>
        
        This is to formally notify you that you have violated my legal rights as follows:<br/><br/>
        
        <b>Facts of the case:</b><br/>
        {case_data.get('issue_description', 'Issue description')}<br/><br/>
        
        <b>Applicable Law:</b><br/>
        The aforementioned act/section is {case_data.get('applicable_section', 'the relevant law section')}<br/><br/>
        
        <b>Compensation Demanded:</b><br/>
        I hereby demand compensation of Rs. {case_data.get('demand_amount', '0')} to be paid within 15 days from receipt of this notice.<br/><br/>
        
        <b>Deadline:</b><br/>
        You are required to comply with this demand by {(datetime.now() + timedelta(days=15)).strftime('%d-%m-%Y')}.<br/><br/>
        
        Failure to comply will result in legal action without further notice.<br/><br/>
        
        Yours faithfully,<br/>
        {case_data.get('sender_name', 'Your Name')}<br/>
        Signature: ___________________
        """
        elements.append(Paragraph(body_text, self.styles['BodyText']))
        
        # Build PDF
        doc.build(elements)
        return output_path

if __name__ == "__main__":
    agent = LegalNoticeAgent()
    
    test_case = {
        "sender_name": "Vatsal Desai",
        "sender_address": "123 Main Street, Ahmedabad, Gujarat 380001",
        "recipient_name": "ABC E-commerce Ltd",
        "recipient_address": "456 Commerce Park, Mumbai, Maharashtra 400001",
        "issue_type": "Defective Product",
        "issue_description": "Purchased laptop (Model XYZ) for Rs 50,000 on 15-07-2024. Product became defective within 2 weeks and seller refuses to provide refund or replacement.",
        "applicable_section": "Consumer Protection Act 2019, Section 35",
        "demand_amount": "50000"
    }
    
    pdf_path = agent.generate_notice(test_case)
    print(f"✅ Legal notice generated: {pdf_path}")
