import sys
import io
from datetime import datetime
from typing import Dict

# Set console output encoding to UTF-8 to prevent UnicodeEncodeError on Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

class RTIApplicationAgent:
    """Generate RTI (Right to Information) application"""
    
    def __init__(self):
        self.pio_addresses = {
            "municipal": {
                "name": "Municipal Corporation Office",
                "address": "Enter your city's municipal office address"
            },
            "police": {
                "name": "Police Commissioner's Office",
                "address": "Enter your district police HQ address"
            },
            "education": {
                "name": "State Education Department",
                "address": "Enter education department address"
            },
            "health": {
                "name": "State Health Department",
                "address": "Enter health department address"
            }
        }
    
    def generate_rti_application(self, data: Dict) -> str:
        """Generate RTI Section 6 application"""
        
        application = f"""
RTI APPLICATION (Section 6, RTI Act 2005)

Date: {datetime.now().strftime('%d-%m-%Y')}

TO,
The Public Information Officer (PIO),
{data.get('pio_office', 'Government Office Name')},
{data.get('pio_address', 'Office Address')},

Dear Sir/Madam,

I hereby apply under Section 6 of the Right to Information Act, 2005, to obtain the following information:

INFORMATION REQUESTED:
{data.get('information_sought', 'Describe the information you want')}

RELEVANT ACTS/SECTIONS:
The above information is available under public domain and I have the right to this information as per RTI Act 2005.

METHOD OF RECEIPT:
I prefer to receive the information through: (Please choose one)
☐ E-mail
☐ Postal mail
☐ In person

APPLICATION FEE:
I am enclosing Rs. 10 (Rupees Ten) as the application fee in the form of:
☐ Indian Postal Order
☐ Bank Draft
☐ Demand Draft

DETAILS OF APPLICANT:
Name: {data.get('applicant_name', 'Your Name')}
Address: {data.get('applicant_address', 'Your Address')}
Phone: {data.get('applicant_phone', 'Your Phone')}
Email: {data.get('applicant_email', 'Your Email')}

Yours faithfully,

{data.get('applicant_name', 'Your Name')}
Signature: ___________________
Date: {datetime.now().strftime('%d-%m-%Y')}

OFFICE USE ONLY:
Date Received: ___________
Reference No: ___________
PIO Sign: ___________
"""
        
        return application

if __name__ == "__main__":
    agent = RTIApplicationAgent()
    
    test_data = {
        "pio_office": "Municipal Corporation, Ahmedabad",
        "pio_address": "Municipal House, Ahmedabad, Gujarat",
        "information_sought": "I request copies of all expenditure details on infrastructure development in ward number 50 for the year 2023-24",
        "applicant_name": "Vatsal Desai",
        "applicant_address": "123 Main Street, Ahmedabad",
        "applicant_phone": "9876543210",
        "applicant_email": "vatsal@example.com"
    }
    
    rti_app = agent.generate_rti_application(test_data)
    print(rti_app)
