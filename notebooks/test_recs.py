import pandas as pd
import os

proc = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\data\processed'
gaps_file = os.path.join(proc, 'employee_skill_gaps.csv')
df_gaps = pd.read_csv(gaps_file, comment='#')

rec_catalog = {
    "Speaking": "Executive Presentation & Public Speaking Masterclass (Toastmasters / Internal Workshop)",
    "Reading Comprehension": "Technical & Regulatory Documentation Analysis Workshop",
    "Active Listening": "Empathetic Leadership & Active Listening for Cross-Functional Collaboration",
    "Critical Thinking": "Strategic Problem Solving & Root Cause Decision Analysis Seminar",
    "Microsoft Office software": "Enterprise Microsoft 365 Productivity & Workflow Automation Bootcamp",
    "Active Learning": "Continuous Professional Learning & Rapid Skill Acquisition Frameworks",
    "Adobe Acrobat": "Adobe Acrobat Pro: Digital Signatures, Forms & Secure Document Workflows",
    "Microsoft Excel": "Advanced Excel: Dynamic Arrays, Power Query & Business Financial Modeling",
    "Monitoring": "Operational Process Auditing & KPI Performance Monitoring Protocols",
    "Amazon Web Services AWS CloudFormation": "AWS Infrastructure as Code: CloudFormation & CDK Automated Deployments",
    "Science": "Scientific Methodology, Evidence-Based Rigor & Laboratory Standards Training",
    "Amazon Elastic Compute Cloud EC2": "AWS Compute Architecture: Scalable EC2 Fleet Management & Auto-Scaling",
    "Amazon Web Services AWS software": "AWS Solutions Architect: Core Cloud Services, IAM & Architecture Design",
    "Amazon DynamoDB": "NoSQL Database Architecture with AWS DynamoDB: Modeling & Scalability",
    "IBM SPSS Statistics": "Advanced Statistical Inference & Predictive Modeling using IBM SPSS",
    "Amazon Redshift": "Cloud Data Warehousing & High-Performance SQL Analytics with Amazon Redshift",
    "MEDITECH software": "MEDITECH EHR Clinical Data Management & Laboratory Information Systems Track",
    "Microsoft Outlook": "Time Management, Calendar Optimization & Executive Email Triage in Outlook",
    "Writing": "Business & Technical Writing: Structuring Executive Summaries & Proposals",
    "Adobe Creative Cloud software": "Adobe Creative Cloud Bootcamp: Multi-App Visual Design & Asset Management",
    "Apple macOS": "macOS for Enterprise: Advanced Terminal, Security & Productivity Tooling",
    "Eclipse IDE": "Java & Multi-Language Software Development with Eclipse IDE & Git Plugins",
    "Google Docs": "Google Workspace Collaboration: Document Co-Authoring & Cloud Governance",
    "Bentley MicroStation": "Bentley MicroStation CAD: 2D/3D Infrastructure Drafting & Asset Modeling",
    "HubSpot software": "Inbound Sales & CRM Pipeline Optimization with HubSpot Sales Hub",
    "Facebook": "B2B Social Media Marketing, Meta Business Suite & Targeted Outreach Campaigns",
    "Adobe Photoshop": "Commercial Image Retouching & Asset Production with Adobe Photoshop",
    "Adobe InDesign": "Corporate Layout Design, Multi-Page Publishing & Pitch Decks with InDesign",
    "Autodesk AutoCAD": "AutoCAD Essentials: Mechanical/Architectural Drafting & Dimensioning Standards",
    "Adobe After Effects": "Motion Graphics, Video Storytelling & Product Animation with After Effects",
    "Adobe Illustrator": "Vector Graphics, Infographics & Brand Asset Design in Adobe Illustrator",
    "Microsoft Access": "Relational Database Design & SQL Querying with Microsoft Access",
    "ESRI ArcGIS software": "Spatial Data Analytics & Geospatial Mapping with ESRI ArcGIS Pro"
}

print(f"Catalog size: {len(rec_catalog)}")

# Process each employee
out_rows = []
for _, r in df_gaps.iterrows():
    emp_id = r['EmployeeNumber']
    role = r['JobRole']
    sev = r['severity']
    ms_str = str(r['missing_skills'])
    
    if pd.isna(r['missing_skills']) or ms_str.strip() in ('', 'None', 'nan'):
        top3_skills_str = 'None'
        top3_recs_str = 'None - No skill gaps identified'
    else:
        skills = [s.strip() for s in ms_str.split(';') if s.strip()]
        top3_skills = skills[:3]
        top3_recs = [rec_catalog.get(s, f"Custom Course for {s}") for s in top3_skills]
        
        top3_skills_str = '; '.join(top3_skills)
        top3_recs_str = '; '.join(top3_recs)
        
    out_rows.append({
        'EmployeeNumber': emp_id,
        'JobRole': role,
        'severity': sev,
        'top_3_missing_skills': top3_skills_str,
        'top_3_recommendations': top3_recs_str
    })

df_out = pd.DataFrame(out_rows)
print(f"Generated {len(df_out)} employee recommendation records")
print("\nFirst 5 records:")
for _, r in df_out.head(5).iterrows():
    print(f"Emp #{r['EmployeeNumber']} ({r['JobRole']} - {r['severity']}):")
    print(f"  Top 3 Missing: {r['top_3_missing_skills']}")
    print(f"  Top 3 Recs   : {r['top_3_recommendations']}")
    print()
