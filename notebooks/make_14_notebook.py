import json

nb = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.12.0"}
 },
 "cells": []
}

def md(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}

def code(cid, src):
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": src}

nb["cells"] = [

# ── Title & Prominent Warning ────────────────────────────────────────────────
md("md-title", [
    "# 14 · Rule-Based Training Recommendation Engine (Version 1)\n",
    "\n",
    "**Project:** Enterprise HR AI  \n",
    "\n",
    "> ### ⚠️ PROMINENT DATA INTEGRITY WARNING\n",
    "> **SYNTHETIC DATA — employee current-skill possession was not present in any source file and has been simulated using a tenure/training-based heuristic for MVP demonstration purposes only. This must NOT be presented to stakeholders as real observed skill data. Real deployment requires an actual skills inventory (HRIS export, LMS completion records, or self-assessment survey).**\n",
    "\n",
    "---"
]),

# ── Roadmap & Architectural Scope ────────────────────────────────────────────
md("md-roadmap", [
    "---\n",
    "## Step 1 · Architecture & Staged Implementation Roadmap\n",
    "\n",
    "Per the project design specification (DOCX), the recommendation engine is built in stages:\n",
    "\n",
    "### Version 1 (Current Implementation)\n",
    "- **Methodology:** Deterministic, direct rule-based dictionary lookup.\n",
    "- **Engine Mechanics:** Each of the 33 benchmark skills is mapped to a concrete, curated course title. For each employee, their top missing skills are mapped directly to actionable training interventions.\n",
    "- **ML Dependency:** **None.** No `sentence-transformers`, vector databases, or complex embeddings are used in this notebook.\n",
    "\n",
    "### Version 2 (Planned Future Upgrade)\n",
    "- **Methodology:** Semantic / Embedding-Based Matching.\n",
    "- **Planned Mechanics:** When an enterprise course catalog (e.g., Coursera, Udemy Business, or internal LMS with 500+ syllabus descriptions) becomes available, a dense embedding model (such as `sentence-transformers/all-MiniLM-L6-v2`) will generate vector representations of each skill gap and compute cosine similarity against all course description embeddings to rank top-k learning resources.\n",
    "- **Status:** Documented architectural roadmap; deferred to later phase per DOCX specifications."
]),

# ── Step 2: Load Employee Skill Gaps ─────────────────────────────────────────
md("md-step2", [
    "---\n",
    "## Step 2 · Load Employee Skill Gaps"
]),

code("cell-load", [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "\n",
    "PROC = os.path.join('..', 'data', 'processed')\n",
    "gaps_file = os.path.join(PROC, 'employee_skill_gaps.csv')\n",
    "\n",
    "# Load employee skill gaps skipping the header comment\n",
    "df_gaps = pd.read_csv(gaps_file, comment='#')\n",
    "print(f'Loaded employee gap records: {len(df_gaps):,} employees (102 Managers excluded per Step 15/16)')\n",
    "print(f'Columns: {list(df_gaps.columns)}')\n",
    "print('\\nSeverity distribution:')\n",
    "print(df_gaps['severity'].value_counts())"
]),

# ── Step 3: Build 33-Skill Concrete Course Catalog ───────────────────────────
md("md-step3", [
    "---\n",
    "## Step 3 · Curated Training Catalog (33 Concrete Course Recommendations)\n",
    "\n",
    "Mapping all 33 unique benchmark skills to concrete, specific professional development and technical training courses."
]),

code("cell-catalog", [
    "RECOMMENDATION_CATALOG = {\n",
    "    # ── Foundational & Essential Skills ──\n",
    "    \"Speaking\": \"Executive Presentation & Public Speaking Masterclass (Toastmasters / Internal Workshop)\",\n",
    "    \"Reading Comprehension\": \"Technical & Regulatory Documentation Analysis Workshop\",\n",
    "    \"Active Listening\": \"Empathetic Leadership & Active Listening for Cross-Functional Collaboration\",\n",
    "    \"Critical Thinking\": \"Strategic Problem Solving & Root Cause Decision Analysis Seminar\",\n",
    "    \"Active Learning\": \"Continuous Professional Learning & Rapid Skill Acquisition Frameworks\",\n",
    "    \"Monitoring\": \"Operational Process Auditing & KPI Performance Monitoring Protocols\",\n",
    "    \"Science\": \"Scientific Methodology, Evidence-Based Rigor & Laboratory Standards Training\",\n",
    "    \"Writing\": \"Business & Technical Writing: Structuring Executive Summaries & Proposals\",\n",
    "    \n",
    "    # ── Cloud & Engineering Tools (AWS / Cloud) ──\n",
    "    \"Amazon Web Services AWS CloudFormation\": \"AWS Infrastructure as Code: CloudFormation & CDK Automated Deployments\",\n",
    "    \"Amazon Elastic Compute Cloud EC2\": \"AWS Compute Architecture: Scalable EC2 Fleet Management & Auto-Scaling\",\n",
    "    \"Amazon Web Services AWS software\": \"AWS Solutions Architect: Core Cloud Services, IAM & Architecture Design\",\n",
    "    \"Amazon DynamoDB\": \"NoSQL Database Architecture with AWS DynamoDB: Modeling & Scalability\",\n",
    "    \"Amazon Redshift\": \"Cloud Data Warehousing & High-Performance SQL Analytics with Amazon Redshift\",\n",
    "    \n",
    "    # ── Enterprise & Office Productivity Software ──\n",
    "    \"Microsoft Office software\": \"Enterprise Microsoft 365 Productivity & Workflow Automation Bootcamp\",\n",
    "    \"Microsoft Excel\": \"Advanced Excel: Dynamic Arrays, Power Query & Business Financial Modeling\",\n",
    "    \"Adobe Acrobat\": \"Adobe Acrobat Pro: Digital Signatures, Forms & Secure Document Workflows\",\n",
    "    \"Microsoft Outlook\": \"Time Management, Calendar Optimization & Executive Email Triage in Outlook\",\n",
    "    \"Google Docs\": \"Google Workspace Collaboration: Document Co-Authoring & Cloud Governance\",\n",
    "    \"Apple macOS\": \"macOS for Enterprise: Advanced Terminal, Security & Productivity Tooling\",\n",
    "    \"Microsoft Access\": \"Relational Database Design & SQL Querying with Microsoft Access\",\n",
    "    \n",
    "    # ── Data, Analytics & Development Software ──\n",
    "    \"IBM SPSS Statistics\": \"Advanced Statistical Inference & Predictive Modeling using IBM SPSS\",\n",
    "    \"Eclipse IDE\": \"Java & Multi-Language Software Development with Eclipse IDE & Git Plugins\",\n",
    "    \"ESRI ArcGIS software\": \"Spatial Data Analytics & Geospatial Mapping with ESRI ArcGIS Pro\",\n",
    "    \n",
    "    # ── Specialized Healthcare & Industry Software ──\n",
    "    \"MEDITECH software\": \"MEDITECH EHR Clinical Data Management & Laboratory Information Systems Track\",\n",
    "    \n",
    "    # ── Design & CAD Software ──\n",
    "    \"Bentley MicroStation\": \"Bentley MicroStation CAD: 2D/3D Infrastructure Drafting & Asset Modeling\",\n",
    "    \"Autodesk AutoCAD\": \"AutoCAD Essentials: Mechanical/Architectural Drafting & Dimensioning Standards\",\n",
    "    \"Adobe Creative Cloud software\": \"Adobe Creative Cloud Bootcamp: Multi-App Visual Design & Asset Management\",\n",
    "    \"Adobe Photoshop\": \"Commercial Image Retouching & Asset Production with Adobe Photoshop\",\n",
    "    \"Adobe InDesign\": \"Corporate Layout Design, Multi-Page Publishing & Pitch Decks with InDesign\",\n",
    "    \"Adobe After Effects\": \"Motion Graphics, Video Storytelling & Product Animation with After Effects\",\n",
    "    \"Adobe Illustrator\": \"Vector Graphics, Infographics & Brand Asset Design in Adobe Illustrator\",\n",
    "    \n",
    "    # ── Sales & Marketing Platforms ──\n",
    "    \"HubSpot software\": \"Inbound Sales & CRM Pipeline Optimization with HubSpot Sales Hub\",\n",
    "    \"Facebook\": \"B2B Social Media Marketing, Meta Business Suite & Targeted Outreach Campaigns\"\n",
    "}\n",
    "\n",
    "print(f'Total catalog courses configured: {len(RECOMMENDATION_CATALOG)}')\n",
    "assert len(RECOMMENDATION_CATALOG) == 33, 'Catalog must contain exactly 33 unique courses!'"
]),

# ── Step 4: Generate Employee Recommendations ────────────────────────────────
md("md-step4", [
    "---\n",
    "## Step 4 · Generate Recommendations per Employee\n",
    "\n",
    "For each employee:\n",
    "- Extract their missing skills list.\n",
    "- Select the **top 3 missing skills** (or fewer if less than 3 are missing).\n",
    "- Map each missing skill directly to its concrete course recommendation.\n",
    "- If an employee has no skill gaps (0 missing), record `'None - No skill gaps identified'`."
]),

code("cell-generate-recs", [
    "rec_rows = []\n",
    "\n",
    "for _, row in df_gaps.iterrows():\n",
    "    emp_id = row['EmployeeNumber']\n",
    "    role = row['JobRole']\n",
    "    sev = row['severity']\n",
    "    missing_raw = row['missing_skills']\n",
    "    \n",
    "    if pd.isna(missing_raw) or str(missing_raw).strip() in ('', 'None', 'nan'):\n",
    "        top3_skills_str = 'None'\n",
    "        top3_recs_str = 'None - No skill gaps identified'\n",
    "    else:\n",
    "        skills = [s.strip() for s in str(missing_raw).split(';') if s.strip()]\n",
    "        top3_skills = skills[:3]\n",
    "        top3_recs = [RECOMMENDATION_CATALOG.get(s, f'Targeted Training for {s}') for s in top3_skills]\n",
    "        \n",
    "        top3_skills_str = '; '.join(top3_skills)\n",
    "        top3_recs_str = '; '.join(top3_recs)\n",
    "        \n",
    "    rec_rows.append({\n",
    "        'EmployeeNumber': emp_id,\n",
    "        'JobRole': role,\n",
    "        'severity': sev,\n",
    "        'top_3_missing_skills': top3_skills_str,\n",
    "        'top_3_recommendations': top3_recs_str\n",
    "    })\n",
    "\n",
    "df_recs = pd.DataFrame(rec_rows)\n",
    "print(f'Total employee recommendation profiles generated: {len(df_recs):,}')\n",
    "assert len(df_recs) == 1368, f'Expected 1,368 non-manager recommendation profiles, got {len(df_recs)}'\n",
    "\n",
    "print('\\nSample Employee Recommendations:')\n",
    "for _, r in df_recs.head(5).iterrows():\n",
    "    print(f'\\nEmployee #{r[\"EmployeeNumber\"]} ({r[\"JobRole\"]} - [{r[\"severity\"]} Severity])')\n",
    "    print(f'  Top Missing Skills : {r[\"top_3_missing_skills\"]}')\n",
    "    print(f'  Recommendations    : {r[\"top_3_recommendations\"]}')"
]),

# ── Step 5: Save Output CSV with Warning Comment ─────────────────────────────
md("md-step5", [
    "---\n",
    "## Step 5 · Save Recommendations Dataset (`employee_recommendations.csv`)\n",
    "\n",
    "Exporting recommendations to `data/processed/employee_recommendations.csv`.  \n",
    "The synthetic data warning is retained in line 1."
]),

code("cell-save", [
    "out_file = os.path.join(PROC, 'employee_recommendations.csv')\n",
    "\n",
    "warning_comment = (\n",
    "    '# SYNTHETIC DATA — employee current-skill possession was not present in any source file '\n",
    "    'and has been simulated using a tenure/training-based heuristic for MVP demonstration purposes only. '\n",
    "    'This must NOT be presented to stakeholders as real observed skill data. Real deployment requires '\n",
    "    'an actual skills inventory (HRIS export, LMS completion records, or self-assessment survey).\\n'\n",
    ")\n",
    "\n",
    "with open(out_file, 'w', encoding='utf-8') as f:\n",
    "    f.write(warning_comment)\n",
    "    df_recs.to_csv(f, index=False)\n",
    "\n",
    "file_size = os.path.getsize(out_file)\n",
    "print(f'Saved recommendations to: {out_file}')\n",
    "print(f'File size: {file_size:,} bytes')\n",
    "print(f'Total records: {len(df_recs):,}')\n",
    "\n",
    "# Round-trip reload verification\n",
    "df_reload = pd.read_csv(out_file, comment='#')\n",
    "assert len(df_reload) == 1368, 'Row count mismatch on reload!'\n",
    "assert list(df_reload.columns) == ['EmployeeNumber', 'JobRole', 'severity', 'top_3_missing_skills', 'top_3_recommendations']\n",
    "print('CONFIRMED: Round-trip verification passed cleanly with comment handling.')"
])

]

out_path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\notebooks\14_recommendation_engine.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Written: {out_path}')
