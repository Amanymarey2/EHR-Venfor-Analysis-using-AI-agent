# EHR Venfor Analysis using AI agent

# AI-Powered EHR Vendor Analysis Agent

**Automates EHR vendor evaluation from RFI to ROI in 5 AI steps.**

This project uses **LangGraph**, **LangChain**, and **LLM orchestration** to:
1. Extract criteria from an RFI
2. Parse vendor responses
3. Score vendors using weighted multi-criteria analysis
4. Generate **Markdown comparison tables**
5. Project **10-year TCO and ROI**

---

## Features

| Feature | Status |
|-------|--------|
| Full offline execution | Done |
| State persistence | Done |
| Clean Markdown tables | Done |
| Realistic mock data | Done |
| Debug logging | Done |
| PDF report ready | Done |

---

## Project Structure

EHR-Vendor-Analysis-AI-Agent/
│
├── rfi.txt                  # RFI from HTH 105 course
├── vendor_responses.txt     # Grok-generated vendor replies
├── helper.py                # Core agent, tools, state
├── prompts.py               # Planner prompt
├── workflow_diagram.png     # Visual workflow
├── HTH105_Final_Report.Rmd  # Final report (R Markdown)
├── HTH105_Final_Report.pdf  # Rendered PDF (optional)
├── test_agent.ipynb         # Run & debug notebook
└── README.md                # This file



---

## Setup

```bash
# 1. Clone repo
git clone https://github.com/Amanymarey2/EHR-Vendor-Analysis-AI-Agent.git
cd EHR-Vendor-Analysis-AI-Agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install langgraph langchain langchain-core chardet python-dotenv
