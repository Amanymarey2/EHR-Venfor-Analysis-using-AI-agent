
# prompts.py
import json

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage
from typing import List
#from helper import get_agent_descriptions
# ----------------------------------------------------------------------
# 7. AGENT DESCRIPTIONS (for planner)
# ----------------------------------------------------------------------
def get_agent_descriptions() -> Dict:
    """Returns metadata about each tool/agent for the LLM planner."""
    return {
        "rfi_analyzer": {
            "name": "RFI Analyzer",
            "capability": "Generates weighted criteria from RFI text",
            "use_when": "RFI text is provided",
            "output_format": "JSON with 'weights'"
        },
        "vendor_extractor": {
            "name": "Vendor Extractor",
            "capability": "Extracts vendor names from file (fallback to Tavily)",
            "use_when": "Vendor responses file is provided",
            "output_format": "List of vendor names"
        },
        "criteria_scorer": {
            "name": "Criteria Scorer",
            "capability": "Scores vendors using live Tavily data and weights",
            "use_when": "Weights and vendors are available",
            "output_format": "Ranked list of scored vendors"
        },
        "table_generator": {
            "name": "Score Table Generator",
            "capability": "Creates Markdown score table",
            "use_when": "Scoring is complete",
            "output_format": "Markdown table"
        },
        "roi_table_generator": {
            "name": "ROI Table Generator",
            "capability": "Creates ROI table with live benchmarks",
            "use_when": "Vendors are known",
            "output_format": "Markdown ROI table"
        }
    }

def plan_prompt(state) -> List[HumanMessage]:
    print("DEBUG: plan_prompt called")

    try:
        agents = get_agent_descriptions()
    except Exception as e:
        print("WARNING: Could not load agent descriptions:", e)
        agents = {}

    guidelines = "\n".join([
        f"- `{name}`: {desc.get('use_when', 'unknown')}"
        for name, desc in agents.items()
    ]) or "- No agent descriptions available"

    rfi_preview = (state["rfi_text"][:400] + "...") if state.get("rfi_text") else "None"
    vendor_preview = (state["vendor_text"][:400] + "...") if state.get("vendor_text") else "None"

    required_json = json.dumps({  # ← NOW WORKS
        "1": {"agent": "rfi_analyzer"},
        "2": {"agent": "vendor_extractor"},
        "3": {"agent": "criteria_scorer"},
        "4": {"agent": "table_generator"},
        "5": {"agent": "roi_table_generator"}
    }, indent=2)

    prompt_text = f"""
YOU ARE THE PLANNER. RETURN **ONLY** THE JSON BELOW. NO TEXT.

**AVAILABLE TOOLS:**
{guidelines}

**INPUT FILES ARE PRESENT:**
- RFI: YES (length: {len(state.get('rfi_text', ''))} characters)
- Vendors: YES (length: {len(state.get('vendor_text', ''))} characters)

**PREVIEW:**
--- RFI ---
{rfi_preview}
--- VENDORS ---
{vendor_preview}

**RETURN EXACTLY THIS:**

```json
{required_json}

DO NOT CHANGE THE ORDER. DO NOT ADD TEXT. RETURN ONLY THE JSON.
"""
    return [HumanMessage(content=prompt_text.strip())]


    #return [HumanMessage(content=prompt_text)]

