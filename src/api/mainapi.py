
# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import re
# import os

# # --- Load your CSV ---
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# file_path = os.path.join(BASE_DIR, "data", "processed", "new_leads_with_cltv.csv")
# df = pd.read_csv(file_path)

# app = FastAPI()

# # --- Request format ---
# class Query(BaseModel):
#     text: str

# # --- Helper functions ---
# def extract_lead_id(text):
#     match = re.search(r'(Lead_\d+)', text, re.IGNORECASE)
#     return match.group() if match else None

# def get_lead_data(lead_id):
#     lead = df[df["LeadID"] == lead_id]
#     if lead.empty:
#         return None
#     return lead.to_dict(orient="records")[0]

# # --- Simple logic (no LLM for now, clean output) ---
# def generate_response(lead):
#     priority = lead["Lead_Priority"].replace(" Priority", "")
#     cltv = round(lead["Predicted_CLTV"], 2)

#     # Clean human-like output
#     if priority in ["Very High", "High"]:
#         action = "Call immediately and prioritize this lead."
#     elif priority in ["Medium", "Lower-Medium"]:
#         action = "Follow up within 24 hours and nurture."
#     else:
#         action = "Low priority. Add to drip campaign or minimal follow-up."

#     return {
#         "lead_id": lead["LeadID"],
#         "priority": priority,
#         "cltv": cltv,
#         "action": action
#     }

# # --- API endpoint ---
# @app.post("/evaluate")
# def evaluate(query: Query):
#     lead_id = extract_lead_id(query.text)

#     if not lead_id:
#         return {"error": "No valid Lead ID found"}

#     lead = get_lead_data(lead_id)

#     if not lead:
#         return {"error": "Lead not found"}

#     return generate_response(lead)



# LLM Implementation


from fastapi import FastAPI
import pandas as pd
import os
import re

app = FastAPI()

# ----------------------------
# LOAD DATA
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
file_path = os.path.join(BASE_DIR, "data", "processed", "new_leads_with_cltv.csv")
df = pd.read_csv(file_path)


# ----------------------------
# EXTRACT LEAD ID
# ----------------------------
def extract_lead_id(text):
    match = re.search(r"(Lead_\d+)", text, re.IGNORECASE)
    return match.group() if match else None


# ----------------------------
# GET LEAD DATA
# ----------------------------
def get_lead_data(lead_id):
    lead = df[df["LeadID"] == lead_id]
    if lead.empty:
        return None
    return lead.to_dict(orient="records")[0]


# ----------------------------
# BUSINESS LOGIC ENGINE
# ----------------------------
def generate_strategy(priority, cltv):

    if priority in ["Very High", "High"]:
        return {
            "action": "Assign immediately to senior sales agent",
            "steps": [
                "Contact within 1 hour",
                "Provide premium financial consultation",
                "Focus on conversion urgency"
            ]
        }

    elif priority in ["Medium", "Lower-Medium"]:
        return {
            "action": "Nurture lead with structured follow-up",
            "steps": [
                "Follow up within 24–48 hours",
                "Send tailored product information",
                "Schedule call if engagement increases"
            ]
        }

    else:
        return {
            "action": "Low priority monitoring only",
            "steps": [
                "Add to automated email drip campaign",
                "One light follow-up touchpoint",
                "No senior agent assignment"
            ]
        }


# ----------------------------
# MAIN API ENDPOINT
# ----------------------------
@app.post("/evaluate")
def evaluate_lead(payload: dict):

    text = payload.get("query", "")
    lead_id = extract_lead_id(text)

    if not lead_id:
        return {"error": "No valid Lead ID found"}

    lead = get_lead_data(lead_id)

    if not lead:
        return {"error": "Lead not found"}

    priority = lead["Lead_Priority"].replace(" Priority", "")
    cltv = round(float(lead["Predicted_CLTV"]), 2)

    strategy = generate_strategy(priority, cltv)

    return {
        "lead_id": lead_id,
        "priority": priority,
        "cltv": cltv,
        "ai_strategy": strategy
    }


from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="src/api/static", html=True), name="static")