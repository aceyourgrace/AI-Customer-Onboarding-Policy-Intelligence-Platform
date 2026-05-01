import pandas as pd
import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =========================
# FILE PATH
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
file_path = os.path.join(BASE_DIR, "data", "processed", "new_leads_with_cltv.csv")

df = pd.read_csv(file_path)

# =========================
# MODEL (LOCAL OR HF AUTO)
# =========================
model_name = "microsoft/phi-2"   # lightweight + strong reasoning

print("🚀 Loading model... this may take a bit")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map=None  # CPU safe mode
)

model.to("cpu")

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1
)

print("✅ Model ready")

# =========================
# HELPERS
# =========================
def extract_lead_id(text):
    match = re.search(r"(Lead_\d+)", text, re.IGNORECASE)
    return match.group() if match else None


def get_lead_data(lead_id):
    lead = df[df["LeadID"] == lead_id]
    if lead.empty:
        return None
    return lead.to_dict(orient="records")[0]


# =========================
# LLM ANALYSIS CORE
# =========================
def generate_strategy(lead):
    lead_id = lead["LeadID"]
    priority = lead["Lead_Priority"].replace(" Priority", "")
    cltv = round(lead["Predicted_CLTV"], 2)

    prompt = f"""
You are a BANK SALES STRATEGY ASSISTANT.

You MUST follow this format exactly.

RULES:
- Do NOT repeat instructions
- Do NOT write questions
- Do NOT explain CLTV or priority
- Do NOT add extra sections

OUTPUT FORMAT (STRICT):

Customer Snapshot:
- Lead ID:
- Priority:
- CLTV:

Action Plan:
1.
2.
3.

Sales Approach Tone:
- one sentence only

NOW ANALYZE:

Lead ID: {lead_id}
Priority: {priority}
CLTV: {cltv}

Answer:
"""

    output = text_generator(
        prompt,
        max_new_tokens=120,   # reduce chaos
        temperature=0.2,      # 🔥 key fix (low randomness = discipline)
        do_sample=True
    )[0]["generated_text"]

    response = output.split("Answer:")[-1].strip()

    return response


# =========================
# FINAL RESPONSE FORMAT
# =========================
def generate_response(lead):
    return f"""
========================
📊 LEAD SUMMARY
========================
Lead ID   : {lead['LeadID']}
Priority  : {lead['Lead_Priority'].replace(' Priority','')}
CLTV      : ${round(lead['Predicted_CLTV'], 2)}

========================
🧠 ACTION PLAN (AI)
========================
{generate_strategy(lead)}
"""


# =========================
# MAIN PIPELINE
# =========================
def process_query(text):
    lead_id = extract_lead_id(text)

    if not lead_id:
        return "❌ No valid Lead ID found (try: Lead_001)"

    lead = get_lead_data(lead_id)

    if not lead:
        return "❌ Lead not found in dataset"

    return generate_response(lead)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    user_input = input("Enter your query: ")
    print(process_query(user_input))