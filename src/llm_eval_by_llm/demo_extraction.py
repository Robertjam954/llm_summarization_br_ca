from enum import Enum
from typing import Union
import openai

product_search_prompt = """
    Template for Summarizing Breast Cancer Workup for surgical consultation

Instruction to LLM: Extract information from radiology and pathology reports according to the structure below. Present findings chronologically.
1. Radiology Reports
Mammogram (Date, screening or diagnostic study)
- Laterality
- Abnormality (e.g., asymmetry, calcifications, distortion, mass)
- Size: cm
- Location: (quadrant, clock face, depth)
- Interval change
- Recommendation
 Ultrasound (Date)
- Laterality: 
- Location: (quadrant, clock face, distance from nipple)
- Lesion size: ___ × ___ × ___ cm
- Morphology: (solid, cystic, complex, irregular, etc.)
- Findings correlate with Mammogram abnormalities
-Lymph node findings 
- Recommendation
MRI (Date)
- Findings Laterality
- Location: (quadrant, clock face, depth)
- Abnormality type
  - Mass (shape, margins, enhancement, size)
  - Non-mass enhancement (distribution, pattern)
- Associated features (skin thickening, nipple retraction, chest wall involvement, multifocality/multicentricity)
- findings correlate with MMG and US abnormalities
-Lymph node findings
- Recommendation
Post-procedure Mammogram (Date)
- Clip placement: (type and location)
- Additional findings: (residual calcifications, migration, post-biopsy change)
2. Pathology Reports
Biopsy (Date)
- Biopsy method: (core, stereotactic, US-guided, MRI-guided, FNA, excision)
- Site: (laterality, quadrant, clock face)
- Lesion size: ___ cm
- Histology (benign, atypical, DCIS, invasive carcinoma subtype)
- Receptor status:
  - ER: ___%
  - PR: ___%
  - HER2: ___ (IHC/FISH result)
  - Ki-67: ___%
- Clip placed (if specified)
3. MSK radiology review comparison
-IF available summarize MSK rad review (identify by “EXAM: REVIEW OF SUBMITTED BREAST IMAGING”)
-Flag discrepancies

4. Final Timeline Summary
Provide a concise chronological narrative of workup progression, If available provide information on what prompted work-up (palpable mass, breast changes, screening)
Screening mammogram on [date] showed  → Diagnostic mammogram/ultrasound confirmed  → Biopsy on [date] showed→ Post-procedure mammogram confirmed clip placement at→ MRI on [date] demonstrated. 

"""


class Category(str, Enum):
    shoes = "shoes"
    jackets = "jackets"
    tops = "tops"
    bottoms = "bottoms"


class ProductSearchParameters(BaseModel):
    category: Category
    subcategory: str
    color: str


def get_response(user_input, context):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": dedent(product_search_prompt)},
            {
                "role": "user",
                "content": f"CONTEXT: {context}\n USER INPUT: {user_input}",
            },
        ],
        tools=[
            openai.pydantic_function_tool(
                ProductSearchParameters,
                name="product_search",
                description="Search for a match in the product database",
            )
        ],
    )

    return response.choices[0].message.tool_calls


example_inputs = [
    {
        "user_input": "I'm looking for a new coat. I'm always cold so please something warm! Ideally something that matches my eyes.",
        "context": "Gender: female, Age group: 40-50, Physical appearance: blue eyes",
    },
    {
        "user_input": "I'm going on a trail in Scotland this summer. It's goind to be rainy. Help me find something.",
        "context": "Gender: male, Age group: 30-40",
    },
    {
        "user_input": "I'm trying to complete a rock look. I'm missing shoes. Any suggestions?",
        "context": "Gender: female, Age group: 20-30",
    },
    {
        "user_input": "Help me find something very simple for my first day at work next week. Something casual and neutral.",
        "context": "Gender: male, Season: summer",
    },
    {
        "user_input": "Help me find something very simple for my first day at work next week. Something casual and neutral.",
        "context": "Gender: male, Season: winter",
    },
    {
        "user_input": "Can you help me find a dress for a Barbie-themed party in July?",
        "context": "Gender: female, Age group: 20-30",
    },
]
