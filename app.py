from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# -------------------------
# Create FastAPI app
# -------------------------
app = FastAPI()

# -------------------------
# Load Model (only once)
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Load SAC Master Data
# -------------------------
sac = pd.read_excel(
    "Service items for SAC Update.xlsx",
    sheet_name="SAC STD",
    engine="openpyxl"
)

# Create embeddings once
sac_embeddings = model.encode(sac["SAC Description"].tolist())

# -------------------------
# Request Body Model
# -------------------------
class Item(BaseModel):
    itemdesc: str

# -------------------------
# Health Check
# -------------------------
@app.get("/")
def home():
    return {"message": "SAC Prediction API running"}

# -------------------------
# Prediction Endpoint
# -------------------------
@app.post("/predict")
def predict_sac(item: Item):

    input_embedding = model.encode([item.itemdesc])
    similarity = cosine_similarity(input_embedding, sac_embeddings)

    best_match_index = np.argmax(similarity)

    matched_sac_code = sac.iloc[best_match_index]["SAC Code"]
    matched_sac_desc = sac.iloc[best_match_index]["SAC Description"]

    return {
        "Matched_SAC_Code": str(matched_sac_code),
        "SAC_Description": matched_sac_desc
    }