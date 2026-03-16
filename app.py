from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

app = FastAPI()

# Global variables
model = None
sac = None
sac_embeddings = None


# -------------------------
# Request Body
# -------------------------
class Item(BaseModel):
    itemdesc: str


# -------------------------
# Startup event (loads model + data)
# -------------------------
@app.on_event("startup")
def load_resources():
    global model, sac, sac_embeddings

    print("Loading SentenceTransformer model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu"   # Force CPU (important for Render)
    )

    print("Loading SAC Excel data...")

    sac = pd.read_excel(
        "Service items for SAC Update.xlsx",
        sheet_name="SAC STD"
    )

    print("Creating SAC embeddings...")

    sac_embeddings = model.encode(
        sac["SAC Description"].tolist(),
        convert_to_numpy=True
    )

    print("API Ready")


# -------------------------
# Health check endpoint
# -------------------------
@app.get("/")
def home():
    return {"message": "SAC Prediction API Running"}


# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
def predict_sac(item: Item):

    input_embedding = model.encode(
        [item.itemdesc],
        convert_to_numpy=True
    )

    similarity = cosine_similarity(input_embedding, sac_embeddings)

    best_match_index = np.argmax(similarity)

    matched_sac_code = sac.iloc[best_match_index]["SAC Code"]
    matched_sac_desc = sac.iloc[best_match_index]["SAC Description"]

    return {
        "Matched_SAC_Code": str(matched_sac_code),
        "SAC_Description": matched_sac_desc
    }