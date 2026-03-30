import streamlit as st
import requests

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="SAC Code Predictor", layout="centered")

st.title("🔍 SAC Code Prediction System")

st.write("Enter Item Description to get SAC Code")

# -----------------------------
# Input Box
# -----------------------------
item_description = st.text_area("Item Description")

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict SAC Code"):

    if item_description.strip() == "":
        st.warning("Please enter item description")
    else:
        try:
            url = "http://127.0.0.1:8000/predict"

            data = {
                "itemdesc": item_description
            }

            response = requests.post(url, json=data)

            if response.status_code == 200:
                result = response.json()

                st.success("Prediction Successful ✅")
                st.write("### Matched SAC Code:")
                st.write(result.get("Matched_SAC_Code"))

                if "SAC_Description" in result:
                    st.write("### SAC Description:")
                    st.write(result.get("SAC_Description"))

                if "Confidence" in result:
                    st.write("### Confidence Score:")
                    st.write(result.get("Confidence"))

            else:
                st.error("API Error")

        except Exception as e:
            st.error("Cannot connect to API. Make sure FastAPI is running.")