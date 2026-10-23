import streamlit as st
import requests
from PIL import Image

# API Endpoint Configuration
# API Endpoint Configuration
API_URL = "https://smart-find.onrender.com"

st.set_page_config(
    page_title="SmartFind - Multimodal Lost & Found",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 SmartFind MLOps - Lost & Found Matcher")
st.markdown("Powered by **FastAPI**, **CLIP Multimodal Embeddings**, **MongoDB**, and **MLflow**.")

# Sidebar Navigation
option = st.sidebar.selectbox(
    "Select Action",
    ["Report Found Item", "Report Lost Item & Search Matches"]
)

# ---------------------------------------------------------
# OPTION 1: REPORT FOUND ITEM
# ---------------------------------------------------------
if option == "Report Found Item":
    st.subheader("📦 Report a Found Item")
    st.write("Upload an image and description of the item you found.")

    with st.form("found_form", clear_on_submit=True):
        description = st.text_area("Item Description", placeholder="e.g., Found a navy blue leather wallet near library...")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        submit_btn = st.form_submit_button("Submit Found Report")

    if submit_btn:
        if not description or not uploaded_file:
            st.warning("Please provide both a description and an image.")
        else:
            with st.spinner("Extracting CLIP embeddings and indexing into MongoDB..."):
                files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"description": description}
                
                try:
                    response = requests.post(f"{API_URL}/reports/found", data=data, files=files)
                    if response.status_code == 200:
                        res_json = response.json()
                        st.success(f"Report submitted successfully! Database Record ID: {res_json.get('id')}")
                    else:
                        st.error(f"Failed to submit report. Server returned status code {response.status_code}")
                except Exception as e:
                    st.error(f"Error connecting to FastAPI backend: {e}")

# ---------------------------------------------------------
# OPTION 2: REPORT LOST ITEM & FIND MATCHES
# ---------------------------------------------------------
elif option == "Report Lost Item & Search Matches":
    st.subheader("🔎 Report a Lost Item & Find Matches")
    st.write("Provide details of your lost item to retrieve the Top-5 most similar found reports.")

    col1, col2 = st.columns([1, 1])

    with col1:
        with st.form("lost_form"):
            description = st.text_area("Lost Item Description", placeholder="e.g., Lost my blue wallet containing cards...")
            uploaded_file = st.file_uploader("Upload Image (or similar reference)", type=["jpg", "jpeg", "png"])
            search_btn = st.form_submit_button("Search Matching Found Items")

    if search_btn:
        if not description or not uploaded_file:
            st.warning("Please provide both a description and an image.")
        else:
            with col2:
                st.markdown("### Query Preview")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Lost Item Image", width=250)

            with st.spinner("Generating embeddings & computing cosine similarity..."):
                files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"description": description}

                try:
                    response = requests.post(f"{API_URL}/reports/lost", data=data, files=files)
                    if response.status_code == 200:
                        res_json = response.json()
                        matches = res_json.get("top_5_matches", [])

                        st.markdown("---")
                        st.subheader(f"🎯 Top {len(matches)} Matching Found Items")

                        if not matches:
                            st.info("No matching found items currently in the database.")
                        else:
                            for idx, match in enumerate(matches, 1):
                                with st.container():
                                    m_col1, m_col2 = st.columns([1, 3])
                                    
                                    with m_col1:
                                        # Construct URL to access static upload image from FastAPI
                                        img_path = match.get("image_path", "")
                                        st.image(f"{API_URL}/{img_path}", width=150)

                                    with m_col2:
                                        st.markdown(f"#### Match #{idx} — Score: **{match.get('similarity_score')}%**")
                                        st.write(f"**Description:** {match.get('description')}")
                                        st.caption(f"Report ID: `{match.get('id')}` | Reported At: {match.get('created_at')}")
                                    st.markdown("---")
                    else:
                        st.error(f"Error searching matches. Backend status: {response.status_code}")
                except Exception as e:
                    st.error(f"Connection error: {e}")