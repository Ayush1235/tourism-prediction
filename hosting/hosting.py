from huggingface_hub import HfApi, create_repo
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Define your Space repo_id (not the model repo!)
repo_id = "Ayush12358/tourism-prediction-app"

# Step 1: Create the Space if it doesn’t exist
create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="streamlit",   # required so HF knows it's a Streamlit app
    exist_ok=True
)

# Step 2: Upload your app folder
api.upload_folder(
    folder_path="tourism_project/deployment",  # this folder should contain app.py
    repo_id=repo_id,
    repo_type="space",
    path_in_repo="",   # root of the Space
)

print(f"✅ App uploaded. Visit: https://huggingface.co/spaces/{repo_id}")
