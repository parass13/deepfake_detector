import gradio as gr
import re
import bcrypt
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


import os
import warnings
import requests
import json


from pages import about, community, user_guide

# --- Config ---
SUPABASE_URL = "YOUR_URL"
SUPABASE_API_KEY = "YOUR_KEY"
SUPABASE_TABLE = "table_name"

headers = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}",
    "Content-Type": "application/json"
}

# --- Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

MODEL_PATH = "model_15_64.h5"
if not os.path.exists(MODEL_PATH):
    print(f"Model file '{MODEL_PATH}' not found. Creating a dummy model for testing.")
    dummy_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dummy_model.save(MODEL_PATH)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    deepfake_model = tf.keras.models.load_model(MODEL_PATH)

# --- Helpers ---
def is_valid_email(email): return re.match(r"[^@]+@[^@]+\.[^@]+", email)
def is_valid_phone(phone): return re.match(r"^[0-9]{10}$", phone)

def preprocess_image(image):
    if image.mode != 'RGB': image = image.convert('RGB')
    image_arr = np.array(image)
    image_arr = cv2.resize(image_arr, (128, 128))
    image_arr = image_arr.astype(np.float32) / 255.0
    return np.expand_dims(image_arr, axis=0)

def predict_image(image):
    if image is None: return "Please upload an image first."
    preprocessed = preprocess_image(image)
    prediction = deepfake_model.predict(preprocessed)[0][0]
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    label = "✅ Real Image" if prediction >= 0.5 else "⚠️ Fake Image"
    return f"{label} (Confidence: {confidence:.2%})"

def register_user(name, phone, email, gender, password):
    if not all([name, phone, email, gender, password]):
        return "❌ All fields are required for signup."
    if not is_valid_email(email): return "❌ Invalid email format."
    if not is_valid_phone(phone): return "❌ Phone must be 10 digits."

    query_url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?email=eq.{email}"
    r = requests.get(query_url, headers=headers)
    if r.status_code == 200 and len(r.json()) > 0:
        return "⚠️ Email already registered."

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode()
    data = {
        "name": name,
        "phone": phone,
        "email": email,
        "gender": gender,
        "password": hashed_pw
    }
    r = requests.post(f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}", headers=headers, data=json.dumps(data))
    return "✅ Registration successful! Please log in." if r.status_code == 201 else "❌ Error during registration."

def login_user(email, password):
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?email=eq.{email}"
    r = requests.get(url, headers=headers)
    if r.status_code == 200 and r.json():
        stored_hash = r.json()[0]["password"]
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    return False

# --- UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="VerifiAI - Deepfake Detector") as demo:
    is_logged_in = gr.State(False)

    HOME_TAB_NAME = "🏠 Home"
    LOGIN_TAB_NAME = "🔐 Login"
    DETECT_TAB_NAME = "🧪 Detect Deepfake"
    ABOUT_TAB_NAME = "ℹ️ About"
    COMMUNITY_TAB_NAME = "🌐 Community"
    GUIDE_TAB_NAME = "📘 User Guide"

    with gr.Tabs(selected=HOME_TAB_NAME) as tabs:
        # --- Home Tab (Intro Page) ---
        with gr.Tab(HOME_TAB_NAME) as home_tab:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    <div class="home-content">
                        <h1>👁️‍🗨️ Welcome to VerifiAI</h1>
                        <p>Your trusted assistant for detecting deepfakes in images using AI.</p>
                        <p>🔍 Upload images, analyze authenticity, and learn how deepfakes work.</p>
                        <p>👉 Use the tabs above to get started.</p>
                    </div>
                    """, elem_id="home-markdown")

        with gr.Tab(LOGIN_TAB_NAME) as login_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Welcome!", "Login to access the detector, or sign up for a new account.")
                with gr.Column(scale=2):
                    gr.Markdown("### Login or Sign Up")
                    message_output = gr.Markdown(visible=False)
                    email_login = gr.Textbox(label="Email")
                    password_login = gr.Textbox(label="Password", type="password")
                    login_btn = gr.Button("Login", variant="primary")
                    with gr.Accordion("New User? Click here to Sign Up", open=False) as signup_accordion:
                        name_signup = gr.Textbox(label="Name")
                        phone_signup = gr.Textbox(label="Phone (10 digits)")
                        email_signup = gr.Textbox(label="Email")
                        gender_signup = gr.Dropdown(label="Gender", choices=["Male", "Female", "Other"])
                        password_signup = gr.Textbox(label="Create Password", type="password")
                        signup_btn = gr.Button("Sign Up")

        with gr.Tab(DETECT_TAB_NAME, visible=False) as detect_tab:
            with gr.Row():
                gr.Markdown("## Deepfake Detector")
                logout_btn = gr.Button("Logout")
            with gr.Row():
                image_input = gr.Image(type="pil", label="Upload Image", scale=1)
                with gr.Column(scale=1):
                    result = gr.Textbox(label="Prediction Result", interactive=False)
                    predict_btn = gr.Button("Predict", variant="primary")

        with gr.Tab(ABOUT_TAB_NAME): about.layout()
        with gr.Tab(COMMUNITY_TAB_NAME): community.layout()
        with gr.Tab(GUIDE_TAB_NAME): user_guide.layout()
       



    # --- CSS Styling ---
    gr.HTML("""
    <style>
    #home-markdown {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 70vh;
        text-align: center;
    }
    </style>
    """)

    def update_ui_on_auth_change(logged_in_status):
        if logged_in_status:
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(selected=DETECT_TAB_NAME),
                gr.update(value="✅ Login successful!", visible=True)
            )
        else:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(selected=LOGIN_TAB_NAME),
                gr.update(value="", visible=False)
            )

    def handle_login(email, password):
        if login_user(email, password):
            return True, gr.update(value="✅ Login successful!", visible=True)
        else:
            return False, gr.update(value="❌ Invalid email or password.", visible=True)

    def handle_logout():
        return False, "", ""

    def handle_signup(name, phone, email, gender, password):
        msg = register_user(name, phone, email, gender, password)
        if msg.startswith("✅"):
            return gr.update(value=msg, visible=True), "", "", "", "", "", gr.update(open=False)
        else:
            return gr.update(value=msg, visible=True), name, phone, email, gender, password, gr.update(open=True)


    

    
    
  


    login_btn.click(fn=handle_login, inputs=[email_login, password_login], outputs=[is_logged_in, message_output])
    logout_btn.click(fn=handle_logout, inputs=[], outputs=[is_logged_in, email_login, password_login])
    is_logged_in.change(fn=update_ui_on_auth_change, inputs=is_logged_in, outputs=[login_tab, detect_tab, tabs, message_output])
    signup_btn.click(
        fn=handle_signup,
        inputs=[name_signup, phone_signup, email_signup, gender_signup, password_signup],
        outputs=[message_output, name_signup, phone_signup, email_signup, gender_signup, password_signup, signup_accordion]
    )
    predict_btn.click(fn=predict_image, inputs=image_input, outputs=result)

if __name__ == "__main__":
    demo.launch()
