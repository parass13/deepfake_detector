import gradio as gr
import sqlite3
import re
import bcrypt
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import warnings

from pages import about, community, user_guide

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

# Load model
deepfake_model = tf.keras.models.load_model("model_15_64.h5")

# Database setup
db_path = os.path.abspath("users.db")
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS user_details (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    NAME TEXT,
    PHONE TEXT,
    EMAIL TEXT UNIQUE,
    GENDER TEXT,
    PASSWORD BLOB
)
''')
conn.commit()
conn.close()

# Validators
def is_valid_email(email): return re.match(r"[^@]+@[^@]+\.[^@]+", email)
def is_valid_phone(phone): return re.match(r"^[0-9]{10}$", phone)

# Image preprocessing
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def predict_image(image):
    preprocessed = preprocess_image(image)
    prediction = deepfake_model.predict(preprocessed)[0][0]
    return "✅ Real Image" if prediction >= 0.5 else "⚠️ Fake Image"

# Auth logic (now using local DB connection inside each function)
def register_user(name, phone, email, password):
    if not is_valid_email(email):
        return "❌ Invalid email"
    if not is_valid_phone(phone):
        return "❌ Phone must be 10 digits"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_details WHERE EMAIL = ?", (email,))
    if cursor.fetchone():
        conn.close()
        return "⚠️ Email already registered"

    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    cursor.execute(
        "INSERT INTO user_details (NAME, PHONE, EMAIL, GENDER, PASSWORD) VALUES (?, ?, ?, ?, ?)",
        (name, phone, email, "U", hashed_pw)
    )
    conn.commit()
    conn.close()
    return "✅ Registration successful! Please log in."

def login_user(email, password):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT PASSWORD FROM user_details WHERE EMAIL = ?", (email,))
    result = cursor.fetchone()
    conn.close()
    if result and bcrypt.checkpw(password.encode(), result[0] if isinstance(result[0], bytes) else result[0].encode()):
        return True
    return False

# Gradio UI
with gr.Blocks() as demo:
    is_logged_in = gr.State(False)

    with gr.Tabs(selected=0) as tabs:
        login_tab = gr.Tab("🔐 Login")
        detect_tab = gr.Tab("🧪 Detect Deepfake", visible=False)

        with login_tab:
            gr.Markdown("### Login or Sign Up")
            name = gr.Textbox(label="Name (Sign Up Only)")
            phone = gr.Textbox(label="Phone (Sign Up Only)")
            email = gr.Textbox(label="Email")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            signup_btn = gr.Button("Sign Up")
            message_output = gr.Markdown("", visible=False)

        with detect_tab:
            gr.Markdown("### Upload an Image to Detect Deepfake")
            image_input = gr.Image(type="pil")
            result = gr.Textbox(label="Prediction Result")
            predict_btn = gr.Button("Predict")
            predict_btn.click(fn=predict_image, inputs=image_input, outputs=result)
            logout_btn = gr.Button("Logout")

        with gr.Tab("ℹ️ About"):
            about.layout()

        with gr.Tab("🌐 Community"):
            community.layout()

        with gr.Tab("📘 User Guide"):
            user_guide.layout()

    # --- Backend Logic ---
    def handle_login(email, password):
        success = login_user(email, password)
        if success:
            return (
                "✅ Login successful!",
                True,
                gr.update(visible=False),  # Hide login tab
                gr.update(visible=True),   # Show detect tab
                gr.update(selected=1),     # Switch to detect tab
                gr.update(visible=True),   # Show logout
            )
        return (
            "❌ Invalid credentials",
            False,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(selected=0),
            gr.update(visible=False),
        )

    def handle_signup(name, phone, email, password):
        msg = register_user(name, phone, email, password)
        return gr.update(value=msg, visible=True)

    def handle_logout():
        return (
            False,
            gr.update(visible=True),   # Show login tab
            gr.update(visible=False),  # Hide detect tab
            gr.update(selected=0),     # Go back to login tab
            gr.update(visible=False),  # Hide logout
        )

    login_btn.click(
        fn=handle_login,
        inputs=[email, password],
        outputs=[message_output, is_logged_in, login_tab, detect_tab, tabs, logout_btn]
    )

    signup_btn.click(
        fn=handle_signup,
        inputs=[name, phone, email, password],
        outputs=[message_output]
    )

    logout_btn.click(
        fn=handle_logout,
        inputs=[],
        outputs=[is_logged_in, login_tab, detect_tab, tabs, logout_btn]
    )

if __name__ == "__main__":
    demo.launch()
