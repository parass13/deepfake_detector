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

# Import pages
from pages import about, community, user_guide

# Suppress logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

# Load deepfake model
deepfake_model = tf.keras.models.load_model("model_15_64.h5")

# SQLite setup
db_path = os.path.abspath("users.db")
print(f"✅ Using database at: {db_path}")
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

# Utilities
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def is_valid_phone(phone):
    return re.match(r"^[0-9]{10}$", phone)

def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def predict_image(image):
    preprocessed = preprocess_image(image)
    prediction = deepfake_model.predict(preprocessed)[0][0]
    return "✅ Real Image" if prediction >= 0.5 else "⚠️ Fake Image"

def register_user(name, phone, email, password):
    if not is_valid_email(email):
        return "❌ Invalid email", False
    if not is_valid_phone(phone):
        return "❌ Phone must be 10 digits", False

    cursor.execute("SELECT * FROM user_details WHERE EMAIL = ?", (email,))
    if cursor.fetchone():
        return "⚠️ Email already registered", False

    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    cursor.execute("INSERT INTO user_details (NAME, PHONE, EMAIL, GENDER, PASSWORD) VALUES (?, ?, ?, ?, ?)",
                   (name, phone, email, "U", hashed_pw))
    conn.commit()
    print(f"✅ Registered new user: {email}")
    return "✅ Registration successful! Please log in.", False

def login_user(email, password):
    cursor.execute("SELECT PASSWORD FROM user_details WHERE EMAIL = ?", (email,))
    result = cursor.fetchone()
    if result and bcrypt.checkpw(password.encode(), result[0]):
        return "✅ Login successful!", True
    return "❌ Invalid credentials", False


# Gradio App
with gr.Blocks() as demo:
    session = gr.State(value=False)  # Stores login state (True/False)

    with gr.Tabs() as tabs:
        with gr.Tab("🔐 Login"):
            gr.Markdown("### Login or Sign Up")

            status = gr.Textbox(label="Status", interactive=False)
            name = gr.Textbox(label="Name (Sign Up Only)")
            phone = gr.Textbox(label="Phone (Sign Up Only)")
            email = gr.Textbox(label="Email")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            signup_btn = gr.Button("Sign Up")

            def handle_login(e, p):
                msg, ok = login_user(e, p)
                return msg, ok

            def handle_signup(n, ph, e, p):
                msg, ok = register_user(n, ph, e, p)
                return msg, ok

            login_btn.click(handle_login, [email, password], [status, session])
            signup_btn.click(handle_signup, [name, phone, email, password], [status, session])

        with gr.Tab("🧪 Detect Deepfake") as detect_tab:
            with gr.Column(visible=False) as detection_content:
                gr.Markdown("### Upload an Image to Detect Deepfake")
                image_input = gr.Image(type="pil")
                result = gr.Textbox(label="Prediction Result")
                predict_btn = gr.Button("Predict")
                predict_btn.click(fn=predict_image, inputs=image_input, outputs=result)

            # Show warning if not logged in
            with gr.Column(visible=True) as login_prompt:
                warning_text = gr.Markdown("⚠️ Please login or sign up to access deepfake detection.")

        def toggle_tab(logged_in):
            return (
                gr.update(visible=logged_in),  # detection_content
                gr.update(visible=not logged_in),  # login_prompt
            )

        # Toggle detection tab visibility based on login state
        session.change(fn=toggle_tab, inputs=session, outputs=[detection_content, login_prompt])

        with gr.Tab("ℹ️ About"):
            about.layout()

        with gr.Tab("🌐 Community"):
            community.layout()

        with gr.Tab("📘 User Guide"):
            user_guide.layout()

# Launch App
if __name__ == "__main__":
    demo.launch()
