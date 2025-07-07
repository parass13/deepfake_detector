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

# Import content pages
from pages import about, community, user_guide

# Suppress TensorFlow and warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

# Load deepfake model
deepfake_model = tf.keras.models.load_model("model_15_64.h5")

# Database setup
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

# Validators
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def is_valid_phone(phone):
    return re.match(r"^[0-9]{10}$", phone)

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

# Auth logic
def register_user(name, phone, email, password):
    if not is_valid_email(email):
        return "❌ Invalid email"
    if not is_valid_phone(phone):
        return "❌ Phone must be 10 digits"
    cursor.execute("SELECT * FROM user_details WHERE EMAIL = ?", (email,))
    if cursor.fetchone():
        return "⚠️ Email already registered"
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    cursor.execute("INSERT INTO user_details (NAME, PHONE, EMAIL, GENDER, PASSWORD) VALUES (?, ?, ?, ?, ?)",
                   (name, phone, email, "U", hashed_pw))
    conn.commit()
    print(f"✅ Registered new user: {email}")
    return "✅ Registration successful! Please log in."

def login_user(email, password):
    cursor.execute("SELECT PASSWORD FROM user_details WHERE EMAIL = ?", (email,))
    result = cursor.fetchone()
    if result and bcrypt.checkpw(password.encode(), result[0] if isinstance(result[0], bytes) else result[0].encode()):
        return True
    return False

# Gradio UI
with gr.Blocks() as demo:
    is_logged_in = gr.State(False)
    tab_index = gr.State(0)

    with gr.Tabs() as tabs:
        with gr.Tab("🔐 Login"):
            gr.Markdown("### Login or Sign Up")

            name = gr.Textbox(label="Name (Sign Up Only)")
            phone = gr.Textbox(label="Phone (Sign Up Only)")
            email = gr.Textbox(label="Email")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            signup_btn = gr.Button("Sign Up")
            message_output = gr.Markdown("", visible=False)

        with gr.Tab("🧪 Detect Deepfake", visible=False) as detect_tab:
            detect_area = gr.Column(visible=True)
            with detect_area:
                gr.Markdown("### Upload an Image to Detect Deepfake")
                image_input = gr.Image(type="pil")
                result = gr.Textbox(label="Prediction Result")
                predict_btn = gr.Button("Predict")
                predict_btn.click(fn=predict_image, inputs=image_input, outputs=result)
            detect_warning = gr.Markdown("❌ Please log in to use this feature.", visible=False)

        with gr.Tab("ℹ️ About"):
            about.layout()

        with gr.Tab("🌐 Community"):
            community.layout()

        with gr.Tab("📘 User Guide"):
            user_guide.layout()

        with gr.Tab("🚪 Logout", visible=False) as logout_tab:
            gr.Markdown("You are logged in.")
            logout_btn = gr.Button("Logout")

    # Handlers
    def handle_login(email, password):
        success = login_user(email, password)
        return (
            "✅ Login successful!" if success else "❌ Invalid credentials",
            success,
            1 if success else 0,
            gr.update(visible=success),
            gr.update(visible=not success),
            gr.update(visible=success),
            gr.update(visible=True)
        )

    def handle_signup(name, phone, email, password):
        msg = register_user(name, phone, email, password)
        return gr.update(value=msg, visible=True)

    def handle_logout():
        return (
            False,
            0,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    login_btn.click(
        fn=handle_login,
        inputs=[email, password],
        outputs=[message_output, is_logged_in, tab_index, detect_area, detect_warning, logout_tab, message_output]
    )

    signup_btn.click(
        fn=handle_signup,
        inputs=[name, phone, email, password],
        outputs=[message_output]
    )

    logout_btn.click(
        fn=handle_logout,
        inputs=[],
        outputs=[is_logged_in, tab_index, detect_area, detect_warning, logout_tab, message_output]
    )

if __name__ == "__main__":
    demo.launch()
