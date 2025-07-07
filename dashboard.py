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

# Suppress TensorFlow and warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

# Load TensorFlow model
deepfake_model = tf.keras.models.load_model("model_15_64.h5")

# Setup SQLite database
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

# Prediction
def predict_image(image):
    preprocessed = preprocess_image(image)
    prediction = deepfake_model.predict(preprocessed)[0][0]
    return "✅ Real Image" if prediction >= 0.5 else "⚠️ Fake Image"

# Register logic
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
    return "✅ Registration successful! Please log in."

# Login logic
def login_user(email, password):
    cursor.execute("SELECT PASSWORD FROM user_details WHERE EMAIL = ?", (email,))
    result = cursor.fetchone()
    if result and bcrypt.checkpw(password.encode(), result[0] if isinstance(result[0], bytes) else result[0].encode()):
        return True
    return False

# Gradio UI
with gr.Blocks() as demo:
    is_logged_in = gr.State(False)
    active_tab = gr.State(0)  # To manage which tab is active

    with gr.Tabs():
        # Login Tab (Visible by default)
        with gr.Tab("🔐 Login") as login_tab:
            gr.Markdown("### Login or Sign Up")
            name = gr.Textbox(label="Name (Sign Up Only)")
            phone = gr.Textbox(label="Phone (Sign Up Only)")
            email = gr.Textbox(label="Email")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            signup_btn = gr.Button("Sign Up")
            message_output = gr.Markdown("", visible=False)

        # Detect Deepfake Tab (Initially Hidden)
        with gr.Tab("🧪 Detect Deepfake", visible=False) as detect_tab:
            gr.Markdown("### Upload an Image to Detect Deepfake")
            image_input = gr.Image(type="pil")
            result = gr.Textbox(label="Prediction Result")
            predict_btn = gr.Button("Predict")
            predict_btn.click(fn=predict_image, inputs=image_input, outputs=result)

            # Logout button inside the Detect tab
            logout_btn = gr.Button("Logout", visible=False)
        
        # About, Community, User Guide (Always visible)
        with gr.Tab("ℹ️ About"):
            about.layout()

        with gr.Tab("🌐 Community"):
            community.layout()

        with gr.Tab("📘 User Guide"):
            user_guide.layout()

    # Handlers
    def handle_login(email, password):
        success = login_user(email, password)
        if success:
            return (
                "✅ Login successful!",  # Success message
                True,  # Update logged-in state
                gr.update(visible=False),  # Hide Login tab
                gr.update(visible=True),  # Show Detect tab
                gr.update(visible=True),  # Show Logout button
                gr.update(visible=True),  # Show Detect Deepfake tab
            )
        return (
            "❌ Invalid credentials",  # Error message
            False,  # Keep logged-in state False
            gr.update(visible=True),  # Keep Login tab visible
            gr.update(visible=False),  # Hide Detect tab
            gr.update(visible=False),  # Hide Logout button
            gr.update(visible=False),  # Hide Detect Deepfake tab
        )
    
    def handle_signup(name, phone, email, password):
        msg = register_user(name, phone, email, password)
        return gr.update(value=msg, visible=True)
    
    def handle_logout():
        return (
            False,  # User logs out
            gr.update(visible=True),  # Show Login tab again
            gr.update(visible=False),  # Hide Detect tab
            gr.update(visible=False),  # Hide Logout button
            gr.update(visible=False),  # Hide Detect Deepfake tab
        )

    # Button clicks
    login_btn.click(
        fn=handle_login,
        inputs=[email, password],
        outputs=[message_output, is_logged_in, login_tab, detect_tab, logout_btn, detect_tab]
    )
    
    signup_btn.click(
        fn=handle_signup,
        inputs=[name, phone, email, password],
        outputs=[message_output]
    )

    logout_btn.click(
        fn=handle_logout,
        inputs=[],
        outputs=[is_logged_in, login_tab, detect_tab, logout_btn, detect_tab]
    )

if __name__ == "__main__":
    demo.launch()
