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

# Suppress all warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

# Load TensorFlow model
deepfake_model = tf.keras.models.load_model("model_15_64.h5")

# Setup SQLite connection
db_path = os.path.abspath("users.db")
print(f"✅ Using database at: {db_path}")
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

# Create user_details table if not exists
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

# Validation functions
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

# Deepfake prediction
def predict_image(image):
    preprocessed = preprocess_image(image)
    prediction = deepfake_model.predict(preprocessed)[0][0]
    return "✅ Real Image" if prediction >= 0.5 else "⚠️ Fake Image"

# Register new user
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
    return "✅ Registration successful! Please log in.", True

# Login user
def login_user(email, password):
    cursor.execute("SELECT PASSWORD FROM user_details WHERE EMAIL = ?", (email,))
    result = cursor.fetchone()
    if result and bcrypt.checkpw(password.encode(), result[0] if isinstance(result[0], bytes) else result[0].encode()):
        return "✅ Login successful!", True
    return "❌ Invalid credentials", False

# Gradio Interface
with gr.Blocks() as demo:
    session = gr.State({})
    show_login = gr.State(True)

    status = gr.Textbox(label="", interactive=False)

    with gr.Column(visible=True) as login_panel:
        gr.Markdown("### Login or Sign Up")
        name = gr.Textbox(label="Name (Sign Up Only)")
        phone = gr.Textbox(label="Phone (Sign Up Only)")
        email = gr.Textbox(label="Email")
        password = gr.Textbox(label="Password", type="password")

        login_btn = gr.Button("Login")
        signup_btn = gr.Button("Sign Up")

    with gr.Column(visible=False) as prediction_panel:
        gr.Markdown("## Upload Image for Deepfake Detection")
        image_input = gr.Image(type="pil")
        result = gr.Textbox(label="Result")
        predict_btn = gr.Button("Predict")
        logout_btn = gr.Button("Logout")

    # Logic
    def handle_login(e, p):
        msg, ok = login_user(e, p)
        return msg, gr.update(visible=not ok), gr.update(visible=ok)

    def handle_signup(n, ph, e, p):
        msg, ok = register_user(n, ph, e, p)
        return msg, gr.update(visible=not ok), gr.update(visible=ok)

    def handle_logout():
        return {}, gr.update(visible=True), gr.update(visible=False)

    login_btn.click(handle_login, [email, password], [status, login_panel, prediction_panel])
    signup_btn.click(handle_signup, [name, phone, email, password], [status, login_panel, prediction_panel])
    predict_btn.click(predict_image, inputs=image_input, outputs=result)
    logout_btn.click(handle_logout, outputs=[session, login_panel, prediction_panel])

# Launch app
if __name__ == "__main__":
    demo.launch()
