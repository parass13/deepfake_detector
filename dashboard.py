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

# Import pages (make sure each page has layout() function defined)
import home
import about
import examples
import community
import user_guide
import install

# Suppress logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

# Load TensorFlow deepfake model
deepfake_model = tf.keras.models.load_model("model_15_64.h5")

# Setup SQLite database
db_path = os.path.abspath("users.db")
print(f"✅ Using database at: {db_path}")
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

# Create table if it doesn't exist
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
        return "✅ Login successful!"
    return "❌ Invalid credentials"

# Gradio App
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("🏠 Home"):
            home.layout()

        with gr.Tab("🔐 Login"):
            gr.Markdown("### Login or Sign Up")

            status = gr.Textbox(label="Status", interactive=False)
            name = gr.Textbox(label="Name (Sign Up Only)")
            phone = gr.Textbox(label="Phone (Sign Up Only)")
            email = gr.Textbox(label="Email")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            signup_btn = gr.Button("Sign Up")

            login_btn.click(fn=login_user, inputs=[email, password], outputs=status)
            signup_btn.click(fn=register_user, inputs=[name, phone, email, password], outputs=status)

        with gr.Tab("🧪 Detect Deepfake"):
            gr.Markdown("### Upload an Image to Detect Deepfake")
            image_input = gr.Image(type="pil")
            result = gr.Textbox(label="Prediction Result")
            predict_btn = gr.Button("Predict")
            predict_btn.click(fn=predict_image, inputs=image_input, outputs=result)

        with gr.Tab("📂 Examples"):
            examples.layout()

        with gr.Tab("ℹ️ About"):
            about.layout()

        with gr.Tab("🌐 Community"):
            community.layout()

        with gr.Tab("📘 User Guide"):
            user_guide.layout()

        with gr.Tab("⚙️ Install"):
            install.layout()

# Launch App
if __name__ == "__main__":
    demo.launch()
