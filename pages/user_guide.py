import gradio as gr

def layout():
    with gr.Blocks() as user_guide:
        gr.Markdown("## 🧭 User Guide")
        gr.Markdown("Follow these simple steps to use the Deepfake Detection System:")

        with gr.Group():
            gr.Markdown("### 🔐 Step 1: Register or Log In")
            gr.Markdown("""
            - If you're a new user, click **Sign Up** and provide your Name, Email, Phone, and Password.  
            - Existing users can directly **Log In** using their registered credentials.
            """)

        with gr.Group():
            gr.Markdown("### 📷 Step 2: Upload an Image")
            gr.Markdown("""
            - Once logged in, you'll be taken to the **Detection Panel**.  
            - Upload any image you suspect might be fake or tampered with.  
            - Supported formats: JPG, PNG, JPEG.
            """)

        with gr.Group():
            gr.Markdown("### 🧠 Step 3: Get Deepfake Detection Results")
            gr.Markdown("""
            - Click the **Predict** button.  
            - The system uses a trained Deep Learning model to analyze the image.  
            - You'll see either:  
              - ✅ **Real Image** – if the image appears authentic, or  
              - ⚠️ **Fake Image** – if the model detects manipulation.
            """)

        with gr.Group():
            gr.Markdown("### 🔓 Step 4: Log Out")
            gr.Markdown("When you're done, just click the **Logout** button to securely end your session.")
    
    return user_guide
