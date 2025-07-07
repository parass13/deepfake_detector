import streamlit as st

def show_user_guide():
    st.title("🧭 User Guide")
    st.markdown("Follow these simple steps to use the Deepfake Detection System:")

    with st.container(border=True):
        st.markdown("### 🔐 Step 1: Register or Log In")
        st.write("""
        - If you're a new user, click **Sign Up** and provide your Name, Email, Phone, and Password.
        - Existing users can directly **Log In** using their registered credentials.
        """)

    with st.container(border=True):
        st.markdown("### 📷 Step 2: Upload an Image")
        st.write("""
        - Once logged in, you'll be taken to the **Detection Panel**.
        - Upload any image you suspect might be fake or tampered with.
        - Supported formats: JPG, PNG, JPEG.
        """)

    with st.container(border=True):
        st.markdown("### 🧠 Step 3: Get Deepfake Detection Results")
        st.write("""
        - Click the **Predict** button.
        - The system uses a trained Deep Learning model to analyze the image.
        - You'll see either:
            - ✅ **Real Image** – if the image appears authentic, or
            - ⚠️ **Fake Image** – if the model detects manipulation.
        """)

    with st.container(border=True):
        st.markdown("### 🔓 Step 4: Log Out")
        st.write("When you're done, just click the **Logout** button to securely end your session.")
