import gradio as gr
import requests
import json
import datetime

SUPABASE_URL = "YOUR_URL"
SUPABASE_API_KEY = "YOUR_KEY"
headers = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}",
    "Content-Type": "application/json"
}

def submit_feedback(name, email, rating, comments):
    if not all([name, email, rating, comments]):
        return "❌ All fields are required.", name, email, rating, comments

    data = {
        "name": name,
        "email": email,
        "rating": rating,
        "comments": comments,
        "submitted": datetime.datetime.now().isoformat()
    }

    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/feedback",
        headers=headers,
        data=json.dumps(data)
    )

    if response.status_code == 201:
        return (
            "✅ Feedback submitted successfully!",
            "", "", 3, ""  # Reset form fields
        )
    else:
        return (
            "❌ Failed to submit feedback. Please try again.",
            name, email, rating, comments
        )

def layout():
    with gr.Column():
        gr.Markdown("## 🌐 Join the Community")

        gr.Markdown("""
Deepfakes are becoming increasingly sophisticated. We believe that fighting misinformation is a community effort.
### 🤝 How You Can Contribute
- **Share your feedback** on the tool’s performance  
- **Report suspicious media** or share verified datasets  
- **Suggest improvements** to the detection model  
- **Educate others** on recognizing and avoiding deepfake scams  
### 💬 Let’s Talk
Join our open discussions and connect with developers, researchers, and digital safety advocates.  
Whether you're a student, developer, or just curious — your voice matters.
        """)

        gr.Markdown("### 📝 Submit Feedback")

        with gr.Row():
            name = gr.Textbox(label="Name")
            email = gr.Textbox(label="Email")

        with gr.Row():
            rating = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Rating", interactive=True)

        comments = gr.Textbox(label="Comments", lines=3, max_lines=4, placeholder="Let us know what you think...")

        submit_btn = gr.Button("Submit", variant="primary")
        response_msg = gr.Markdown("")

        submit_btn.click(
            fn=submit_feedback,
            inputs=[name, email, rating, comments],
            outputs=[response_msg, name, email, rating, comments]
        )
