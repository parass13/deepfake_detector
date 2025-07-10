import gradio as gr
import requests
import json
import datetime

# --- Supabase Configuration ---
SUPABASE_URL = "YOUR_URL"
SUPABASE_API_KEY = "YOUR_KEY"
SUPABASE_TABLE = "TABLE_NAME"

headers = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}",
    "Content-Type": "application/json"
}

# --- Backend Function ---
def submit_feedback(name, email, rating, comments):
    """Handles the submission of feedback to the Supabase database."""
    if not all([name, email, rating, comments]):
        return "❌ All fields are required.", name, email, rating, comments

    data = {
        "name": name,
        "email": email,
        "rating": rating, # This will be an integer (1-5) from the Radio component
        "comments": comments,
        "submitted": datetime.datetime.now().isoformat()
    }

    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
        headers=headers,
        data=json.dumps(data)
    )

    if response.status_code == 201:
        # On success, return a confirmation message and clear the form fields
        return ("✅ Feedback submitted successfully!", "", "", 3, "")
    else:
        # On failure, return an error and keep the user's input
        return ("❌ Failed to submit feedback. Please try again.", name, email, rating, comments)

# --- UI Layout Function ---
def layout(is_logged_in: gr.State):
    """
    Creates the UI for the Community tab.
    Accepts the global `is_logged_in` state to control form visibility.
    """
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

        # --- Login-Dependent Components ---

        # Message to show when the user is logged OUT
        logged_out_message = gr.Markdown(
            "### <center>Please log in to leave feedback.</center>",
            visible=True 
        )

        # The entire feedback form, visible only when logged IN
        with gr.Group(visible=False) as feedback_form_group:
            gr.Markdown("### 📝 Submit Feedback")
            with gr.Row():
                name = gr.Textbox(label="Name")
                email = gr.Textbox(label="Email")

        
           
            rating = gr.Radio(
                label="Rating",
                choices=[
                    ("⭐", 1),
                    ("⭐⭐", 2),
                    ("⭐⭐⭐", 3),
                    ("⭐⭐⭐⭐", 4),
                    ("⭐⭐⭐⭐⭐", 5)
                ],
                value=3, # Default value is 3
                interactive=True
            )

            comments = gr.Textbox(label="Comments", lines=3, max_lines=4, placeholder="Let us know what you think...")
            submit_btn = gr.Button("Submit", variant="primary")
            response_msg = gr.Markdown()

            submit_btn.click(
                fn=submit_feedback,
                inputs=[name, email, rating, comments],
                outputs=[response_msg, name, email, rating, comments]
            )

        # --- UI Control Logic ---

        def toggle_feedback_visibility(logged_in_status):
            """Shows/hides components based on the login status."""
            return {
                feedback_form_group: gr.update(visible=logged_in_status),
                logged_out_message: gr.update(visible=not logged_in_status)
            }

        # This listener is the key: it triggers the visibility update
        # whenever the is_logged_in state changes anywhere in the app.
        is_logged_in.change(
            fn=toggle_feedback_visibility,
            inputs=is_logged_in,
            outputs=[feedback_form_group, logged_out_message]
        )
