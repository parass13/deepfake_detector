import gradio as gr

def layout():
    return gr.Column([
        gr.Markdown("## What is a Deepfake?"),
        gr.Markdown(
            "A **deepfake** is a type of synthetic media generated using deep learning techniques, "
            "particularly deep neural networks. The term **'deepfake'** is a combination of *'deep learning'* and *'fake'*."
            "\n\nIn deepfake technology, algorithms are used to create or manipulate audio, video, or images to depict something "
            "that did not actually occur or that alters the appearance or actions of individuals. This can involve superimposing "
            "images or videos of people onto existing footage, making individuals appear to say or do things they never said or did."
        ),
        gr.Markdown("## Real Cases"),
        gr.Markdown("**Kerala Man Loses Rs 40,000 to AI-Based Deepfake WhatsApp Fraud**"),
        gr.Markdown(
            "A man in Kerala lost Rs 40,000 in an online scam on WhatsApp involving AI-based deepfake technology. "
            "The scammer impersonated the victim's former colleague via video call, fabricating a medical emergency and requesting money. "
            "This incident underscores the danger of sophisticated online fraud using deepfake technology and emphasizes the importance "
            "of verifying unexpected financial requests to avoid falling victim to such scams."
        )
    ])
