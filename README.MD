Clinical Note Summarizer

A Streamlit application designed to efficiently summarize clinical notes into concise, digestible summaries using the power of Groq's llama-3.3-70b-versatile model. This tool helps healthcare professionals quickly grasp the essence of lengthy medical documentation, improving workflow and information retrieval.

✨ Features

    Fast Summarization: Leverages Groq's high-speed inference to generate summaries in a couple seconds.

    Concise Output: Summarizes clinical notes into a brief 3-5 sentence format that obtains crucial infromation.

    User-Friendly Interface: Built with Streamlit for an intuitive and easy-to-use web interface.

    Secure API Handling: Integrates with Groq API keys for secure access to the summarization model.

🚀 Installation

To get this application running locally, follow these steps:

    1. Clone the Repo

    2. pip install -r requirements.txt

    3. streamlit run streamlit_app.py

🔑 Configuration

This application requires a Groq API key to function.

    Obtain a Groq API Key:

        Visit the Groq website and sign up for an account.

        Generate a new API key from your dashboard.

    Set the API Key:
    You can set your Groq API key in one of two ways:

        Environment Variable (Recommended for production):

        export GROQ_API_KEY="your_groq_api_key_here"

        (Replace "your_groq_api_key_here" with your actual key.)

        Streamlit Secrets (Recommended for local development):
        Create a folder named .streamlit in your project root, and inside it, create a file named secrets.toml:

        # .streamlit/secrets.toml
        GROQ_API_KEY="your_groq_api_key_here"

🏃‍♀️ Usage

Once you have installed the dependencies and configured your API key, you can run the Streamlit application:

streamlit run app.py

(Assuming your main Streamlit application file is named app.py)

A new tab will open in your web browser, displaying the Clinical Note Summarizer interface.

    Paste your clinical notes into the provided text area.

    Click the "Generate Summary" button.

    The summarized version (3-5 sentences) will appear below.