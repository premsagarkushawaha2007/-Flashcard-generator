import streamlit as st
import os
import json
import re
import pandas as pd

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pypdf import PdfReader

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="AI Flashcard Generator",
    page_icon="📚",
    layout="wide"
)

# Hugging Face API
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "openai/gpt-oss-120b"

# Page heading
st.title("📚 AI Flashcard Generator")
st.write(
    "Generate smart study flashcards "
    "using Hugging Face AI."
)

# Read PDF
def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# Generate flashcards
def generate_flashcards(topic, notes, count, language):

    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN missing. Check your .env file."
        )

    client = InferenceClient(
        api_key=HF_TOKEN
    )

    prompt = f"""
    You are an expert educational flashcard generator.

    Topic: {topic}
    Study Material: {notes}
    Number of cards: {count}
    Language: {language}

    Generate exactly {count} useful flashcards.

    Return ONLY valid JSON in this format:

    {{
      "flashcards": [
        {{
          "question": "Question here",
          "answer": "Answer here"
        }}
      ]
    }}

    Keep questions and answers clear,
    short and educational.

    Use only the supplied study material
    where study material is provided.
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate accurate educational "
                    "flashcards. Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=3000,
        temperature=0.3
    )

    result = response.choices[0].message.content

    # Extract JSON
    match = re.search(
        r"\{.*\}",
        result,
        re.DOTALL
    )

    if not match:
        raise ValueError(
            "AI did not return valid JSON. Try again."
        )

    data = json.loads(match.group(0))

    cards = data.get("flashcards", [])

    if not cards:
        raise ValueError(
            "No flashcards generated."
        )

    return cards


# Sidebar
st.sidebar.header("⚙️ Settings")

count = st.sidebar.slider(
    "Number of Flashcards",
    min_value=5,
    max_value=20,
    value=10
)

language = st.sidebar.selectbox(
    "Select Language",
    ["English", "Hindi", "Hinglish"]
)

# User input
topic = st.text_input(
    "Enter your topic",
    placeholder="Example: DBMS, Python, AI, Java"
)

notes = st.text_area(
    "Paste your study notes (optional)",
    height=180
)

uploaded_file = st.file_uploader(
    "Upload PDF Notes (optional)",
    type=["pdf"]
)

if uploaded_file:
    pdf_text = read_pdf(uploaded_file)
    notes = pdf_text
    st.success("PDF uploaded successfully!")

# Generate button
if st.button(
    "✨ Generate Flashcards",
    type="primary"
):

    if not topic.strip():
        st.error("Please enter a topic.")

    elif not notes.strip() and not topic.strip():
        st.error("Please enter study material.")

    else:
        try:
            with st.spinner(
                "AI is generating flashcards..."
            ):

                cards = generate_flashcards(
                    topic,
                    notes,
                    count,
                    language
                )

                st.session_state["cards"] = cards

            st.success("Flashcards generated!")

        except Exception as e:
            st.error(f"Error: {e}")


# Display flashcards
if "cards" in st.session_state:

    cards = st.session_state["cards"]

    st.subheader("🎯 Your Flashcards")

    for i, card in enumerate(cards, 1):

        with st.expander(
            f"Card {i}: {card['question']}"
        ):

            st.markdown("**Answer:**")
            st.write(card["answer"])

    # Download CSV
    df = pd.DataFrame(cards)

    csv = df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Download Flashcards CSV",
        data=csv,
        file_name="flashcards.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption(
    "Developed using Python, Streamlit "
    "and Hugging Face AI."
)