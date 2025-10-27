import streamlit as st
from transformers import pipeline

# --- Load models ---
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return summarizer, qa_model

summarizer, qa_model = load_models()

# --- UI ---
st.title("🧠 Text Summarizer & Question Answering App")
st.write("Paste a long article below, get a summary, and ask any question about it!")

text = st.text_area("📄 Enter your article or text:", height=250)
question = st.text_input("❓ Ask a question based on the text:")

if st.button("Summarize and Answer"):
    if text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

        st.subheader("📝 Summary")
        st.write(summary)

        if question.strip() != "":
            with st.spinner("Finding the answer..."):
                answer = qa_model(question=question, context=text)['answer']

            st.subheader("💡 Answer")
            st.write(answer)
        else:
            st.info("Enter a question to get an answer.")
