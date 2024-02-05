import streamlit as st
from bs4 import BeautifulSoup
from transformers import pipeline
import requests

def main():
    st.title("Wikipedia Summarizer")
    URL = st.text_input("Enter URL:")
    Fixed_word_count = st.text_input("Enter Word Count:")

    if st.button("Summarize") and Fixed_word_count.strip():
        process_url(URL, Fixed_word_count)

def process_url(URL, fixed_word_count):
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    # Title Detection
    Title = soup.find('title').get_text()
    st.header(f"Title: {Title}")
    Body_Parag = soup.find_all('p')
    paragraph_text = ''
    for paragraph in Body_Parag:
        paragraph_text += paragraph.get_text() + ' '

    article_text = paragraph_text  # Use the combined paragraph text
    max_length = 512
    segments = [article_text[i:i + max_length] for i in range(0, len(article_text), max_length)]
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")

    # Validate input
    if not fixed_word_count.isdigit():
        st.error("Please enter a valid integer for Word Count.")
        return

    fixed_word_count = int(fixed_word_count)
    st.write(f"Fixed Word Count: {fixed_word_count}")

    current_word_count = 0
    summary_text = ""
    for segment in segments:
        summary = summarizer(segment, max_length=100, min_length=30, do_sample=False)
        current_summary_text = summary[0]['summary_text']
        current_word_count += len(current_summary_text.split())
        if current_word_count <= fixed_word_count:
            summary_text += current_summary_text + " "
        else:
            break
    st.write(summary_text)

if __name__ == "__main__":
    main()
