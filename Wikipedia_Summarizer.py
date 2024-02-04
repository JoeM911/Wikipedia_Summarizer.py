import streamlit as st
from bs4 import BeautifulSoup
from transformers import pipeline
import requests

def main():
     st.title("Wikipedia Summarizer")


URL = st.text_input("Enter URL:")
if st.button("Summarize"):
     process_url(URL)

def process_url(URL):
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

#Title Detection
Title = soup.find('title').get_text()
st.header(f"Title: {Title}")

#Body Paragraph Detection
Body_Parag = soup.find_all('p')
st.subheader("Body Paragraphs:")
for paragraph in Body_Parag:
    st.write(paragraph.text)

article_text = ''.join([paragraph.text for paragraph in Body_Parag])
#Splitting into 512-character segments
max_length = 512
segments = [article_text[i:i + max_length] for i in range(0, len(article_text), max_length)]
#Summarizer Tool
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

st.subheader('Summary of Link:')
word_count = 0
for segment in segments:
        summary = summarizer(segment, max_length=100, min_length=30, do_sample=False)
        summary_text = summary[0]['summary_text']
        word_count += len(summary_text.split())
        st.write(summary_text)
        if word_count >= 50:
            break

if __name__ == "__main__": 
    main()
