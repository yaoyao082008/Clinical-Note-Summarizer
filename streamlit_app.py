import streamlit as st
from io import StringIO
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv("keys.env")

with open("context.txt",'r') as c:
    context = c.read()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature=0
)

messages = [(context),()]


st.title("Clinical Note Summarizer")
st.divider()

string_note = st.text_input("Please input the clinical note you would like to summarize:")
Uploaded = st.file_uploader("Or upload the note as a text file",type=['txt'])

if Uploaded is not None:
    # To read file as string:
    stringio = StringIO(Uploaded.getvalue().decode("utf-8"))
    string_note = stringio.read()


generate = st.button("Generate Summary")


if generate and string_note:
    
    messages[1] = string_note
    summary = llm.invoke(messages)
    st.header("AI Summary")
    st.divider()
    st.write(summary.content)

