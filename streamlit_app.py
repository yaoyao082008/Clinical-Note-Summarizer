import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import pandas

load_dotenv("keys.env")

with open("context.txt",'r') as c:
    context = c.read()

models = {
    "Llama 3":"llama-3.3-70b-versatile",
    "Distill":"distil-whisper-large-v3-en",
    "Gemma":"gemma2-9b-it",

}
messages = [(context),()]

patients=()

def retreive_notes(patient_name, database):
    notes = ''
    for i in range (len(database['encounter_data'])):
        metadata = database['encounter_data'][i]
        if metadata['name'] == patient_name:
            notes= notes + "\n" + database['note'][i]

    if not notes:
        return -1
    return notes


st.title("Clinical Note Summarizer")
st.divider()


database = st.file_uploader("Please upload the clinical data base ( parquet )",type=['parquet'])
llm_option = st.selectbox(
    "Which language model would you like to use?",
    ("Llama 3", "Distill", "Gemma"),
)

temprature = st.slider("Desired temprature for the language model", 0.0, 1.5, 0.2)

tokens = st.slider("Number of tokens used (lower amount = shorter summary but may be less accurate)", 0, 512,256)

if database is not None:
    database = pandas.read_parquet(database)

    patients = set()

    for val in database['encounter_data']:
        patients.add(val['name'])

patient_name = st.selectbox("Please enter the patient name for the summary" , tuple(patients),)
generate = st.button("Generate Summary")


if generate:
    llm = ChatGroq(
        model = models[llm_option],
        temperature=temprature,
        max_tokens=tokens
    )
    if database is not None and patient_name:
        patient_note = retreive_notes(patient_name,database)
        if patient_note == -1:
            st.error("The name is not in the database")
        else:
            messages[1] = patient_note
            summary = llm.invoke(messages)
            st.header('Clinical Summary')
            st.divider()
            st.write(summary.content)
    else:
        st.error("please input a name and a clinical database")

