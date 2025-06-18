import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import pandas
import uuid


#Add actual notes in the output, polish UI
load_dotenv("keys.env")

with open("context.txt",'r') as c:
    context = c.read()

models = {
    "Llama 3":"llama-3.3-70b-versatile",
    "Distill":"distil-whisper-large-v3-en",
    "Gemma":"gemma2-9b-it",

}
messages = [(context),()]


def retreive_notes(patient_id, database):
    
    notes = []
    for i in range (len(database['patient_id'])):
        id = database['patient_id'][i]
        if id == patient_id:
            notes.append(database['note'][i])

    return notes


# returns new database with an extra column of UUID
def generate_uniqueID(database):
    unique = []
    
    patients = {}

    for encounter_data in database['encounter_data']:
        if (encounter_data['age'],encounter_data['name']) not in patients:
            patients[(encounter_data['age'],encounter_data['name'])] = str(uuid.uuid4())[:8]
        unique.append(patients[(encounter_data['age'],encounter_data['name'])])

    database['patient_id']=unique

    return database,tuple(patients.values())



st.title("Clinical Note Summarizer")
st.divider()


if 'patients' not in st.session_state:
    st.session_state['patients'] = ()

if 'database' not in st.session_state:
    st.session_state['database'] = None

uploaded_database = st.file_uploader("Please upload the clinical data base ( parquet )",type=['parquet'])
submit_database = st.button("Upload Databse")
llm_option = st.selectbox(
    "Which language model would you like to use?",
    ("Llama 3", "Distill", "Gemma"),
)

temprature = st.slider("Desired temprature for the language model", 0.0, 1.5, 0.2)

tokens = st.slider("Number of tokens used (lower amount = shorter summary but may be less accurate)", 0, 512,256)

if submit_database and uploaded_database is not None:
    uploaded_database = pandas.read_parquet(uploaded_database)

    st.session_state['database'] , st.session_state['patients'] = generate_uniqueID(uploaded_database)


patientID = st.selectbox("Please enter the patient ID for the summary" , st.session_state['patients'],)


generate = st.button("Generate Summary")


if generate:
    llm = ChatGroq(
        model = models[llm_option],
        temperature=temprature,
        max_tokens=tokens
    )
    if st.session_state['database'] is not None and patientID:
        patient_note = retreive_notes(patientID,st.session_state['database'])

        patient_note = "\n".join(patient_note)

        if not patient_note:
            st.error("The name is not in the database")
        else:
            with st.status("Creating Patient Summary"):
                messages[1] = patient_note
                summary = llm.invoke(messages)
                st.header('Patient Note')
                st.divider()
                st.write(patient_note)
                st.header('Summary')
                st.write(summary.content)
    else:
        st.error("please input a name and a clinical database")

