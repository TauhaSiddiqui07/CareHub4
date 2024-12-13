import streamlit as st
from pinecone import Pinecone
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.tokenize import MWETokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
import os
import google.generativeai as genai


# Configure the Generative AI client with the API key
genai.configure(api_key="AIzaSyABey4cEodWBa_ptTJ-MmO5M5DAPlPjEhs")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

# Agent_1 for Input preprocessing
model_1 = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="""
  TASKS:\n 1. Correct all spelling mistakes.\n
           2. Fix grammatical errors.\n
           3. Normalize text:\n  - Convert text to lowercase.\n    - Remove extra spaces.\n    - Ensure proper punctuation.\n
           4. Return cleaned and corrected text.\n
           5. Do not use contraction of the words \n
           6.Translate into english \n
           7. If user say bye, good bye or wants to stop taking then always return back with goodbye.
  """
  )
Agent_1 = model_1.start_chat(
  history=[]
)
response = Agent_1.send_message("hi")
print(response.text)

# Agent_2 for AI medical Diagnoses
model_2 = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="""
    You are a polite and helpful healthcare assistant agent (Agent 2). Your job is to:
    1. Start with a greeting the patient humbly and establish a friendly tone.
       And let the patient's name as it is; don't call them by the translation of their name.
    2. Collect the following information step by step:
       - Name
       - Age
       - Gender
       - Blood Pressure status (HIGH/LOW/NORMAL)
       - Diabetic status (YES/NO)
       - Symptoms: Ask one by one
       - Ask the patient to describe their symptoms in detail.
       - Follow up with these questions to extract more information: ask one by one
       - *Provocation/Palliation:* What makes the symptoms better or worse?
        a. *Sensation:* What do the symptoms feel like? (e.g., sharp, dull, aching, burning)
        b. **Ask assosiated questions to the symptoms, ask related one?
        c. *Timing:* Are the symptoms constant or do they come and go?
        d. *Associated symptoms:* Suggest the Associated symptoms according to patient described symptoms?
        e. Duration of symptoms (in days, weeks, etc.).
        f. *Severity:* On a scale of 1 to 10, how severe are the symptoms? (1 = mild, 10 = severe)
    3. After collecting all the details, present the all information back to the patient
       in a summarized and well-structured format for verification.
    4. Ask the patient if the information is correct:
       - If the patient confirms (YES), respond with: "Thank you! The doctor will review your application." after this add another text response of goodbye.
       - If the patient wants to edit any information (NO):
         - Ask which field they want to edit.
         - Update the information based on their input.
         - Repeat the process until the patient confirms all details are accurate.
    5. Maintain a professional yet empathetic tone throughout the interaction.
    6. **Modification 1: If the user input contains any vulgar or offensive language,
       respond with "I'm sorry, I cannot process that request. Please use respectful language." and terminate the conversation.**
    7. **Modification 2: If the user provides irrelevant information or goes off-topic,
       politely guide them back to the relevant questions and information needed.**
    """
)

Agent_2 = model_2.start_chat(history=[])
response = Agent_2.send_message("Hi")
print(response.text)

# Agent_3 for Diesease Diagnose and Treatment recommandations.
model_3 = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="""
    You are a healthcare assistant agent (Agent 3). Your job is to:
    1. Receive a summary of the patient's condition and similarity search results.
    2. Generate a prompt for doctor review that includes:
        - Patient summary
        - Top similar cases (Patient IDs and metadata)
    3. Recommend medicine based on:
        - Similar cases (metadata should contain medicine information)
        - Patient age (ensure age-appropriateness of medicine)
    4. Display the summary and recommended medicine to the user.

    Example:
    Input:
    - Summary: "Patient is a 25-year-old male experiencing fever, cough, and body aches for 3 days."
    - Similarity Results: [{"id": "patient123", "metadata": {"age": 22, "medicine": "Paracetamol"}, "score": 0.85}, ...]

    Output:
    - Doctor Review Prompt: "25-year-old male with fever, cough, and body aches for 3 days. Similar cases: patient123 (age 22, medicine: Paracetamol)."
    - Recommended Medicine: "Paracetamol"
    """
)

Agent_3 = model_3.start_chat(history=[])

# Summary Preproccessing and tokenization then Embedding model is Word2Vec and the Vector Database is Pinecone.
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    mwe = MWETokenizer([('chest', 'pain'), ('abdominal', 'pain'), ('body', 'pain')], separator=' ')
    tokens = mwe.tokenize(word_tokenize(text))
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]
    return (tokens)

# Load the Word2Vec model
with open('Treatment_model.pkl', 'rb') as file:
    Embedding_Treatment_model = pickle.load(file)
# Load the Word2Vec model
with open('Diagnose_Model.pkl', 'rb') as file:
    Embedding_Diagnose_model = pickle.load(file)
# load pinecone with API key
pc = Pinecone(api_key='pcsk_3rT1tk_PgWmLU8Bo72KTcSZob68RVdKSsQVfReo13nXpmi2JCfCJzHfYLCzNJBiwgkptk5')
# call the index from the data base
index_treatment = pc.Index("data-carehub")
index_diagnose = pc.Index("diagnose")

# Function to Search in vector database
def qurey_Search(query_symptoms,index,embedding_model):
  query_symptoms = preprocess_text(query_symptoms)
  query_embedding = get_embedding(query_symptoms,embedding_model)
  query_embedding_list = query_embedding.tolist()
  results = index.query(namespace = "ns1",vector = query_embedding_list, top_k=3, include_values = True, include_metadata=True,filter={})
  return results



# Function to generate embeddings for symptoms
def get_embedding(symptoms_list,model):
    vectors = [model.wv[word] for word in symptoms_list if word in model.wv]
    vectors = [v.astype(float) for v in vectors]
    return sum(vectors) / len(vectors) if vectors else np.array([0.0] * 3072)


# Patient talking part and workflow
def refine_and_converse(user_input, conversation_history=[]):
    """
    Refines user input, engages in conversation,
    and generates a summary at the end.
    """
    # Refine user input with Agent 1
    refined_input = Agent_1.send_message(user_input).text
    global conversation_active

    # Check if user wants to end the conversation
    if refined_input.lower() == "goodbye" or user_input.lower() == "goodbye":
        print(f"CareHUB: Goodbye!!")  # Optional: Print a goodbye message
        conversation_active = False
        return conversation_history

    # Pass refined input to Agent 2 for conversation
    agent_2_response = Agent_2.send_message(refined_input).text
    print(f"CareHUB: {agent_2_response}")
    # Append refined input and agent response to history
    conversation_history.extend([refined_input, agent_2_response])

    # Check if Agent 2 says goodbye
    if agent_2_response.lower() == "goodbye":
        print(f"CareHUB: Goodbye!!")  # Optional: Print a goodbye message
        conversation_active = False
        return conversation_history

    return conversation_history


# Main interaction loop
conversation_history = []
conversation_active = True
user_input = input("Patient: ")

while conversation_active:
  if conversation_active == False:
    break
  conversation_history = refine_and_converse(user_input, conversation_history)
  user_input = input("Patient: ")
  if conversation_active == False:
    break


# Generate summary using Agent 2 after conversation ends
summary_prompt = f"Please summarize the following conversation:\n\n{conversation_history}"
summary = Agent_2.send_message(summary_prompt).text


# Searching Vector Database for similarities of Diasease and Treatments
print("Proceeding to database search...")
result_Treat = qurey_Search(summary,index_treatment,Embedding_Treatment_model)
result_Diag= qurey_Search(summary,index_diagnose,Embedding_Diagnose_model)


# Fetch out the best results based on the similarity score greater than 80%
for r1, r2 in zip(result_Treat["matches"], result_Diag["matches"]):
  #Your code to process r1 and r2
  if r1['score'] > 0.8:
    print(f"Patient ID: {r1['id']}, Metadata: {r1['metadata']}")
    print(f"Similarity: {r1['score']}")
  if r2['score'] > 0.8:
    print(f"Patient ID: {r2['id']}, Metadata: {r2['metadata']}")
    print(f"Similarity: {r2['score']}")
  else:
    print("No matching patients found.")
# Now create a prompt for the Agent 3.
prompt = f"""
Current Patient:
{summary}
Similar Cases and Treatment:
"""
def add_similar_cases(prompt, results):
   for result in results:
     prompt += f"- Patient ID: {result['id']}, Metadata: {result['metadata']}, Similarity Score:{result['score']}\n"
   return prompt

prompt = add_similar_cases(prompt, result_Treat["matches"])
prompt += f"\nDiagnose and Symptoms:\n"
prompt = add_similar_cases(prompt, result_Diag["matches"])

# Pass the prompt to Agent_3 to genrate prompt for Doctor
doctor_review = Agent_3.send_message(prompt).text
print(doctor_review)
