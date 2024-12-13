import streamlit as st
import random

# Function to generate a unique reference number
def generate_reference_number():
    return f"REF-{random.randint(1000, 9999)}"

# Streamlit App
st.title("AI Medical Assistant")
st.write("Welcome to the AI Medical Assistant. Choose your interface below.")

# Two buttons for Patient and Doctor interfaces
col1, col2 = st.columns(2)

with col1:
    if st.button("Patient", key="patient_button"):
        st.session_state["interface"] = "patient"

with col2:
    if st.button("Doctor", key="doctor_button"):
        st.session_state["interface"] = "doctor"

# Interface Navigation
if "interface" in st.session_state:
    if st.session_state["interface"] == "patient":
        st.subheader("Patient Interface")

        # Pop-up for Patient Information
        with st.form(key="patient_form"):
            name = st.text_input("Full Name")
            dob = st.date_input("Date of Birth")
            submit_patient = st.form_submit_button("Start Chat")

        if submit_patient:
            if name and dob:
                reference_number = generate_reference_number()
                st.success(f"Welcome, {name}! Your reference number is {reference_number}.")
                st.session_state["reference_number"] = reference_number
                st.session_state["patient_name"] = name
                st.session_state["dob"] = dob
            else:
                st.error("Please provide your full name and date of birth.")

        # Chatbot Placeholder (You can integrate your chatbot functionality here)
        if "reference_number" in st.session_state:
            st.subheader("Chatbot Conversation")
            st.write("Chatbot functionality will go here...")

    elif st.session_state["interface"] == "doctor":
        st.subheader("Doctor Interface")

        # Password Protected Area
        password = st.text_input("Enter Password", type="password")
        if st.button("Login"):
            if password == "doctor123":  # Replace with your secure password
                st.success("Welcome, Doctor!")
                st.write("Review diagnostic prompts here...")
                # Display the prompts generated for the doctor
                # Add your doctor-specific code here (e.g., `doctor_review` variable)
                st.write("Doctor's review section will display generated prompts.")
            else:
                st.error("Incorrect password. Please try again.")
