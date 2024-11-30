import streamlit as st
import pandas as pd
import os
from jinja2 import Environment, FileSystemLoader
from utils import init_LLM, call_llm, testo_to_utf8

# Initialize the LLM with the Google API key from secrets
llm = init_LLM(API_KEY=st.secrets["GROQ_API_KEY"])
llm_model_name = "llama3-70b-8192"


# Load the Jinja template from the file
def load_template(template_path: str) -> str:
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)))
    template = env.get_template(os.path.basename(template_path))
    return template


# Main function
def main():
    st.set_page_config(page_title="llama-first-aid", page_icon="🦙")
    
    # Sidebar for project details
    st.sidebar.header("Dettagli")
    st.sidebar.write(""" 
        Sei pronto a intervenire in un'emergenza sanitaria?
         
        Con l'app **LLAMA** (Life-saving Live Assistant for Medical Assistance) **FIRST AID**, 
        avrai un operatore sanitario esperto sempre al tuo fianco. Che tu sia un neofita o abbia già esperienza nel primo soccorso, 
        l'app ti guiderà passo dopo passo nella gestione di situazioni critiche, offrendoti consigli rapidi e precisi. 
        Grazie a un'interfaccia intuitiva, potrai ricevere risposte in tempo reale alle domande cruciali e ottenere le istruzioni giuste per 
        intervenire al meglio. Inoltre, avrai accesso a video tutorial utili per apprendere e perfezionare le manovre di soccorso. Non lasciare
        nulla al caso, con **LLAMA** ogni emergenza diventa più gestibile!
    """)

    template_path = "sys_message_template.jinja"  #"./src/prompt_template.jinja"

    # User query input
    st.title("LLAMA FIRST AID 🦙")
    if query := st.chat_input("Come posso aiutarti"):
        # Load the Jinja template
        template = load_template(template_path)
        sys_message = template.render()

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        # Call the LLM with the Jinja prompt and DataFrame context
        with st.chat_message("assistant"):        
            stream = call_llm(llm, llm_model_name, sys_message, f'"{query}"')

            # Initialize an empty string to store the full response as it is built
            response = ""
            line_placeholder = st.empty()
            for chunk in stream:
                chunk_text = chunk.choices[0].delta.content
                clean_chunk = testo_to_utf8(chunk_text)
                response += clean_chunk
                line_placeholder.markdown(response, unsafe_allow_html=True)


if __name__ == "__main__":
    main()