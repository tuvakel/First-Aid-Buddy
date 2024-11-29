import streamlit as st
import pandas as pd
import os
from jinja2 import Environment, FileSystemLoader
from utils import init_LLM, testo_to_utf8

# Load dotenv
from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM with the Google API key from secrets
llm = init_LLM(AIML_API_KEY=os.getenv('AIML_API_KEY'))
llm_model_name = "nvidia/llama-3.1-nemotron-70b-instruct"

def call_llm(sys_message: str, context_message: str, base64_image: str = "") -> str:
    if base64_image == "":
        messages = [
            {"role": "system", "content": sys_message},
            {"role": "user", "content": context_message}
        ]
    else:
        messages = [
            {"role": "system", "content": sys_message},
            {"role": "user", "content": context_message},
            {"role": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]

    ai_message = llm.chat.completions.create(
        model=llm_model_name,
        messages=messages
    )
    return ai_message.choices[0].message.content


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
        Questa applicazione consente di ...
    """)

    template_path = "prompt_template.jinja"  #"./src/prompt_template.jinja"

    # User query input
    st.title("Comunica col tuo operatore sanitario virtuale")
    if query := st.chat_input("Come posso aiutarti"):
        # Load the Jinja template
        template = load_template(template_path)
        prompt = template.render(query=query)
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        # Call the LLM with the Jinja prompt and DataFrame context
        with st.chat_message("assistant"):        
            stream = call_llm(prompt)
            # Initialize an empty string to store the full response as it is built
            response = ""
            line_placeholder = st.empty()
            for chunk in stream:
                chunk_text = str(chunk.content)
                # Clean each chunk as it arrives
                clean_chunk = testo_to_utf8(chunk_text)
                # Append to the full response
                response += clean_chunk
                # Display the cleaned chunk
                line_placeholder.markdown(response, unsafe_allow_html=True)


if __name__ == "__main__":
    main()