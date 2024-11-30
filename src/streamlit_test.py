import streamlit as st
from utils import *
import tempfile


# Initialize the LLM with the Google API key from secrets
llm = init_LLM(API_KEY=st.secrets["GROQ_API_KEY"])

# Applicazione Streamlit
def streamlit_app():
    st.title("Audio Recorder and Saver")

    # Input per registrare l'audio
    audio_value = st.audio_input("Record a voice message")

    if audio_value:
        st.audio(audio_value, format="audio/wav")  # Riproduce l'audio registrato
        st.success("Audio registrato con successo!")

        # Salva l'audio registrato
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_path = save_uploaded_audio(audio_value.getvalue(), temp_audio_file.name)
            st.success(f"File audio salvato in: {temp_audio_path}")

        # Aggiungi un link per scaricare il file audio
        st.download_button(
            label="Scarica il file audio",
            data=audio_value.getvalue(),
            file_name="recorded_audio.wav",
            mime="audio/wav",
        )
        st.write(transcribe_audio(llm, temp_audio_path))
        # Trascrivi l'audio
        transcription = transcribe_audio(llm, temp_audio_path)
        if transcription:
            st.write("Trascrizione completata:")
            st.text_area("Trascrizione", transcription)
        else:
            st.error("Errore durante la trascrizione.")

if __name__ == "__main__":
    streamlit_app()