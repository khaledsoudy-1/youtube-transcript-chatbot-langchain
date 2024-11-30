from youtube_retriever import *
import streamlit as st

# Set up Streamlit page
st.set_page_config(page_title="YouTube Transcript Chatbot", layout="centered")

st.title("🎥 YouTube Transcript Chatbot")
st.write("Ask questions based on a YouTube video's transcript!")

# Input section
api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password", placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
video_url = st.text_input("🔗 Enter YouTube video URL:", placeholder="https://www.youtube.com/watch?v=0_guvO2MWfE&t=3611s")
question = st.text_input("❓ Enter your question:", placeholder="What was the topic about?")

# Add a link to help users get their API key
st.markdown("""
    If you don't have an API key, you can get one from here: [OpenAI API Key](https://platform.openai.com/signup) 💡.
""")

# Process and respond
if st.button("🚀 Get Answer"):
    if not api_key or not video_url or not question:
        st.warning("Please enter your API key, video URL, and question!")
    else:
        with st.spinner("Processing..."):
            try:
                # Instantiate model and process
                model = instantiate_model(api_key)
                split_transcript = load_split_transcript(video_url)
                vector_store = create_vector_store(split_transcript, api_key)
                retriever = create_retriever_chain(vector_store, model)
                response = generate_response(retriever, question)
                
                # Display the response
                st.success("Answer:")
                st.write(response)

            except ValueError as ve:
                st.error(f"Value Error: {ve}")
            except KeyError as ke:
                st.error(f"Key Error: {ke}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# App Footer 
st.markdown("""
---
Made with ❤️ by [Khaled Soudy](https://github.com/khaledsoudy-1)
""")