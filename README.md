# 🎥 YouTube Transcript Chatbot

## 📖 Description
A powerful Streamlit web application that leverages OpenAI's language models and LangChain to extract insights directly from YouTube video transcripts. This chatbot allows users to ask questions about the content of any YouTube video, providing detailed, context-aware responses based on the video's transcript.


---

## 🌐 Live Demo
Check out the live app here: [YouTube Transcript Chatbot](https://youtube-transcript-chatbot-langchain-256oa2kv437oq6xpaqsfs3.streamlit.app/) 🚀

---


## ✨ Features
- **Intelligent Transcript Analysis**: Uses advanced AI to extract and understand video content
- **Interactive Web Interface**: Built with Streamlit for a user-friendly experience
- **OpenAI Integration**: Utilizes GPT-3.5-turbo for generating accurate responses
- **Flexible Querying**: Ask any question about the video's content
- **Robust Error Handling**: Provides clear feedback and error messages
- **Transcript Chunking**: Implements intelligent text splitting for comprehensive analysis

## 🚀 How to Use
### Prerequisites
- Python 3.8+
- [OpenAI API Key](https://platform.openai.com/signup)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/youtube-transcript-chatbot.git
   cd youtube-transcript-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API Key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

### Running the Application
1. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. In the web interface:
   - Enter your OpenAI API Key
   - Paste a YouTube video URL
   - Ask a question about the video's content
   - Click "Get Answer" to receive a detailed response

## 💻 Code Highlights
### Key Components
- **`app.py`**: Streamlit user interface handling
- **`youtube_retriever.py`**: Core logic for transcript retrieval and AI processing
- Utilizes LangChain for document loading, text splitting, and retrieval

### Transcript Processing Flow
```python
def load_split_transcript(url):
    """Loads YouTube transcript and splits into manageable chunks"""
    loader = YoutubeLoader.from_youtube_url(url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    split_transcript = text_splitter.split_documents(transcript)
    return split_transcript
```

## 🛠️ Potential Enhancements
- Support for multiple video sources
- Caching mechanism for faster repeated queries
- Enhanced error handling for various URL formats
- Option to download or export conversation
- Multilingual transcript support
- Improved context retention across multiple questions

## 📦 Dependencies
- Streamlit
- LangChain
- OpenAI
- YouTube Transcript API
- python-dotenv

## ⚠️ Disclaimer
- Requires a valid OpenAI API key
- Be mindful of API usage and associated costs
- Responses are generated based on transcript content
- Not all videos may have accessible transcripts

## 👨‍💻 Author
Khaled Soudy

## 🤝 Contributing
Contributions are welcome! Please:
- Fork the repository
- Create a feature branch
- Submit a pull request

## 📄 License
This project is open-source and available under the MIT License.

---
🎬 Unlock the insights hidden in YouTube videos, one question at a time! 🤖