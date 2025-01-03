import os
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


def instantiate_model(api_key) -> ChatOpenAI:
    """Instantiates the model."""
    try:
        model = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo-1106", temperature=0.3)
        return model
    except Exception as e:
        raise ValueError(f"Failed to instantiate model: {e}")


def load_split_transcript(url):
    """Loads the YouTube transcript and splits it into chunks."""
    try:
        loader = YoutubeLoader.from_youtube_url(url)
        transcript = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_transcript = text_splitter.split_documents(transcript)
        return split_transcript
    except Exception as e:
        raise ValueError(f"Failed to load or process YouTube transcript: {e}")


def create_vector_store(split_transcript, api_key) -> FAISS:
    """Creates a vector store to store out embedded vectors."""
    try:
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vector_store = FAISS.from_documents(split_transcript, embedding=embeddings)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {e}")


def create_retriever_chain(vector_store, model):
    """Creates a retriever chain to retrieve answers from the vector store."""
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k":4})
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant that answer questions about YouTube Videos based on the video's transcript."),
            ("human", """Answer the following question: {input}
            By searching the following video transcript: {context}
            Only use the factual information from the transcript to answer the question. If you feel like you don't have enough information to answer the question, say I don't know. Your answers should be verbose and deatailed.""")
        ])
        document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
        retriever_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
        return retriever_chain
    except Exception as e:
        raise RuntimeError(f"Failed to create retriever chain: {e}")


def generate_response(retriever, question: str):
    """Generates a response to the user's question using the retriever chain."""
    try:
        response = retriever.invoke({
            "input": question
        })
        return response['answer']
    except Exception as e:
        raise RuntimeError(f"Failed to generate response: {e}")


if __name__ == "__main__":
    # Instantiate Model
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = instantiate_model(api_key)

    # Load and Split Transcript
    url = input("🔗 Enter the YouTube video URL: ").strip()
    split_transcript = load_split_transcript(url)
    vector_store = create_vector_store(split_transcript, api_key)

    # Create a retriever to search answers in the vector store.
    retriever = create_retriever_chain(vector_store, model)

    # Generate a response for the query
    question = input("❓ Enter your question about the video: ").strip()
    response = generate_response(retriever, question)
    print(f"\n{response}")
