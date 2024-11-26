from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Load Environment varibales
load_dotenv()

# Step 1: Load the YouTube transcript
url = "https://www.youtube.com/watch?v=0_guvO2MWfE"
loader = YoutubeLoader.from_youtube_url(url)
transcript = loader.load()

# Step 2: Split the transcript document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

split_transcript = text_splitter.split_documents(transcript)

# Step 3: Create a vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(split_transcript, embedding=embeddings)

# Step 4: Create a Retriver
retriever = vector_store.as_retriever(search_kwargs={"k":4})

# Step 5: Creat a prompt tempalate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that answer questions about YouTube Videos based on the video's transcript."),
    ("human", """Answer the following question: {input}
    By searching the following video transcript: {context}
    Only use the factual information from the transcript to answer the question. If you feel like you don't have enough information to answer the question, say I don't know. Your answers should be verbose and deatailed.""")
])

# Step 6: Load the model
model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.3
)

# Step 7: Create a Retriver chain
# First we have to create a document chain Then pass it to the retriever chain
document_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt
)

retriever_chain = create_retrieval_chain(
    retriever=retriever, 
    combine_docs_chain=document_chain
)

# Step 8: Generate a response
response = retriever_chain.invoke({
    "input": "What was Mohammed Hijab's opinion on divorce?"
})

print(response['answer'])