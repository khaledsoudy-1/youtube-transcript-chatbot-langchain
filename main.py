from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


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

print(split_transcript[0].page_content)
print(split_transcript[1].page_content)
print(split_transcript[2].page_content)
