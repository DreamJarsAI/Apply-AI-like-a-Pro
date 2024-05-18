# Import modules
import os
from dotenv import load_dotenv
import chromadb
from langchain.retrievers import (
    ContextualCompressionRetriever,
    MergerRetriever,
)
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain_chroma import Chroma
from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_transformers import LongContextReorder
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


# The warnings about forking processes and parallelism in the tokenizers library can be resolved 
# by setting an environment variable to control the parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Load environment variables
load_dotenv()


# Get the OpenAI API key
openai_api_key = os.environ["OPENAI_API_KEY"]


# Load a PDF file
loader = PyPDFLoader("chatgpt_book.pdf")
docs = loader.load_and_split()


# Get 3 different embeddings from HuggingFace and OpenAI
huggingface_embeddings1 = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
huggingface_embeddings2 = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-dot-v1")
openai_embeddings = OpenAIEmbeddings()


# Initialize Chroma with the documents and the first HuggingFace embedding
vector_db1 = Chroma.from_documents(
  documents=docs,
  embedding=huggingface_embeddings1,
  persist_directory='./db',
)


# Initialize Chroma with the documents and the second HuggingFace embedding
vector_db2 = Chroma.from_documents(
  documents=docs,
  embedding=huggingface_embeddings2,
  persist_directory='./db',
)


# Define 2 different retrievers with 2 unique embeddings and search types
retriever_huggingface1 = vector_db1.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
retriever_huggingface2 = vector_db2.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 5},
)


# The Lord of the Retrievers (LOTR) will hold the output of both retrievers and can be used as any other
# retriever on different types of chains
lotr = MergerRetriever(retrievers=[retriever_huggingface1, retriever_huggingface2])


## Option 1: Remove redundant results from the merged retrievers ##

# # We can remove the redundant results from both retrievers using yet another embedding
# Note: Using multiples embeddings in different steps could help reduce biases
# filter = EmbeddingsRedundantFilter(embeddings=openai_embeddings)
# # Note: No matter the architecture of your model, there is a substantial performance degradation 
# # when you include 10+ retrieved documents. 
# # In brief: When models must access relevant information in the middle of long contexts, 
# # then tend to ignore the provided documents.
# # You can use an additional document transformer to reorder documents after removing redundancy.
# reordering = LongContextReorder()
# pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=pipeline, base_retriever=lotr
# )


## Option 2: Pick a representative sample of documents from the merged retrievers ##

# This filter will divide the documents vectors into clusters or "centers" of meaning
# Then it will pick the closest document to that center for the final results
# By default the result document will be ordered/grouped by clusters
filter_ordered_by_retriever = EmbeddingsClusteringFilter(
    embeddings=openai_embeddings,
    num_clusters=10,
    num_closest=1,
    # If you want the final document to be ordered by the original retriever scores
    # you need to add the "sorted" parameter
    sorted=True,
)
reordering = LongContextReorder()
pipeline = DocumentCompressorPipeline(transformers=[filter_ordered_by_retriever, reordering])
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=lotr
)



# Initialize the LLM for augmented QA
llm = ChatOpenAI(model_name="gpt-4", temperature=0)


# Instantiate the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = compression_retriever,
)


# Run the QA
query ="What are the ten principles for prompt engineering?"
response = qa_chain.invoke(query)
print(response)
