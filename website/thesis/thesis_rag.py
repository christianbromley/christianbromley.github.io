"""Create a RAG app for my thesis."""
# %%
# Imports
import pandas as pd
import getpass
import os
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader  # Load PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

# %%
# Constants
THESIS_PATH = "/Users/cbromley/Documents/Colab/bert_transformer/Christian_Bromley_Final_Thesis_20211221_cover.pdf"
LANGCHAIN_API_KEY=""
OPENAI_API_KEY=""
LANGCHAIN_TRACING_V2 = "true"

# %%

# LANGCHAIN_TRACING_V2 = os.environ["LANGCHAIN_TRACING_V2"] 
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["OPENAI_API_KEY="] = getpass.getpass()

LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_PROJECT="thesis_rag"

# %%
# Instantiate LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# %%
# 1. Load the PDF
pdf_loader = PyMuPDFLoader(THESIS_PATH)
documents = pdf_loader.load()

# %%
# 2. Chunk the text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Adjust based on your use case
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)
# %%
# docs
# %%
# Embed
vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

# %%
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")