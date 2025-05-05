

import streamlit as st
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


def generate_response(uploaded_file, openai_api_key,query_text):
	import os
	os.environ["OPENAI_API_KEY"] = openai_api_key

	# Load document if file is uploaded
	if uploaded_file is None:
		return "Please upload a file."
	
	# Load and read document
	documents = [uploaded_file.read().decode()]

	# Split documents into chunks
	text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
	texts = text_splitter.create_documents(documents)

	# Select embeddings
	embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

	# Create a vectorstore using FAISS (instead of Chroma) from documents
	db = FAISS.from_documents(texts, embeddings)

	# Create retriever interface
	retriever = db.as_retriever()

	# Create QA chain
	qa = RetrievalQA.from_chain_type(
		llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
		
	return qa.run(query_text)
	
# Streamlit UI
# Page title
st.set_page_config(page_title='  Ask the Doc App')
st.title('  Ask the Doc App')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')

# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
	openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not
(uploaded_file and query_text))
	submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
	with st.spinner('Calculating. . .'):
			response = generate_response(uploaded_file, openai_api_key, query_text)
			result.append(response)
			del openai_api_key
			
if len(result):
	st.info(response)
	
	