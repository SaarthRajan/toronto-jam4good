import streamlit as st
from dotenv import load_dotenv
import os
import time
import tempfile
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGroq(api_key=SecretStr(groq_api_key), model="Llama3-8b-8192") # type: ignore

def vector_embedding(uploaded_pdf):

    if "vectors" in st.session_state:
        del st.session_state.vectors
    if "embeddings" in st.session_state:
        del st.session_state.embeddings
    if "loader" in st.session_state:
        del st.session_state.loader

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_pdf.read())
        temp_file_path = temp_file.name

    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFLoader(temp_file_path)        
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings



# Set Page Config -> title, logo, layout (centred by default)
st.set_page_config(
    page_title="InsurEase", 
    page_icon="⛑️",
    initial_sidebar_state="collapsed" # hiding the sidebar
)

# Used to remove the streamlit branding - and format other stuff (also check configure.toml for theme)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles/style.css")


# prompt=ChatPromptTemplate.from_template(
# """
# You are an assistant that help people figure out their healthcare
# benefits. You are friendly and can answer stuff based on the context. 
# You can talk like a person and not an AI. mimic the person's language. 
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}

# """
# )


prompt = ChatPromptTemplate.from_template(
"""
You are an assistant that helps people figure out their healthcare benefits.
You are from InsurEase Company and only analyze external health policies.
You are friendly and mimic the user's language and tone.
Answer the questions based on the provided context and chat history only.
Please provide the most accurate response based on the question =

<context>
{context}
</context>

<chat_history>
{chat_history}
</chat_history>

Question: {input}
"""
)


def title_ui():

    st.markdown("""
        <h1 style='text-align: center; color: #FF3333; font-family: sans-serif;'>
            InsurEase
        </h1>
        <p style='text-align: center; color: #FF3333; font-family: sans-serif;'>Navigate your Health Benefits with Ease</p>
    """, unsafe_allow_html=True)

def upload_ui():
    # Upload Your Insurance Information PDF - Upload Button
    uploaded_pdf = st.file_uploader("Upload Your Health Benefits PDF", type="pdf")

    if st.button("Generate Benefits"):
        if uploaded_pdf is not None:
            vector_embedding(uploaded_pdf)
            st.session_state.upload = True

            st.session_state.history = []
        else:
            st.warning("Please upload a PDF before generating.")
            st.session_state.upload = False

    if "upload" in st.session_state:
        return st.session_state.upload
    else:
        return False
    

def chat_ui():
    st.header("Chat", anchor=False)

    if "history" not in st.session_state:
        st.session_state.history = []


    for message in st.session_state.history:
        if message["role"] == "user":
            st.write(f"**You**: {message['content']}")
        else:
            st.write(f"**Assistant**: {message['content']}")
            with st.expander("Document Similarity Search"):
                # Find the relevant chunks
                for i, doc in enumerate(message["extra"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")

    prompt1 = st.chat_input("Ask Questions about your Insurance", key="prompt")

    if prompt1:
        st.session_state.history.append({"role": "user", "content": prompt1, "extra":False})
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vectors.as_retriever()
        retrieval_chain=create_retrieval_chain(retriever,document_chain)
        start=time.process_time()
        # response=retrieval_chain.invoke({'input':prompt1})

        history_context = ""
        for msg in st.session_state.history:
            if msg["role"] == "user":
                history_context += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                history_context += f"Assistant: {msg['content']}\n"

        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({
                'input': prompt1,
                'chat_history': history_context
            })



        st.session_state.history.append({"role": "assistant", "content": response['answer'], "extra":response["context"]})

        # With a streamlit expander
        st.write(f"**You**: {prompt1}")
        st.write(f"**Assistant**: {response['answer']}")
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

# Simulate the process of extracting available benefits
def extract_benefits():
    st.session_state.benefits = [
        {
            "benefit_name": "Health Coverage",
            "terms": "Valid for individuals up to 65 years old.",
            "coverage": "Up to $100,000 per year",
        },
        {
            "benefit_name": "Dental Coverage",
            "terms": "Valid for individuals of all ages.",
            "coverage": "Up to $10,000 per year",
        },
        {
            "benefit_name": "Vision Coverage",
            "terms": "Valid for individuals up to 70 years old.",
            "coverage": "Up to $5,000 per year",
        },
    ]

def benefits_ui():
    extract_benefits()
    
    st.header("Available Benefits", anchor=False)

    if "benefits" in st.session_state and st.session_state.benefits:
        for benefit in st.session_state.benefits:
            with st.expander(benefit["benefit_name"]):
                st.write(f"**Terms:** {benefit['terms']}")
                st.write(f"**Coverage:** {benefit['coverage']}")
                st.checkbox(f"Select {benefit['benefit_name']}")

    else:
        st.write("No benefits to display. Please upload a document and click 'Generate Benefits'.")

def main():

    title_ui()

    if upload_ui():
        benefits_ui()
        chat_ui()
        


if __name__ == "__main__" :
    main()