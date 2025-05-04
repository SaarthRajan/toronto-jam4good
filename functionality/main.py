import streamlit as st
from dotenv import load_dotenv
import os
import time
from groq import Groq
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



def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFLoader("./sample_datasets/ex2.pdf")        
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



# remove later
prompt=ChatPromptTemplate.from_template(
"""
You are an assistant that help people figure out their healthcare
benefits.
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)





def title_ui():

    # might remove icon and add a logo
    # might even remove the title - just have the image

    # # to center the image
    # cols = st.columns(4) # change accordingly
    # with cols[1]:
        # InsurEase logo here
        # st.image("./space_apps_logo.png", width=300)

    st.markdown("""
        <h1 style='text-align: center; color: #FF3333; font-family: sans-serif;'>
            InsurEase ⛑️
        </h1>
        <p style='text-align: center; color: #FF3333; font-family: sans-serif;'>Navigate the Complexity of Insurance with Ease</p>
    """, unsafe_allow_html=True)

def upload_ui():
    # Upload Your Insurance Information PDF - Upload Button
    uploaded_pdf = st.file_uploader("Upload Your Insurance Information PDF", type="pdf")

    if st.button("Generate Benefits"):
        if uploaded_pdf is not None:
            vector_embedding()
            st.write("Vector Store DB Is Ready")
        else:
            st.warning("Please upload a PDF before generating.")

def chat_ui():
    st.header("Chat", anchor=False)

    if "history" not in st.session_state:
        st.session_state.history = []

    messages = st.container(border=True, height=600)

    with messages:
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
        response=retrieval_chain.invoke({'input':prompt1})
        st.session_state.history.append({"role": "assistant", "content": response['answer'], "extra":response["context"]})

        # With a streamlit expander
        st.write(response['answer'])
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")


def benefits_ui():
    st.header("Available Benefits", anchor=False)

def main():

    title_ui()

    upload_ui()

    col1, col2 = st.columns(2, gap="large") #, vertical_alignment="center"

    with col2 :
        chat_ui()
    
    with col1 :
        benefits_ui()


if __name__ == "__main__" :
    main()