import streamlit as st
import pandas as pd
import io
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_stuff_documents_chain, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_llms import HuggingFacePipeline

# HuggingFace imports
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Load environment variables (optional)
load_dotenv()

# -----------------------------
# Streamlit title
# -----------------------------
st.title("📄 Requirement → Test Scenario Generator (Offline)")

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Requirement PDF", type="pdf")

if uploaded_file:
    with open("uploaded_req.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # -----------------------------
    # Load PDF
    # -----------------------------
    loader = PyPDFLoader("uploaded_req.pdf")
    data = loader.load()

    # -----------------------------
    # Split into chunks
    # -----------------------------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)

    # -----------------------------
    # Create embeddings + vector store (offline)
    # -----------------------------
    from langchain.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    persist_directory = "chroma_db"

    if not os.path.exists(persist_directory):
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectorstore.persist()
    else:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # -----------------------------
    # HuggingFace LLM
    # -----------------------------
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    hf_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    # -----------------------------
    # Prompt template
    # -----------------------------
    system_prompt = """
    You are a QA test scenario generator.
    You will read requirements from the context and generate **UI Test Scenarios**.
    For each requirement:
    1. Extract requirement text.
    2. Suggest 2–3 test scenarios.
    3. Assign complexity: 
       - Simple → Basic validation (single field/button check).
       - Medium → Multiple fields, conditions, or validations.
       - Complex → Multi-step workflow, role-based checks, or dependencies.

    Return output strictly in **CSV format** with columns:
    Requirement, Scenario, Complexity

    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # -----------------------------
    # User query
    # -----------------------------
    query = st.chat_input("Enter requirement area (e.g., Login, Dashboard):")

    if query:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": query})

        # -----------------------------
        # Convert CSV-like response to DataFrame
        # -----------------------------
        try:
            df = pd.read_csv(io.StringIO(response["answer"]))
            st.dataframe(df)

            # Option to download as CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Scenarios as CSV",
                data=csv,
                file_name="test_scenarios.csv",
                mime="text/csv"
            )
        except Exception:
            st.warning("⚠️ Could not parse as CSV. Showing raw output instead:")
            st.text(response["answer"])
