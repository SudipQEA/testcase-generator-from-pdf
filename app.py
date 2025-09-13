import streamlit as st
import pandas as pd
import io
import os

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# Streamlit title
# -----------------------------
st.title("📄 Requirement → Test Scenario Generator (Offline with Ollama)")

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
    # Create embeddings + vector store
    # -----------------------------
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
    # Ollama LLM (offline)
    # -----------------------------
    llm = Ollama(model="llama3")  # make sure you have run: ollama pull llama3

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
    Requirement,Scenario,Complexity

    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # -----------------------------
    # User query
    # -----------------------------
    query = st.chat_input("Enter requirement area (e.g., Login, Dashboard) or 'all' for all modules:")

    if query:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        # If user requests ALL modules → feed full document, not retriever
        if query.strip().lower() in ["all", "all modules", "share me all list of scenarios"]:
            context_text = "\n\n".join([d.page_content for d in docs])
            response = question_answer_chain.invoke({"input": query, "context": context_text})
            answer_text = response
        else:
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = rag_chain.invoke({"input": query})
            answer_text = response["answer"]

        # -----------------------------
        # Convert CSV-like response to DataFrame
        # -----------------------------
        try:
            df = pd.read_csv(io.StringIO(answer_text))
            st.subheader("✅ Generated Test Scenarios")
            st.dataframe(df, use_container_width=True)

            # CSV download code (commented out for now)
            """
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Scenarios as CSV",
                data=csv,
                file_name="test_scenarios.csv",
                mime="text/csv"
            )
            """
        except Exception:
            st.warning("⚠️ Could not parse as CSV. Showing raw output instead:")
            st.text(answer_text)
