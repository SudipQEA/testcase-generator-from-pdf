import streamlit as st
import pandas as pd
import io
import os
import re
import json

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
# App title
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
    with st.spinner("📖 Reading PDF..."):
        loader = PyPDFLoader("uploaded_req.pdf")
        data = loader.load()

    # -----------------------------
    # Split into chunks
    # -----------------------------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)  # list[Document]

    # -----------------------------
    # Create embeddings + vector store
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_directory = "chroma_db"

    with st.spinner("⚡ Building vector DB..."):
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
    # Ollama LLM (offline, deterministic)
    # TIP: if formatting is unstable, try model="llama3:instruct"
    # -----------------------------
    llm = Ollama(model="llama3", temperature=0)

    # -----------------------------
    # Step 1: Extract modules (clean)
    # -----------------------------
    module_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a QA assistant. From the provided requirements, "
         "list the DISTINCT high-level modules/features ONLY as a numbered list "
         "(e.g., '1. Login & Authentication'). No extra text.\n\n{context}"),
        ("human", "List the modules.")
    ])
    module_chain = create_stuff_documents_chain(llm, module_prompt)

    with st.spinner("🔍 Extracting modules..."):
        raw_modules = module_chain.invoke({"context": docs})  # pass Documents
        modules_text = str(raw_modules).strip()

    if modules_text:
        st.subheader("📌 Detected Modules")
        st.code(modules_text, language=None)

    # -----------------------------
    # Helpers: output normalization, JSON & pipe fallback, parsing
    # -----------------------------
    def _to_text(resp) -> str:
        """Normalize LangChain chain outputs to a plain string."""
        if isinstance(resp, dict):
            for k in ("answer", "output_text", "text", "result"):
                if k in resp and isinstance(resp[k], str):
                    return resp[k]
        return str(resp)

    def build_json_chain(the_llm):
        """Chain that asks for JSON array output."""
        system_prompt = """
You are a STRICT JSON generator for QA test scenarios.

Task:
- Read the provided requirements (context).
- Generate UI test scenarios for the requested module.
- Each scenario object MUST contain:
  - "Module": the exact requested module name.
  - "Scenario": one concise sentence describing the test.
  - "Complexity": exactly "Simple", "Medium", or "Complex".

Output rules (MANDATORY):
- Output ONLY a valid JSON array (no prose, no markdown, no code fences).
- 3 to 6 items.
- No trailing commas. Keys exactly: Module, Scenario, Complexity.

Return ONLY the JSON array.
{context}
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        return create_stuff_documents_chain(the_llm, prompt)

    def build_pipe_chain(the_llm):
        """Chain that asks for pipe-delimited lines as a fallback."""
        system_prompt = """
You are a QA scenario lister.

Output rules (MANDATORY):
- Output ONLY lines in this exact format (no headers, no extra text):
Module | Scenario | Complexity
- 3 to 6 lines.
- Complexity must be exactly one of: Simple, Medium, Complex.

Examples:
Login & Authentication | User logs in with valid credentials and is redirected to homepage | Simple
Checkout | User applies invalid coupon code and sees error message near promo field | Medium

{context}
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        return create_stuff_documents_chain(the_llm, prompt)

    def extract_json_array(text: str) -> str:
        """Extract the first JSON array from text (tolerates stray text/fences)."""
        s = re.sub(r".*?", "", _to_text(text), flags=re.S).strip()
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            return s[start:end+1]
        return ""

    def parse_json_to_df(json_text: str, expected_module: str) -> pd.DataFrame:
        """Parse JSON array to DataFrame and sanitize."""
        if not json_text:
            return pd.DataFrame(columns=["Module", "Scenario", "Complexity"])
        try:
            data = json.loads(json_text)
        except Exception:
            return pd.DataFrame(columns=["Module", "Scenario", "Complexity"])

        if not isinstance(data, list):
            return pd.DataFrame(columns=["Module", "Scenario", "Complexity"])

        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            module = str(item.get("Module", expected_module or "")).strip()
            scenario = str(item.get("Scenario", "")).strip()
            complexity = str(item.get("Complexity", "")).strip().title()
            out.append({"Module": module or expected_module, "Scenario": scenario, "Complexity": complexity})

        df = pd.DataFrame(out, columns=["Module", "Scenario", "Complexity"])
        return _clean_df(df, expected_module)

    def parse_pipe_to_df(text: str, expected_module: str) -> pd.DataFrame:
        """
        Parse lines like:
        Module | Scenario | Complexity
        """
        rows = []
        for ln in _to_text(text).splitlines():
            ln = ln.strip()
            if not ln or " | " not in ln:
                continue
            # ignore headings or examples labeled explicitly
            if re.match(r"^(module|scenario|complexity|example)", ln, re.I):
                continue
            parts = [p.strip().strip('"') for p in ln.split(" | ")]
            if len(parts) < 3:
                continue
            module, scenario, complexity = parts[0], parts[1], parts[2]
            rows.append({"Module": module or expected_module, "Scenario": scenario, "Complexity": complexity})
        df = pd.DataFrame(rows, columns=["Module", "Scenario", "Complexity"])
        return _clean_df(df, expected_module)

    def _clean_df(df: pd.DataFrame, expected_module: str) -> pd.DataFrame:
        """Common cleaning and validation."""
        if df.empty:
            return pd.DataFrame(columns=["Module", "Scenario", "Complexity"])
        df["Module"] = df["Module"].fillna(expected_module).replace("", expected_module).astype(str).str.strip()
        df["Scenario"] = df["Scenario"].astype(str).str.strip()
        df["Complexity"] = df["Complexity"].astype(str).str.title().str.strip()

        valid_levels = {"Simple", "Medium", "Complex"}
        df = df[df["Complexity"].isin(valid_levels)]
        df = df[df["Scenario"].str.contains(r"[A-Za-z]", na=False)]
        df = df[df["Scenario"].str.len() >= 8]
        df = df[df["Module"].str.len() > 0]
        df = df.drop_duplicates(subset=["Module", "Scenario", "Complexity"]).reset_index(drop=True)
        return df

    def generate_for_module(module_name: str, docs_list):
        """
        Try JSON first; if empty/invalid, fall back to pipe-delimited lines.
        """
        # JSON attempt
        json_chain = build_json_chain(llm)
        user_input_json = (
            f"Generate 3–6 UI test scenarios for the module: {module_name}.\n"
            f"Return ONLY a JSON array with keys: Module, Scenario, Complexity."
        )
        raw1 = json_chain.invoke({"input": user_input_json, "context": docs_list})
        json_block = extract_json_array(raw1)
        df = parse_json_to_df(json_block, expected_module=module_name)

        if not df.empty:
            return df

        # Pipe fallback
        pipe_chain = build_pipe_chain(llm)
        user_input_pipe = (
            f"Generate 3–6 UI test scenarios for the module: {module_name}.\n"
            f"Return ONLY lines in the format: Module | Scenario | Complexity."
        )
        raw2 = pipe_chain.invoke({"input": user_input_pipe, "context": docs_list})
        df2 = parse_pipe_to_df(raw2, expected_module=module_name)
        return df2

    # -----------------------------
    # Step 2: User query → scenarios (table)
    # -----------------------------
    query = st.chat_input("Enter module name (or 'all' for all modules):")

    if query:
        with st.spinner("🛠️ Generating test scenarios..."):
            q = query.strip()
            if q.lower() in ["all", "all modules", "share me all list of scenarios"]:
                detected = [
                    re.sub(r"^\s*\d+\.\s*", "", m).strip(" -•\t")
                    for m in re.split(r"\n+", modules_text or "")
                    if m.strip()
                ]
                detected = [m for m in detected if m and not m.isdigit()]

                frames = []
                for m in detected:
                    try:
                        df_m = generate_for_module(m, docs)  # pass Documents
                        if not df_m.empty:
                            frames.append(df_m)
                    except Exception as e:
                        st.error(f"Failed to generate for module '{m}': {e}")

                df_final = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                    columns=["Module", "Scenario", "Complexity"]
                )
            else:
                df_final = generate_for_module(q, docs)  # pass Documents

        st.subheader(f"✅ Clean scenarios ({len(df_final)} rows)")
        st.dataframe(df_final, use_container_width=True)

else:
    st.info("Upload a requirement PDF to begin.")
