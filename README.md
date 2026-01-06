# 📄 PDF Test Case Generator (Offline AI)

An AI-powered application that automatically generates **UI test scenarios** from **requirement PDF documents** using **offline Large Language Models**.

This tool helps QA teams reduce manual effort in requirement analysis and speeds up test case creation.

---

## 🚀 Features

* Upload requirement PDF
* Automatically detects high-level modules/features
* Generates UI test scenarios per module
* Classifies scenarios as **Simple / Medium / Complex**
* Supports generating scenarios for a single module or all modules
* Fully **offline** (no API keys required)
* Clean, structured tabular output

---

## 🛠️ Tech Stack

* Python 3.10
* Streamlit
* LangChain
* Ollama (llama3 – offline LLM)
* HuggingFace Embeddings
* ChromaDB
* PyPDFLoader

---

## 🧠 How It Works

1. User uploads a requirement PDF
2. PDF content is split into chunks
3. Chunks are embedded and stored in a vector database
4. Offline LLM (Ollama) extracts modules
5. Test scenarios are generated using RAG
6. Output is displayed in a structured table

---

## 📦 Environment Setup

```bash
conda create -n env_langchain1 python=3.10
conda activate env_langchain1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open browser at:

```
http://localhost:8501
```

---

## 📄 Usage

1. Upload a requirement PDF
2. Wait for module detection
3. Enter a module name (e.g., Login)
   **OR** type `all` to generate scenarios for all modules
4. View generated test scenarios in tabular format

---

## 📊 Output Format

| Module   | Scenario                                   | Complexity |
| -------- | ------------------------------------------ | ---------- |
| Login    | User logs in with valid credentials        | Simple     |
| Checkout | User applies invalid coupon and sees error | Medium     |

---

## 🔐 Security

* No API keys committed
* No secrets stored in repository
* Uses offline LLM execution
* Environment variables are excluded

---

## 🎯 Use Cases

* QA requirement analysis
* Test scenario generation
* Automation planning
* GenAI proof of concept for testing
* Interview and portfolio demonstration

---

## 🚧 Future Enhancements

* Export scenarios to Excel / CSV
* Integration with JIRA / Zephyr
* API test case generation
* BDD (Gherkin) output support

---

## 👤 Author

Sudip Dutta

