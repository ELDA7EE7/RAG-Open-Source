# 📘 RAG Pipeline with Local Models (Ollama)

This repository contains an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline for analyzing PDF documents (such as annual reports).
The project is **inspired by** and builds upon the code demonstrated in the excellent YouTube video:

▶️ **[Build a RAG Pipeline from Scratch](https://youtu.be/ea2W8IogX80?si=YpyBuH6e-Jgamiti)**
by **[Vincibits](https://www.youtube.com/@vincibits)**

Vincibits’ original project uses **OpenAI models** for query expansion and answer generation.
In this repo, OpenAI has been **fully replaced with the open-source Ollama backend**, allowing:

* 🚫 **No API key required**
* 🖥️ **100% local inference**
* 🌱 **Fully open-source RAG pipeline**
* 🔌 **Use any local LLM available through Ollama** (Llama 3, Mistral, Phi-3, Qwen, Gemma, etc.)

Everything else (text splitting, embeddings, vector search, and visualization) follows the structure of Vincibits’ original implementation.

---

# ✨ Features

* PDF loading and text extraction
* Recursive + token-aware text chunking
* SentenceTransformer embeddings
* ChromaDB vector storage
* Query augmentation using **local Ollama models**
* Retrieval of relevant chunks
* Final answer generation using **LLM reasoning + retrieved context**
* UMAP 2D embedding visualization

---

# 📁 Repository Structure

```
.
├── expansion_answer.py     # Main script
├── requirements.txt        # Python dependencies
├── data/
│   └── your-pdf-file.pdf   # For example: microsoft-annual-report.pdf
└── README.md
```

---

# 🚀 How to Run

### **1. Install Ollama**

Download from:
[https://ollama.com](https://ollama.com)

Make sure the Ollama service is running:

```bash
ollama serve
```

### **2. Pull the model you want to use**

Example:

```bash
ollama pull mistral
```

Or:

```bash
ollama pull llama3
ollama pull qwen2
ollama pull phi3
```

### **3. Create a virtual environment (optional but recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### **4. Install dependencies**

```bash
pip install -r requirements.txt
```

### **5. Run the RAG pipeline**

```bash
python expansion_answer.py
```

---

# 🧠 Notes

* The entire pipeline now works **offline**, using only local models.
* You can switch models by changing the `model="mistral"` argument in the functions that call `ollama.chat()`.
* ChromaDB will automatically store and index embeddings locally.

---

# 🙏 Acknowledgements

Special thanks to **Vincibits** for providing the original inspiration and foundational explanation of the RAG pipeline.
This repository adapts his work to run fully open-source and locally via **Ollama**.

---
