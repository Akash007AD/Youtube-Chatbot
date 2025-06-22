# 💬 YouTube Transcript Chatbot

An interactive chatbot that allows users to ask questions about the content of any YouTube video using its transcript. Built using **Streamlit**, **LangChain**, **LLaMA 3 via Ollama**, **FAISS**, and **HuggingFace embeddings**.

---

## 🚀 Features

* 🔍 Extracts transcripts from YouTube videos automatically
* 💡 Answers user queries based only on video content (RAG approach)
* 🤖 Powered by LLaMA 3 via Ollama for natural language responses
* 🧠 Uses FAISS and HuggingFace embeddings for semantic search
* 🖼️ Clean, chat-style user interface built with Streamlit

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** – UI framework
* **YouTubeTranscriptAPI** – For fetching transcripts
* **LangChain** – For chaining retriever and LLM
* **FAISS** – Vector store for transcript chunks
* **HuggingFace (Sentence Transformers)** – Embeddings
* **Ollama** – Local LLaMA 3 model backend

---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/youtube-transcript-chatbot.git
cd youtube-transcript-chatbot
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run Ollama (if not already installed)

```bash
ollama run llama3
```

### 4. Launch the App

```bash
streamlit run app.py
```

---

## 📺 How It Works

1. Paste a YouTube video URL in the app.
2. The app extracts and embeds the transcript.
3. Ask questions in natural language.
4. The app retrieves relevant content and responds based on it.

---

## ❗ Limitations

* Transcript must be available (either in English or Hindi).
* LLaMA 3 model should be set up locally via Ollama.

---

## 📄 License

MIT License.
Feel free to use, modify, and build upon this project.

---