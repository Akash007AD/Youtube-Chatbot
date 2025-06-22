import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import re

# --- UI CONFIG ---
st.set_page_config(page_title="YouTube RAG Chatbot", layout="centered")
st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 10px 15px;
        border-radius: 10px;
        margin: 10px 0;
        max-width: 80%;
        font-size: 16px;
        line-height: 1.6;
        word-wrap: break-word;
    }

    .user {
        background-color: #dcf8c6;  /* Light green */
        color: #000;
        margin-left: auto;
        margin-right: 0;
        text-align: right;
    }

    .bot {
        background-color: #ffffff;  /* White */
        color: #000;
        margin-left: 0;
        margin-right: auto;
        text-align: left;
        border: 1px solid #ccc;
    }

    /* Background for whole app */
    .main {
        background-color: #111 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("💬 YouTube Transcript Chatbot")
st.caption("Paste a YouTube video URL and ask questions based on its transcript.")

# --- Extract YouTube Video ID ---
def extract_video_id(url):
    patterns = [
        r"v=([^&]+)",                      # regular watch link
        r"youtu\.be/([^?&]+)",             # short link
        r"youtube\.com/embed/([^?&]+)",    # embed
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# --- Transcript + Retriever Pipeline Setup ---
@st.cache_resource(show_spinner="Loading video transcript and embedding…")
@st.cache_resource(show_spinner="Loading video transcript and embedding…")
def setup_chain(video_id):
    transcript = ""
    try:
        # Try English first
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    except:
        try:
            # If English fails, try Hindi (auto-generated)
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi"])
        except TranscriptsDisabled:
            return None, "❌ Transcript not available for this video."
        except Exception as e:
            return None, f"❌ Failed to retrieve transcript: {e}"

    transcript = " ".join(chunk['text'] for chunk in transcript_list)

    # Proceed as before
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = FAISS.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer only from the provided transcript content.
        If the context is insufficient, just say you don't know.

        {context}

        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    llm = ChatOllama(model="llama3")
    parser = StrOutputParser()

    final_chain = parallel_chain | prompt | llm | parser
    return final_chain, None

# --- Sidebar: Paste Video URL ---
youtube_url = st.text_input("📺 Paste YouTube Video URL:", placeholder="e.g. https://www.youtube.com/watch?v=yVTNge3sXpg")

if youtube_url:
    video_id = extract_video_id(youtube_url)
    if not video_id:
        st.error("⚠️ Could not extract video ID from URL.")
    else:
        chain, error_msg = setup_chain(video_id)
        if error_msg:
            st.error(error_msg)
        else:
            # --- Chat Session Memory ---
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # --- Input Box ---
            with st.form("chat_form", clear_on_submit=True):
                user_query = st.text_input("💬 Your question:", placeholder="Ask about the video content...", label_visibility="collapsed")
                submitted = st.form_submit_button("Send")

            # --- Ask the LLM ---
            if submitted and user_query:
                with st.spinner("Thinking..."):
                    try:
                        answer = chain.invoke(user_query)
                        st.session_state.chat_history.append(("user", user_query))
                        st.session_state.chat_history.append(("bot", answer))
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            # --- Show Chat History ---
            for role, msg in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f"<div class='stChatMessage user'>{msg}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='stChatMessage bot'>{msg}</div>", unsafe_allow_html=True)
