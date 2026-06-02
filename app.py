import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()

st.set_page_config(
    page_title="Interactive PDF Reader",
    page_icon="PDF",
    layout="wide",
)


def file_digest(uploaded_file) -> str:
    uploaded_file.seek(0)
    digest = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return digest


def provider_env_key(provider: str) -> str:
    if provider == "Google Gemini":
        return "GOOGLE_API_KEY"
    return "OPENAI_API_KEY"


def default_chat_model(provider: str) -> str:
    if provider == "Google Gemini":
        return os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash")
    return os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")


def default_embedding_model(provider: str) -> str:
    if provider == "Google Gemini":
        return os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
    return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def make_embeddings(provider: str, embedding_model: str):
    if provider == "Google Gemini":
        return GoogleGenerativeAIEmbeddings(model=embedding_model)
    return OpenAIEmbeddings(model=embedding_model)


@st.cache_resource(show_spinner=False)
def build_vectorstore(
    pdf_bytes: bytes,
    file_name: str,
    chunk_size: int,
    chunk_overlap: int,
    provider: str,
    embedding_model: str,
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        loader = PyPDFLoader(str(tmp_path))
        pages = loader.load()
    finally:
        tmp_path.unlink(missing_ok=True)

    for page in pages:
        page.metadata["source"] = file_name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    embeddings = make_embeddings(provider, embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, pages, len(chunks)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful PDF reading assistant. Answer only from the provided "
            "PDF context. If the answer is not in the context, say that you cannot "
            "find it in the document.\n\nPDF context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


def make_llm(provider: str, chat_model: str, temperature: float):
    if provider == "Google Gemini":
        llm = ChatGoogleGenerativeAI(model=chat_model, temperature=temperature)
    else:
        llm = ChatOpenAI(model=chat_model, temperature=temperature)
    return ANSWER_PROMPT | llm


def format_context(docs) -> str:
    return "\n\n".join(
        f"[{source_label(doc)}]\n{doc.page_content.strip()}" for doc in docs
    )


def format_history(messages):
    history = []
    for message in messages:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            history.append(AIMessage(content=message["content"]))
    return history


def answer_question(
    vectorstore,
    question: str,
    messages,
    provider: str,
    chat_model: str,
    temperature: float,
    top_k: int,
):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)
    llm_chain = make_llm(provider, chat_model, temperature)
    response = llm_chain.invoke(
        {
            "context": format_context(docs),
            "chat_history": format_history(messages),
            "question": question,
        }
    )
    return response.content, docs


def reset_chat():
    st.session_state.messages = []


def source_label(doc) -> str:
    page = doc.metadata.get("page")
    page_text = f"page {int(page) + 1}" if page is not None else "unknown page"
    source = doc.metadata.get("source", "PDF")
    return f"{source}, {page_text}"


st.title("Interactive PDF Reader")

with st.sidebar:
    st.header("Document")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    st.header("Retrieval")
    chunk_size = st.slider("Chunk size", 500, 2500, 1200, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, 25)
    top_k = st.slider("Source chunks", 2, 8, 4)
    temperature = st.slider("Creativity", 0.0, 1.0, 0.1, 0.1)

    st.header("Model")
    provider_default = 0 if os.getenv("AI_PROVIDER", "Google Gemini") == "Google Gemini" else 1
    provider = st.selectbox(
        "Provider",
        ["Google Gemini"],
        index=provider_default,
    )
    chat_model = st.text_input("Chat model", value=default_chat_model(provider))
    embedding_model = st.text_input(
        "Embedding model",
        value=default_embedding_model(provider),
    )
    st.text_input(
        "Required key",
        value=provider_env_key(provider),
        disabled=True,
    )
    st.caption("Set this key in your environment or .env file, then restart Streamlit.")

    if st.button("Clear chat", use_container_width=True):
        reset_chat()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_digest" not in st.session_state:
    st.session_state.active_digest = None

if not uploaded_pdf:
    st.info("Upload a PDF to start reading and asking questions.")
    st.stop()

api_key_name = provider_env_key(provider)
if not os.getenv(api_key_name):
    st.error(f"Set {api_key_name} in your environment or in a .env file, then restart Streamlit.")
    st.stop()

digest = file_digest(uploaded_pdf)
settings_key = f"{digest}:{chunk_size}:{chunk_overlap}:{provider}:{embedding_model}"

if st.session_state.active_digest != settings_key:
    reset_chat()
    st.session_state.active_digest = settings_key

pdf_bytes = uploaded_pdf.getvalue()

with st.spinner("Reading and indexing the PDF..."):
    vectorstore, pages, chunk_count = build_vectorstore(
        pdf_bytes,
        uploaded_pdf.name,
        chunk_size,
        chunk_overlap,
        provider,
        embedding_model,
    )

reader_col, chat_col = st.columns([0.9, 1.1], gap="large")

with reader_col:
    st.subheader(uploaded_pdf.name)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Pages", len(pages))
    metric_cols[1].metric("Chunks", chunk_count)
    metric_cols[2].metric("Retrieval", f"Top {top_k}")

    selected_page = st.number_input(
        "Page",
        min_value=1,
        max_value=max(len(pages), 1),
        value=1,
        step=1,
    )
    page = pages[selected_page - 1]
    st.text_area(
        "Page text",
        value=page.page_content.strip() or "No extractable text found on this page.",
        height=520,
    )

with chat_col:
    st.subheader("Ask the PDF")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['label']}**")
                        st.caption(source["snippet"])

    prompt = st.chat_input("Ask a question about this PDF")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the document..."):
                answer, source_documents = answer_question(
                    vectorstore,
                    prompt,
                    st.session_state.messages[:-1],
                    provider,
                    chat_model,
                    temperature,
                    top_k,
                )

            sources = [
                {
                    "label": source_label(doc),
                    "snippet": doc.page_content.strip()[:700],
                }
                for doc in source_documents
            ]

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"**{source['label']}**")
                        st.caption(source["snippet"])

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
