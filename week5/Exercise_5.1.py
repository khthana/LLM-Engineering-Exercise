#!/usr/bin/env python3
"""
Exercise_5.1: Insurellm RAG — Local Ollama + Gradio UI
Adapted from day1.ipynb (brute-force RAG), day2.ipynb (chunk/embed/visualize)
and day3.ipynb (vector RAG chat).

Swaps OpenAI for a local Ollama chat model and OpenAI/HuggingFace-MiniLM
embeddings for local BGE-M3 embeddings (HuggingFace, runs on GPU via
sentence-transformers — no Ollama pull needed for embeddings).

This script does NOT start Ollama or contact it at import time — nothing
runs until you click a button/send a chat message in the UI. Before using
tabs 1 or 3, make sure `ollama serve` is running and the chat model below
has been pulled (`ollama pull <CHAT_MODEL>`).
"""

import glob
import os
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.manifold import TSNE

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

# ============================================================================
# CONFIG
# ============================================================================
KB_DIR = Path("week5/knowledge-base")
DB_DIR = Path("week5/vector_db")

CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:e4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = "BAAI/bge-m3"

DOC_TYPE_COLORS = {"products": "blue", "employees": "green", "contracts": "red", "company": "orange"}


def _history_to_messages(history: list[dict]) -> list:
    messages = []
    for turn in history:
        role, content = turn.get("role"), turn.get("content")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


# ============================================================================
# LAZY SINGLETONS — nothing touches Ollama/downloads models until first use
# ============================================================================
_llm = None
_embeddings = None
_vectorstore = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    return _llm


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        if not DB_DIR.exists():
            raise gr.Error("Vector store not found — build it in the 'Build & Visualize' tab first.")
        _vectorstore = Chroma(persist_directory=str(DB_DIR), embedding_function=get_embeddings())
    return _vectorstore


# ============================================================================
# TAB 1: BRUTE-FORCE RAG  (day1.ipynb)
# ============================================================================
BRUTE_FORCE_SYSTEM_PREFIX = """
You represent Insurellm, the Insurance Tech company.
You are an expert in answering questions about Insurellm; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.

Relevant context:
"""

_knowledge: dict[str, str] | None = None


def load_knowledge() -> dict[str, str]:
    global _knowledge
    if _knowledge is not None:
        return _knowledge

    knowledge: dict[str, str] = {}
    for filename in glob.glob(str(KB_DIR / "employees" / "*")):
        name = Path(filename).stem.split(" ")[-1]
        knowledge[name.lower()] = Path(filename).read_text(encoding="utf-8")
    for filename in glob.glob(str(KB_DIR / "products" / "*")):
        name = Path(filename).stem
        knowledge[name.lower()] = Path(filename).read_text(encoding="utf-8")

    _knowledge = knowledge
    return knowledge


def get_relevant_context_brute_force(message: str) -> list[str]:
    knowledge = load_knowledge()
    text = "".join(ch for ch in message if ch.isalpha() or ch.isspace())
    words = text.lower().split()
    return [knowledge[word] for word in words if word in knowledge]


def additional_context_brute_force(message: str) -> str:
    relevant_context = get_relevant_context_brute_force(message)
    if not relevant_context:
        return "There is no additional context relevant to the user's question."
    return "The following additional context might be relevant in answering the user's question:\n\n" + "\n\n".join(
        relevant_context
    )


def chat_brute_force(message: str, history: list[dict]) -> str:
    system_message = BRUTE_FORCE_SYSTEM_PREFIX + additional_context_brute_force(message)
    messages = [SystemMessage(content=system_message)] + _history_to_messages(history) + [HumanMessage(content=message)]
    response = get_llm().invoke(messages)
    return response.content


# ============================================================================
# TAB 2: BUILD & VISUALIZE VECTOR STORE  (day2.ipynb)
# ============================================================================
def build_vector_store() -> str:
    global _vectorstore

    if not KB_DIR.exists():
        return f"Knowledge base not found at {KB_DIR.resolve()}"

    documents = []
    for folder in KB_DIR.iterdir():
        if not folder.is_dir():
            continue
        loader = DirectoryLoader(str(folder), glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        for doc in loader.load():
            doc.metadata["doc_type"] = folder.name
            documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    if DB_DIR.exists():
        Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=str(DB_DIR))
    _vectorstore = vectorstore  # refresh the cached singleton used by tab 3

    count = vectorstore._collection.count()
    dims = len(vectorstore._collection.get(limit=1, include=["embeddings"])["embeddings"][0])

    return (
        f"Loaded {len(documents)} documents -> {len(chunks)} chunks\n"
        f"Vector store persisted to {DB_DIR.resolve()}\n"
        f"{count:,} vectors, {dims:,} dimensions ({EMBED_MODEL})"
    )


def visualize(n_components: int):
    if not DB_DIR.exists():
        raise gr.Error("Vector store not found — click 'Build / rebuild vector store' first.")

    collection = Chroma(persist_directory=str(DB_DIR), embedding_function=get_embeddings())._collection
    result = collection.get(include=["embeddings", "documents", "metadatas"])
    vectors = np.array(result["embeddings"])
    if len(vectors) < 3:
        raise gr.Error("Not enough vectors to visualize yet.")

    documents = result["documents"]
    doc_types = [metadata["doc_type"] for metadata in result["metadatas"]]
    colors = [DOC_TYPE_COLORS.get(t, "gray") for t in doc_types]
    hover_text = [f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)]

    tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(vectors) - 1))
    reduced = tsne.fit_transform(vectors)

    if n_components == 3:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    z=reduced[:, 2],
                    mode="markers",
                    marker=dict(size=5, color=colors, opacity=0.8),
                    text=hover_text,
                    hoverinfo="text",
                )
            ]
        )
        fig.update_layout(
            title="3D Chroma Vector Store Visualization",
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
            margin=dict(r=10, b=10, l=10, t=40),
        )
    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    mode="markers",
                    marker=dict(size=5, color=colors, opacity=0.8),
                    text=hover_text,
                    hoverinfo="text",
                )
            ]
        )
        fig.update_layout(title="2D Chroma Vector Store Visualization", margin=dict(r=20, b=10, l=10, t=40))

    return fig


# ============================================================================
# TAB 3: VECTOR RAG CHAT  (day3.ipynb)
# ============================================================================
RAG_SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""


def answer_question(message: str, history: list[dict]) -> str:
    retriever = get_vectorstore().as_retriever()
    docs = retriever.invoke(message)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = RAG_SYSTEM_PROMPT_TEMPLATE.format(context=context)
    messages = [SystemMessage(content=system_prompt)] + _history_to_messages(history) + [HumanMessage(content=message)]
    response = get_llm().invoke(messages)
    return response.content


# ============================================================================
# GRADIO UI
# ============================================================================
def build_ui():
    with gr.Blocks(title="Insurellm RAG — Local Ollama") as demo:
        gr.Markdown(
            f"# Insurellm Expert Knowledge Worker (RAG)\n"
            f"Adapted from `week5/day1.ipynb`-`day3.ipynb` to run fully locally: chat via Ollama "
            f"model `{CHAT_MODEL}` ({OLLAMA_BASE_URL}), embeddings via HuggingFace `{EMBED_MODEL}`.\n\n"
            f"**Before chatting:** make sure `ollama serve` is running and `ollama pull {CHAT_MODEL}` "
            f"has been done."
        )

        with gr.Tabs():
            with gr.Tab("1. Brute-force RAG"):
                gr.Markdown(
                    "Naive keyword-match RAG (Day 1): scans the question for words that exactly match "
                    "an employee surname or product name and injects the matching document verbatim as "
                    "context. No embeddings or vector search involved."
                )
                gr.ChatInterface(chat_brute_force)

            with gr.Tab("2. Build & Visualize Vector Store"):
                gr.Markdown(
                    "Chunk the knowledge base, embed with BGE-M3, and persist to a local Chroma store "
                    "(Day 2). Run this once — or whenever `knowledge-base/` changes — before using the "
                    "'Vector RAG Chat' tab."
                )
                build_btn = gr.Button("Build / rebuild vector store", variant="primary")
                build_status = gr.Textbox(label="Status", lines=3, interactive=False)
                build_btn.click(build_vector_store, outputs=[build_status])

                gr.Markdown("### Visualize the stored vectors (t-SNE)")
                with gr.Row():
                    viz_2d_btn = gr.Button("2D plot")
                    viz_3d_btn = gr.Button("3D plot")
                viz_plot = gr.Plot(label="Vector store visualization")
                viz_2d_btn.click(lambda: visualize(2), outputs=[viz_plot])
                viz_3d_btn.click(lambda: visualize(3), outputs=[viz_plot])

            with gr.Tab("3. Vector RAG Chat"):
                gr.Markdown(
                    "Full RAG (Day 3): Chroma similarity search over the persisted vector store feeds "
                    "context into the Ollama chat model. Build the vector store in tab 2 first."
                )
                gr.ChatInterface(answer_question)

    return demo


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("Exercise 5.1: Insurellm RAG - Local Ollama")
    print("=" * 70)
    print(f"Chat model (Ollama):        {CHAT_MODEL} @ {OLLAMA_BASE_URL}")
    print(f"Embedding model (HF/local): {EMBED_MODEL}")
    print(f"Knowledge base:             {KB_DIR.resolve()}")
    print(f"Vector store:               {DB_DIR.resolve()}")
    print()
    print("This script does not start Ollama for you.")
    print(f"Run `ollama serve` and `ollama pull {CHAT_MODEL}` separately before chatting.")
    print()

    demo = build_ui()
    demo.launch()
