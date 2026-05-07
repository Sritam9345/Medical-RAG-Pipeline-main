import os
import time
from dotenv import load_dotenv

# Document loading (unchanged)
from langchain_community.document_loaders import PyPDFLoader

# Moved from langchain.text_splitter → langchain_text_splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store (unchanged)
from langchain_community.vectorstores import FAISS

# Moved from langchain_community.embeddings → langchain_huggingface
from langchain_huggingface import HuggingFaceEmbeddings

# LLM (unchanged)
from langchain_google_genai import ChatGoogleGenerativeAI

# Modern prompt & LCEL imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

# Replaces ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

# In-memory session store (keyed by session_id)
_session_store: dict[str, ChatMessageHistory] = {}


def _get_session_history(session_id: str) -> ChatMessageHistory:
    """Return (or create) the chat history for a given session."""
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def load_pipeline():
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_index", hf_embeddings, allow_dangerous_deserialization=True
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        max_tokens=2048,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 2}
    )

    # ChatPromptTemplate replaces PromptTemplate;
    # MessagesPlaceholder injects the full chat history automatically.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a medical consultant with expertise in understanding doctor-patient
conversations, symptom descriptions and medical chat transcripts.
Use only the information provided from the 'Medical Care and Chats' dataset to answer
the user's question. Stay strictly within the given chats, symptoms, diagnoses and
conversation notes. If the dataset contains the relevant information, provide a clear,
short and medically accurate response. If the answer is not present in the dataset,
say "The answer is not available in provided context."

Context:
{context}""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),  # injected by RunnableWithMessageHistory
        ("human", "{question}"),
    ])

    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain: retrieve → format → prompt → llm → parse
    core_chain = (
        RunnablePassthrough.assign(
            context=lambda x: _format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap with session-aware message history (replaces ConversationBufferMemory)
    chain_with_history = RunnableWithMessageHistory(
        core_chain,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    # Return chain + retriever as a tuple so ask_question can fetch source docs
    return chain_with_history, retriever


def ask_question(pipeline, question: str, session_id: str = "default"):
    chain_with_history, retriever = pipeline
    start = time.time()

    # Retrieve source docs independently (chain output is now a plain string)
    docs = retriever.invoke(question)

    answer = chain_with_history.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )

    latency = time.time() - start

    return {
        "answer": answer,
        "retrieved_docs": [doc.page_content[:200] for doc in docs],
        "sources": [doc.metadata for doc in docs],
        "latency": latency,
    }


# ── Usage ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = load_pipeline()

    response = ask_question(pipeline, "What are the symptoms of diabetes?")
    print("Answer   :", response["answer"])
    print("Latency  :", f"{response['latency']:.2f}s")
    print("Sources  :", response["sources"])