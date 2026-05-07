# 1. Import Library
from langchain_community.document_loaders import JSONLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings          # ← fixed
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()


def build_index():
    # 2. Load JSON File from Data Directory & Create Document
    jq_schema = ".[] | {instruction: .instruction, input: .input, output: .output}"

    loader = JSONLoader(
        file_path="./data/chatdoctor5k.json",
        jq_schema=jq_schema,
        text_content=False,
    )
    medical_json_docs = loader.load()

    csv_loader = CSVLoader(file_path="./data/format_dataset.csv")
    medical_csv_docs = csv_loader.load()

    medical_docs = medical_csv_docs + medical_json_docs

    # 3. RecursiveCharacter Text Splitter
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", "", ".", ",", ";"],
    )
    recursive_tokens = recursive_splitter.split_documents(medical_docs)

    # 4. Create embeddings using HuggingFaceEmbeddings
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 5. Create Vector Store
    faiss_store = FAISS.from_documents(
        documents=recursive_tokens,
        embedding=hf_embeddings,
    )

    # Persist the vector store
    faiss_store.save_local("faiss_index")
    print("FAISS faiss_index created successfully!")


if __name__ == "__main__":
    build_index()