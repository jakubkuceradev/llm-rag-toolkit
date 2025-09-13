import os
import pickle

from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_litellm import ChatLiteLLM
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

VECTORSTORE_PATH = "data/vectorstore"
DOCSTORE_PATH = "data/docstore.pkl"
DOCUMENTS_PATH = "assets/fit-html"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

<context>
{context}
</context>

Answer the question based on the above context:

<question>
{question}
</question>
"""


def load_html_documents() -> list[Document]:
    """Loads HTML documents from the specified directory."""
    print("Loading HTML documents...")
    loader = DirectoryLoader(
        DOCUMENTS_PATH,
        glob="*.html",
        recursive=False,
        loader_cls=UnstructuredHTMLLoader,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents


def get_or_create_retriever() -> ParentDocumentRetriever:
    """
    Creates a ParentDocumentRetriever, loading from disk if it exists,
    otherwise creating and persisting it.
    """
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Check if the vector store and docstore already exist
    if os.path.exists(VECTORSTORE_PATH) and os.path.exists(DOCSTORE_PATH):
        print("Loading existing vector store and document store from disk...")
        vectorstore = Chroma(
            collection_name="ma2_scriptum_local",
            embedding_function=embeddings,
            persist_directory=VECTORSTORE_PATH,
        )
        with open(DOCSTORE_PATH, "rb") as f:
            store = pickle.load(f)

        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
        )
    else:
        print("Creating new vector store and document store...")
        documents = load_html_documents()

        vectorstore = Chroma(
            collection_name="ma2_scriptum_local",
            embedding_function=embeddings,
            persist_directory=VECTORSTORE_PATH,
        )
        store = InMemoryStore()

        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
        )
        print("Adding documents to the retriever...")
        retriever.add_documents(documents, ids=None)

        # Persist the document store to disk
        print("Persisting document store...")
        os.makedirs(os.path.dirname(DOCSTORE_PATH), exist_ok=True)
        with open(DOCSTORE_PATH, "wb") as f:
            pickle.dump(store, f)
        print("Persistence complete.")

    return retriever


def format_document_context(document: Document) -> str:
    """Formats a single document for inclusion in the prompt context."""
    metadata = " ".join(
        f'{name}="{value}"' for name, value in document.metadata.items()
    )
    header = f"<document {metadata}>\n"
    footer = "\n</document>\n"
    context = header + document.page_content + footer
    return context


def format_prompt(query: str, documents: list[Document]) -> str:
    """Formats the final prompt with the retrieved context and query."""
    context = "\n".join(format_document_context(doc) for doc in documents)
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    return prompt


def generate_response(prompt: str) -> str:
    """Generates a response using the local Ollama LLM."""
    model = ChatOllama(model="gpt-oss:20b")
    response = model.invoke(prompt)
    return str(response.content)


def generate_response_stream(prompt: str) -> str:
    """
    Generates and streams the response from the local Ollama LLM,
    printing each chunk to stdout in real time.
    """
    model = ChatLiteLLM(model="gemini/gemini-2.5-flash")
    full_response = ""

    for chunk in model.stream(prompt):
        content = str(chunk.content)
        print(content, end="", flush=True)
        full_response += content

    return full_response


def main():
    """Main function to run the RAG pipeline."""
    print("# MA2\n")
    retriever = get_or_create_retriever()
    query = input("Ask a quetion!\n")
    print(f"\nInvoking retriever with the query: '{query}'")
    documents = retriever.invoke(query)
    if not documents:
        print("No relevant documents found for your query.")
        return
    print("Formatting a prompt...")
    prompt = format_prompt(query, documents)
    print("Generating a response with the local LLM (streaming)...\n")
    print("## STREAMED RESPONSE\n")
    response = generate_response_stream(prompt)
    sources = [doc.metadata.get("source", "Unknown source") for doc in documents]
    print(f"\n\n## PROMPT\n{prompt}\n\n## RESPONSE\n{response}\n\n## SOURCES\n")
    for source in set(sources):
        print(source)


if __name__ == "__main__":
    main()
