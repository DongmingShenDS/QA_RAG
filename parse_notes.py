import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables from .env file
load_dotenv()
mistral_api_key = os.environ["MISTRAL_API_KEY"]


class QA_PDF:
    """A class to handle PDF processing for question answering using the LangChain and MistralAI models."""

    def __init__(self, chat_model="open-mixtral-8x7b", emb_model="mistral-embed"):
        """Initialize the PDF chat system with necessary configurations for embedding and model interaction."""
        self.model = ChatMistralAI(model=chat_model, temperature=0, mistral_api_key=mistral_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        self.embeddings = MistralAIEmbeddings(model=emb_model, mistral_api_key=mistral_api_key)
        self.system_prompt = (
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context
            to answer the question. If you don't know the answer, just say that you don't know. Use five sentences
            maximum and keep the answer concise. [/INST] </s>
            [INST] Context: {context} [/INST]
            """
        )

    def read_directory(self, directory_path: str):
        """Process all PDF files within the specified directory and load their content into the Chroma vector store."""
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        all_docs = []

        for pdf_file in pdf_files:
            full_path = os.path.join(directory_path, pdf_file)
            docs = PyPDFLoader(file_path=full_path).load()
            all_docs.extend(docs)

        chunks = self.text_splitter.split_documents(all_docs)
        chunks = filter_complex_metadata(chunks)

        # Initialize Chroma vector store with documents
        self.vector_store = Chroma.from_documents(documents=chunks, embedding=self.embeddings,
                                                  persist_directory="chroma_db")
        self.retriever = self.vector_store.as_retriever(search_type="similarity_score_threshold",
                                                        search_kwargs={"k": 5, "score_threshold": 0.5})

        # Set up the processing chain for handling queries
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.model, self.prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)

    def ask(self, query: str):
        """Respond to a query by providing the answer and the list of retrieved context documents."""
        if not self.rag_chain:
            return "No Vector DB.", []
        answer = self.rag_chain.invoke({"input": query})
        return answer

    def clear(self):
        """Clear all stored data and reset the system."""
        self.vector_store = None
        self.retriever = None
        self.chain = None