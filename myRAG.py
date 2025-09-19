#{{ Readme
'''
To do in advance:
Get a node with 2 GPUs, and run a ollama server on each:
OLLAMA_HOST=127.0.0.1:11434 CUDA_VISIBLE_DEVICES=0 ollama serve >& ollama_server1.log &
OLLAMA_HOST=127.0.0.1:11435 CUDA_VISIBLE_DEVICES=1 ollama serve >& ollama_server2.log &
'''
#}}
#{{ Load packages
import os
import readline
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Repress deprecation warning
import warnings
# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#}}
class myRAG:
    #{{ init
    def __init__(self):
        print("RAG init....")
        
        self.llm = ChatOllama(model="llama3.1:70b", base_url="http://localhost:11434")
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11435")

        #test
        assert self.llm.invoke("Hello")
        print("LLM test OK.")
        assert self.embeddings.embed_query("Hello")
        print("Embeddings test OK.")
    #}}
    #{{ Function for saving data to lib
    def data_lib_init(self):
        pdf_folder = "/vsc-hard-mounts/leuven-user/323/vsc32366/projects/LLM/RAG_data"
        documents = []
        for filename in os.listdir(pdf_folder):
            if filename.endswith(".pdf"):
                print(f"Loading {filename}")
                loader = PyPDFLoader(os.path.join(pdf_folder, filename))
                docs = loader.load()
                documents.extend(docs)
        # Split Documents into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=self.embeddings,
            #persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
            )
        self.vector_store.add_documents(texts)
        # retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    #}}
    #{{ Create retriver
    def retriver_init(self, k=1):
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    

    def custom_retreiver(self, query: str) -> str:
        txt=self.retriever.get_relevant_documents(query)
        return "".join([f"<doc id={i} >{x.page_content}</doc>" for i, x in enumerate(txt)])
    #}}
    #{{ Class for ConversationalRunnable
    class ConversationalRunnable(RunnableLambda):
        def __init__(self, llm):
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            self.chat_template=self.get_chat_template()
            self.llm=llm

        def invoke(self, inputs: dict) -> dict:
            # Extract inputs
            context = inputs.get("context", "")
            question = inputs.get("question", "")
            chat_history = self.memory.load_memory_variables({})["chat_history"]

            # Format the prompt
            formatted_prompt = self.chat_template.format_prompt(context=context, chat_history=chat_history, question=question).to_messages()

            # Generate the response
            response = self.llm.invoke(formatted_prompt)

            # Update memory with the new interaction
            self.memory.save_context({"input": question}, {"output": response.content})

            # Return the response and updated memory
            return response

        @staticmethod
        def get_chat_template():
            from langchain.prompts.chat import (
                ChatPromptTemplate,
                SystemMessagePromptTemplate,
                HumanMessagePromptTemplate
            )
            system_template = """
You are an AI assistant. 
- Use <context> as the primary source. Each document is between <doc id=N> and </doc>, where N is the identify of this document.
- Use <chat_history> to maintain context.
- Base answers on retrieved context; do not invent facts.
                    """
            system_prompt = SystemMessagePromptTemplate.from_template(system_template)
            human_template = """
<context>
{context}
</context>

<chat_history>
{chat_history}
</chat_history>

<question>
{question}
</question>
                    """
            human_prompt = HumanMessagePromptTemplate.from_template(human_template)
            return ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    #}}
    #{{ Chat function
    def chat(self):
        print("Let talk :) (type 'bye' to quit)")
        chain=self.ConversationalRunnable(self.llm)
        while True:
            query = input("\nYou: ").strip()
            if query.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break

            #result = qa_chain({"query": query})
            retdoc=self.custom_retreiver(query)
            result = chain.invoke({"question": query, "context": retdoc})
            print(f"\nBot: {result.content}\n")
    #}}
#{{ main
def main():
    rag=myRAG()
    rag.data_lib_init()
    rag.retriver_init()
    rag.chat()
#}}
if __name__ == "__main__":
    main()
