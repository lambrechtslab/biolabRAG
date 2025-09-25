#{{ Readme
'''
To do in advance:
Get a node with 2 GPUs, and run a ollama server on each:
OLLAMA_HOST=127.0.0.1:11434 CUDA_VISIBLE_DEVICES=0 ollama serve >& ollama_server1.log &
OLLAMA_HOST=127.0.0.1:11435 CUDA_VISIBLE_DEVICES=1 ollama serve >& ollama_server2.log &
'''
#}}
#{{ Load packages
# Repress deprecation warning
import warnings
# Suppress LangChain deprecation warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import readline #Make the input() function works better.
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import time
import threading
import warnings
import json
#}}
class RAG:
    #{{ init
    def __init__(self):
        print("RAG init....")
        
        # self.llm = ChatOllama(model="llama3.1:70b", base_url="http://localhost:11434")
        # self.embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11435")
        self.llm = ChatOllama(model="llama3.1:70b")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        #test
        assert self.llm.invoke("Hello")
        print("LLM test OK.")
        assert self.embeddings.embed_query("Hello")
        print("Embeddings test OK.")
        threading.Thread(target=self.keep_warm, args=(300,), daemon=True).start() #Keep LLM warm.
    #}}
    #{{ Keep LLM warm
    def keep_warm(self, interval: int = 300):
        """Send a ping to Ollama every `interval` seconds."""
        while True:
            try:
                _ = self.llm.invoke("ping")  # dummy prompt
            except Exception as e:
                print("Keep-warm failed:", e)
            time.sleep(interval)
    #}}
    #{{ Function for saving data to lib
    def data_lib_init(self):
        pdf_folder = "/vsc-hard-mounts/leuven-user/323/vsc32366/projects/LLM/RAG_data"
        persist_directory = '/vsc-hard-mounts/leuven-user/323/vsc32366/projects/LLM/Chroma_save'
        # documents = []
        if os.path.isdir(persist_directory):
            print("Found RAG library. Loading")
            self.vector_store = Chroma(
                collection_name="example_collection",
                embedding_function=self.embeddings,
                persist_directory=persist_directory,  # Where to save data locally, remove if not necessary
            )
        else:
            print(f"Prepare RAG library to folder: {persist_directory}")
            self.vector_store = Chroma(
                collection_name="example_collection",
                embedding_function=self.embeddings,
                persist_directory=persist_directory,  # Where to save data locally, remove if not necessary
            )
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            for filename in os.listdir(pdf_folder):
                if filename.endswith(".pdf"):
                    print(f"Loading {filename} ...", end='')
                    loader = PyPDFLoader(os.path.join(pdf_folder, filename))
                    docs = loader.load()
                    print(f"Splitting...", end='')
                    chunk = text_splitter.split_documents(docs)
                    print(f"Embedding & saving...", end='')
                    self.vector_store.add_documents(chunk)
                    print("Done.")
    #}}
    #{{ Create retriver
    def retriver_init(self, k=10):
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # def custom_retreiver(self, query: str) -> str:
    #     txt=self.retriever.invoke(query)
    #     files=[]
    #     pages=[]
    #     texts=[]
    #     for x in txt:
    #         texts.append(x.page_content)
    #         files.append(os.path.basename(x.metadata['source']))
    #         pages.append(x.metadata['page_label'])
        
    #     return {'context': '\n'.join([f'<doc id={i} >\n{x.page_content}\n</doc>' for i, x in enumerate(texts)]),
    #             'file': files, 'page':page}
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
- Use <context> as the primary source. Each document is between <doc filename="FILENAME" page="PAGE"> and </doc>.
- Use <chat_history> to maintain context.
- Base answers on retrieved context; do not invent facts.
                    """
#             system_template = """
# You are an AI assistant. 
# - Use <context> as the primary source. Each document is between <doc filename="FILE_NAME" page="PAGE"> and </doc>.
# - Base answers only on retrieved context; do not invent facts.
# - Insert concise citation markers like **[1]**, **[2]**, etc. directly after the relevant statement.
# - If there are any citations, at the end of your answer, add a "References" section that lists each citation number with the corresponding file name and page in the format:
#   **[1]** FILE_NAME p.PAGE
#   **[2]** FILE_NAME p.PAGE
# - If multiple passages support the same point, list all relevant sources under the same number.
# - If no relevant information is found in <context>, say you don't know.
# - Use <chat_history> to maintain conversation context.
# """
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
    #{{ readly
    def ready(self):
        self.data_lib_init()
        self.retriver_init(10)
        self.chain=self.ConversationalRunnable(self.llm)
    #}}
    #{{ Chat function
    def ask(self, query:str) -> str:
        retdoc, retdoc_meta = self.custom_retreiver(query, memory=self.chain.memory)
        result = self.chain.invoke({"question": query, "context": retdoc}).content
        if retdoc_meta:
            doc={}
            for x in retdoc_meta:
                if not(x['filename'] in doc):
                    doc[x['filename']]=set()
                doc[x['filename']].add(x['page'])

            refstr=[]
            for filename, pageset in doc.items():
                refstr.append(filename+' (p.'+', '.join(sorted(pageset))+')')
                
            result = result + '\n\n_References:_\n' +\
                "\n".join([f"- _{x}_" for x in refstr])
        return result
        
    def chat(self):
        print("Let talk :) (type 'bye' to quit)")
        while True:
            query = input("\nYou: ").strip()
            if query.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break

            #result = qa_chain({"query": query})
            answer = self.ask(query)
            print(f"\nBot: {answer}\n")
    #}}
    #{{ QA if need retrival
    def QA_need_retrival(self, quest:str, memory):
        system_template = """
You are a classifier. Decide whether a user’s question (see <UserQuestion>) should trigger retrieval from an external knowledge source. 

Decision rules:
- Use the chat history (<ChatHistory>) to resolve pronouns, references, and context in the user’s latest question.  
- Output "RETRIEVAL_NEEDED" if retrieval could *probably* improve the answer, even if the question might be answerable without it. 
- Output "NO_RETRIEVAL_NEEDED" only if retrieval would add no meaningful value (e.g., simple math, logic puzzles, or widely known facts such as “What is 2+2?” or “Who wrote Hamlet?”).

Output format:
- Output either "RETRIEVAL_NEEDED" or "NO_RETRIEVAL_NEEDED"
- No any other texts.
        """
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = """
<ChatHistory>
{chat_history}
</ChatHistory>

<UserQuestion>
{question}
</UserQuestion>
        """
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_history = memory.load_memory_variables({})["chat_history"]
        return self.llm.invoke(ChatPromptTemplate.from_messages([system_prompt, human_prompt]
            ).format_prompt(question=quest, chat_history=chat_history).to_messages()
           ).content == "RETRIEVAL_NEEDED"
    #}}
    #{{ QA rewrite user's question for retrival
    def QA_rewrite_question_for_retrival(self, quest:str, memory):
        system_template = """
You are a query rewriter for a Retrieval-Augmented Generation (RAG) system. 
Your task is to rewrite the user’s latest question (see <UserQuestion>) into one or more clear, self-contained search queries 
that are optimized for semantic similarity search using embeddings.

Guidelines:
- Use the chat history (<ChatHistory>) to resolve pronouns, references, and context in the user’s latest question.
- Preserve the user’s original intent.
- When useful, break the question into multiple sub-queries that capture different aspects of the information need.
- You may also generate "step-back" queries (broader or higher-level versions of the question) to improve retrieval coverage.
- Make the queries explicit and unambiguous (expand pronouns and vague references).
- Remove polite phrases or conversational fluff.
- Be concise and focus only on the key information to retrieve.
- Do not add information not present in the chat history or user’s question.

Output:
Provide the rewritten queries as plain text, each query on its own line, with no numbering or bullets. Do not output explanations.
        """
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = """
<ChatHistory>
{chat_history}
</ChatHistory>

<UserQuestion>
{question}
</UserQuestion>
    """
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_history = memory.load_memory_variables({})["chat_history"]
        return self.llm.invoke(ChatPromptTemplate.from_messages([system_prompt, human_prompt]).format_prompt(question=quest, chat_history=chat_history).to_messages()).content
    #}}
    #{{ QA if tetrivalled documents is relevant
    def QA_if_doc_relevant_to_question(self, quest:str, doc:str, memory):
        system_template = """
You are a relevance classifier for a retrieval-augmented generation (RAG) system. 
Your task is to decide whether the retrieved document (see <RetrievedDocument>) is relevant to the user’s question (see <UserQuestion>). 

Decision rules:
- Use the chat history (<ChatHistory>) to resolve pronouns, references, and context in the user’s latest question.  
- Output "RELEVANT" if the document contains information that could help answer the question, 
  even if partially or indirectly.
- Output "NOT_RELEVANT" if the document does not provide useful information for answering the question.

Output format:
- Output either "RELEVANT" or "NOT_RELEVANT"
- No any other texts.
        """
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = """
<ChatHistory>
{chat_history}
</ChatHistory>
        
<UserQuestion>
{question}
</UserQuestion>

<RetrievedDocument>
{retrieved_document}
</RetrievedDocument>
        """
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_history = memory.load_memory_variables({})["chat_history"]
        decision = self.llm.invoke(ChatPromptTemplate.from_messages([system_prompt, human_prompt])
                               .format_prompt(question=quest, retrieved_document=doc, chat_history=chat_history).to_messages()).content
        if decision == "RELEVANT":
            return True
        elif decision == "NOT_RELEVANT":
            return False
        else:
            warnings.warn(f"LLM unexpected output: ${decision}")
            return False
    #}}
    #{{ QA if a list of documents are relevant
    def QA_if_lis_of_docs_relevant_to_question(self, quest:str, docs:list, memory):
        system_template = """
You are a relevance classifier for a retrieval-augmented generation (RAG) system. 
Your task is to decide whether each of the retrieved documents (see <RetrievedDocument>) is relevant to the user’s question (see <UserQuestion>). Each retrieved document is warped between <doc id=N> and </doc>, where N is 0,1,2...

Decision rules:
- Use the chat history (<ChatHistory>) to resolve pronouns, references, and context in the user’s latest question.  
- Label "RELEVANT" if the document contains information that could help answer the question, 
  even if partially or indirectly.
- Label "NOT_RELEVANT" if the document does not provide useful information for answering the question.

Output strictly in JSON format as a list of objects. No any other contents.
Output format example:
[
  {{"id": 0, "relevance": "NOT_RELEVANT"}},
  {{"id": 1, "relevance": "RELEVANT"}},
  {{"id": 2, "relevance": "NOT_RELEVANT"}},
  ...
]
        """
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = """
<ChatHistory>
{chat_history}
</ChatHistory>
        
<UserQuestion>
{question}
</UserQuestion>

<RetrievedDocument>
{retrieved_document}
</RetrievedDocument>
        """
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_history = memory.load_memory_variables({})["chat_history"]
        doc = '\n'.join([f"<doc id={i}>\n{x['context']}\n</doc>" for i, x in enumerate(docs)])
        decision = self.llm.invoke(ChatPromptTemplate.from_messages([system_prompt, human_prompt])
                               .format_prompt(question=quest, retrieved_document=doc, chat_history=chat_history).to_messages()).content
        index=[]
        try:
            t = json.loads(decision)
        except Exception as e:
            warnings.warn(f"Error in parse below JSON:\n{decision}\nError: {e.message}")
            return []
            
        for x in t:
            if x['relevance']=="RELEVANT":
                index.append(int(x['id']))
        return sorted(index)
    #}}
    #{{ Reload custom_retreiver
    def custom_retreiver(self, query: str, memory) -> str:
        def parse_doc(txt):
            filename = os.path.basename(txt.metadata["source"])
            page = txt.metadata["page_label"]
            return {'context': f'<doc filename="{filename}" page="{page}"  >\n{txt.page_content}\n</doc>',
                 'filename': filename, 'page': page}
            
        if not(self.QA_need_retrival(query, memory=memory)):
            return ("No context needed.", [])
        print("Retrival...")
        squerys = self.QA_rewrite_question_for_retrival(query, memory=memory)
        doc = {}
        for squery in squerys.splitlines():
            # baddoc=set()
            print(f"Retrival quest: {squery}")
            txt = list(map(parse_doc, self.retriever.invoke(squery)))
            txt =[ x for x in txt if not(x['context'] in doc) ]
            print(f"{len(txt)} nonduplicated document(s) are found.")
            if txt:
                l = self.QA_if_lis_of_docs_relevant_to_question(query, docs=txt, memory=memory)
                print(f"{len(l)} document(s) are relavent.")
                for i in l:
                    doc[txt[i]['context']]=txt[i]
            # for i, x in enumerate(txt):
            #     print(f"Check doc {x['filename']} p.{x['page']}...", end='')
            #     if (x['context'] in doc) or (x['context'] in baddoc):
            #         print("Duplicate.")
            #     elif self.QA_if_doc_relevant_to_question(squery, x['context'], memory=memory):
            #         doc[x['context']]=x
            #         print("Relevant.")
            #     else:
            #         baddoc.add(x['context'])
            #         print("Not relevant.")

        return ("\n".join([x['context'] for x in doc.values()]), list(doc.values()))
    #}}
if __name__ == "__main__":
    rag=RAG()
    rag.ready()
    rag.chat()
