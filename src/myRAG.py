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

import os, re
import readline #Make the input() function works better.
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
# from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import time
import threading
import warnings
import json
import requests
from xml.etree import ElementTree as ET
import numpy as np
def sigmoid(x):
    return (1/(1 + np.exp(-np.array(x)))).tolist()
#}}
class RAG:
    #{{ init
    def __init__(self, crossencoder_normscore_cutoff:float=0.6, crossencoder_normscore_cutoff_loose:float=0.3):
        self.crossencoder_normscore_cutoff = crossencoder_normscore_cutoff
        self.crossencoder_normscore_cutoff_loose = crossencoder_normscore_cutoff_loose
        self.chunk_size_by_char = 1500    # ~400 tokens
        self.chunk_overlap_by_char = 300  # ~80 tokens
        
        print("RAG init....")
        
        # self.llm = ChatOllama(model="llama3.1:70b", base_url="http://localhost:11434")
        # self.embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11435")
        # # Using LLama3.1 70b
        self.num_ctx = 16384
        self.llm = ChatOllama(model="llama3.1:70b",
                              num_ctx=self.num_ctx,    # desired context window fastest: 8192; balanced: 16384 / 32768; max: 131072
                              num_keep=256,            # keep system/instructions when sliding
                              temperature=0)

        # # Using OpenBioLLM
        # self.num_ctx = 8192
        # self.llm = ChatOllama(model="s10350330/openbiollm-llama3-70b.i1-q4_k_m.gguf",
        #                       num_ctx=self.num_ctx,
        #                       num_keep=256,            # keep system/instructions when sliding
        #                       temperature=0.7,         # Slight creativity for fluent writing
        #                       top_p=0.9,               # Diverse but focused sampling
        #                       top_k=50,                # Helps content variety
        #                       repetition_penalty=1.05, # Avoids loops without truncating
        #                       num_predict=1800         # Ensures multi-paragraph output
        #                       )
        
        self.chat_history_num_ctx = round(self.num_ctx / 3)
        self.doc_num_ctx = round(self.num_ctx / 2)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.crossencoder = CrossEncoder(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2",
                                      device="cpu")

        #test
        assert self.llm.invoke("Hello")
        print("LLM test OK.")
        assert self.embeddings.embed_query("Hello")
        print("Embeddings test OK.")
        assert self.crossencoder.predict([("How many people live in Berlin?", "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.")])
        print("Cross Encoder test OK.")
        
        threading.Thread(target=self.keep_warm, args=(60,), daemon=True).start() #Keep LLM warm.
        
    #}}
    #{{ clear_chat_history
    def clear_chat_history(self):
        self.chain.memory.clear()
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
        persist_directory = '/vsc-hard-mounts/leuven-user/323/vsc32366/projects/LLM/Chroma_save_big'
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
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size_by_char, chunk_overlap=self.chunk_overlap_by_char)
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
    def retriver_init(self, k=30):
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
        def __init__(self, llm, max_token_limit=4000):
            # Use token-aware memory
            self.memory = ConversationTokenBufferMemory(
                llm=llm,
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=max_token_limit
            )
            self.chat_template = self.get_chat_template()
            self.llm = llm

        def invoke(self, inputs: dict) -> dict:
            # Extract user inputs
            context = inputs.get("context", "")
            question = inputs.get("question", "")

            # Load chat history from memory
            chat_history = self.memory.load_memory_variables({})["chat_history"]

            # Build the prompt messages
            formatted_prompt = self.chat_template.format_prompt(
                context=context,
                chat_history=chat_history,
                question=question
            ).to_messages()

            # Get model response
            response = self.llm.invoke(formatted_prompt)

            # Save interaction into memory (token-aware truncation happens automatically)
            self.memory.save_context(
                {"input": question},
                {"output": response.content}
            )

            # Return model response
            return response

        @staticmethod
        def get_chat_template():
            from langchain.prompts.chat import (
                ChatPromptTemplate,
                SystemMessagePromptTemplate,
                HumanMessagePromptTemplate
            )
            system_template = """
You are an AI assistant specialized in retrieval-augmented generation (RAG).

Your task:
1. Use ONLY the content inside <context> as your factual source. If some information is missing, say so clearly rather than guessing.
2. Provide detailed, well-reasoned, and structured answers. Prefer depth over brevity — include context, explanations, examples, and relationships between ideas.
3. Prefer short paragraphs and readable formatting over dense text. Use Markdown formatting for clarity.
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
    #{{ ready
    def ready(self):
        self.data_lib_init()
        self.retriver_init()
        self.chain=self.ConversationalRunnable(llm=self.llm)
    #}}
    #{{ Chat function
    def ask(self, query:str, retrival_option:str="localFirst") -> str:
        if query.strip() == "new":
            self.clear_chat_history()
            return "OK, let's start a new chat."
        elif query.strip() == "ping":
            try:
                _ = self.llm.invoke("ping")  # dummy prompt
                return "Yes, yes, I'm still alive."
            except Exception as e:
                return f"Ping failed: {e}"
            
        retdoc, retdoc_meta = self.custom_retreiver(query, memory=self.chain.memory, retrival_option=retrival_option)
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
    #{{ QA rewrite user's question for retrival
    def QA_rewrite_question_for_retrival(self, quest:str, memory):
        system_template = """
You are a query rewriter and retrieval decision maker for a Retrieval-Augmented Generation (RAG) system.

Your primary task is to rewrite the user’s latest question (<UserQuestion>) into one or more clear, self-contained, and nonredundant search queries optimized for semantic similarity search using embeddings.

Your secondary task is to determine whether external retrieval is needed to answer the question.

Guidelines:
- Use the chat history (<ChatHistory>) to resolve pronouns, references, and context relevant to the latest question.
- Preserve the user’s intent exactly — do not invent details.
- When useful, break the question into multiple sub-queries capturing distinct aspects of the information need.
- Optionally include broader “step-back” queries to improve retrieval coverage.
- Make rewritten queries explicit, unambiguous, and sufficiently detailed.
- If the user’s question can be fully answered from existing context (e.g., it’s a greeting, command, clarification, or instruction about the chat itself), then retrieval is **not needed**.
- Never explain your reasoning or include any extra words.
- Do not add punctuation, quotes, or introductory phrases.
        
Output Format:
If retrieval is needed:
RETRIEVE
<query 1>
<query 2>
...

If retrieval is NOT needed:
NO RETRIEVAL NEEDED
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
        out = self.llm.invoke(ChatPromptTemplate.from_messages([system_prompt, human_prompt]).format_prompt(question=quest, chat_history=chat_history).to_messages()).content.strip()
        if out == "NO RETRIEVAL NEEDED":
            return None
        else:
            return out.removeprefix("RETRIEVE").strip().splitlines()
                
    #}}
    #{{ QA if need PubMed search
    def QA_if_need_pubmed_search(self, quest:str, memory, retrieved_docs:list[dict]):
        system_template = """
You are an expert biomedical research assistant operating within a Retrieval-Augmented Generation (RAG) system. 
Your primary role is to determine whether additional retrieval from PubMed is necessary to accurately and comprehensively answer the user's question. 
If retrieval *is* required, you must produce two key outputs:

1. A concise, well-structured search query string suitable for the PubMed E-Utilities API:
   https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
2. A short, descriptive text summarizing the information need. This text will later be used by a cross-encoder to assess the relevance of retrieved content.

You will receive the following inputs:
- <ChatHistory>: The complete conversation history so far.
- <UserQuestion>: The user’s most recent question.
- <RetrievedResults>: Any content retrieved from a local library (not from the internet). This may be empty.

---

### Guidelines for PubMed Query Construction
- Always construct clear, specific, and concise queries using appropriate biomedical terminology.
- Focus queries on key entities, conditions, mechanisms, interventions, or relationships explicitly or implicitly mentioned in the user’s question.
- Avoid unnecessary words, stopwords, or overly broad terms.
- If the user's question is ambiguous, err on the side of broader but still relevant terms, and note the ambiguity in the "notes" field.

---

Your final output **must be a valid JSON object** following the schema described in the human prompt. Do not include any other text.
"""
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = """
You are given the following inputs:

<ChatHistory>
{chat_history}
</ChatHistory>

<UserQuestion>
{question}
</UserQuestion>

<RetrievedResults>
{retrieved_results}
</RetrievedResults>

---

### Your Task

1. **Assess sufficiency of retrieved context:**  
   Decide whether the information in <RetrievedResults> is adequate to answer <UserQuestion> accurately, comprehensively, and with scientific rigor.

2. **Determine retrieval need:**  
   - If the context is sufficient, set `"needs_pubmed_search": false`.
   - If it is insufficient, incomplete, outdated, or irrelevant, set `"needs_pubmed_search": true`.

3. **If retrieval is needed:**  
   - Generate a high-quality PubMed query string (`pubmed_query_term`) following the construction guidelines.
   - Provide a short, clear `description` summarizing the intended information need (for cross-encoder filtering).
   - Optionally, include a `"notes"` field if there is ambiguity, missing context, or other special considerations.

---

### Output Format

Your response **must be a valid JSON object** and follow exactly one of the following structures:
{{
  "needs_pubmed_search": false
}}

Or:
{{
  "needs_pubmed_search": true,
  "pubmed_query_term": "concise query for PubMed",
  "description": "short description of the information need"
}}

Or:
{{
  "needs_pubmed_search": true,
  "pubmed_query_term": "concise query for PubMed",
  "description": "short description of the information need",
  "notes": "optional explanation of ambiguity, missing context, or other special considerations"
}}
"""
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_history = memory.load_memory_variables({})["chat_history"]
        retrieved_results = "\n".join(x['context'] for x in retrieved_docs)
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt]).format_prompt(question=quest, chat_history=chat_history, retrieved_results=retrieved_results)
        outstr = self.llm.invoke(prompt).content.strip()
        match = re.search(r'\{.*\}', outstr, re.DOTALL)
        if match:
            out = json.loads(match.group(0))
        else:
            warnings.warn(f"Error in parse below JSON:\n{outstr}\nError: {e}")
            return None
    
        if 'notes' in out:
            print(out['notes'])
        if not out['needs_pubmed_search']:
            return None
        else:
            return (out['pubmed_query_term'], out['description'])
                
    #}}
    #{{ PubMed full text search
    def pubmed_fulltext_search(self, term:str, num:int=10)->list[str]:
        # Search PMC IDs
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pmc",                   # Search PubMed Central
            "term": term,                  # Your query
            "retmax": num,                   # Max number of results
            "retmode": "xml",               # Response format
            "sort": "relevance"
        }
        response = requests.get(base_url, params=params)
        root = ET.fromstring(response.content)
        return [id_elem.text for id_elem in root.findall(".//Id")]

    def pubmed_fulltext_fetch_and_chop(self, pmc_id:str)->(list[dict], dict):
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pmc",
            "id": pmc_id,
            "retmode": "xml"
        }
        response = requests.get(efetch_url, params=params)
        xml_data = response.content
        root = ET.fromstring(xml_data)
        # Extract title
        title = root.findtext(".//article-title")
        if not title:
            print(xml_data)
            raise Exception("Cannot get ArticleTitle")
        # Extract abstract text
        abstract_texts = [elem.text for elem in root.findall(".//abstract//p")]
        abstract = "\n".join(filter(None, abstract_texts))
        # Extract main text body
        body_paragraphs = [elem.text for elem in root.findall(".//body//p")]
        body_text = "\n".join(filter(None, body_paragraphs))
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size_by_char, chunk_overlap=self.chunk_overlap_by_char)
        # text_splitter.split_text(str) ->list(str)

        page = 'title'
        out=[{'context': f'<doc filename="{title}" page="{page}"  >\n{abstract}\n</doc>',
                 'content': title, 'filename': title, 'page': page}]
        for chunk in text_splitter.split_text(abstract):
            page = 'abstract'
            t = {'context': f'<doc filename="{title}" page="{page}"  >\n{abstract}\n</doc>',
                 'content': chunk, 'filename': title, 'page': page}
            out.append(t)
        for chunk in text_splitter.split_text(body_text):
            page = 'fulltext'
            t = {'context': f'<doc filename="{title}" page="{page}"  >\n{chunk}\n</doc>',
                 'content': chunk, 'filename': title, 'page': page}
            out.append(t)

        return (out, {'title': title, 'abstract': abstract, 'full_body_text': body_text})
    #}}
    #{{ PubMed abstract text search
    def pubmed_abstract_search(self, term:str, num:int=10)->list[str]:
        # Search PMC IDs
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",                   # Search PubMed Central
            "term": term,                  # Your query
            "retmax": num,                   # Max number of results
            "retmode": "xml",               # Response format
            "sort": "relevance"
        }
        response = requests.get(base_url, params=params)
        root = ET.fromstring(response.content)
        return [id_elem.text for id_elem in root.findall(".//Id")]

    def pubmed_abstract_fetch_and_chop(self, pm_id:str)->(list[dict], dict):
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pm_id,
            "retmode": "xml"
        }
        response = requests.get(efetch_url, params=params)
        xml_data = response.content
        root = ET.fromstring(xml_data)
        title = root.findtext(".//ArticleTitle")
        if not title:
            print(xml_data)
            raise Exception("Cannot get ArticleTitle")
        
        abstract = " ".join([a.text for a in root.findall(".//AbstractText") if a.text])
        pmcid = root.findtext(".//ArticleIdList/ArticleId[@IdType='pmc']")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size_by_char, chunk_overlap=self.chunk_overlap_by_char)
        # text_splitter.split_text(str) ->list(str)

        #If either title or any chunk of abstract is hitted, the whole abstract will be put in context.
        page = 'title'
        out=[{'context': f'<doc filename="{title}" page="{page}"  >\n{abstract}\n</doc>',
                 'content': title, 'filename': title, 'page': page}]
        for chunk in text_splitter.split_text(abstract):
            page = 'abstract'
            t = {'context': f'<doc filename="{title}" page="{page}"  >\n{abstract}\n</doc>',
                 'content': chunk, 'filename': title, 'page': page}
            out.append(t)

        return (out, {'title': title, 'abstract': abstract, 'pmcid': pmcid})

    def pubmed_fetch_and_chop(self, pm_id:str)->list[dict]:
        docs, meta = self.pubmed_abstract_fetch_and_chop(pm_id)
        if meta['pmcid']:
            # print(f"Full text avaiable: {meta['pmcid']}")
            docs, _ = self.pubmed_fulltext_fetch_and_chop(meta['pmcid'])
        return docs
    #}}
    #{{ Reload custom_retreiver
    def parse_doc(self, doc):
        filename = os.path.basename(doc.metadata["source"])
        page = doc.metadata["page_label"]
        return {'context': f'<doc filename="{filename}" page="{page}"  >\n{doc.page_content}\n</doc>',
                'content': doc.page_content,
                'filename': filename, 'page': page}
    
    def custom_retreiver(self, query: str, memory, retrival_option:str="localFirst") -> str:
        assert retrival_option in ['localFirst', 'localOnly', 'PubMedOnly']
        def fill_docs_to_context(docs:list[dict], scores:list[float])->list[dict]:
            if not docs:
                return []
            doc_score_pool = {}
            for D, S in zip(docs, scores):
                if not (D['context'] in doc_score_pool and doc_score_pool[D['context']][1]>=S):
                    doc_score_pool[D['context']]=(D, S)

            docs_score_sorted = sorted(list(doc_score_pool.values()), key=lambda x: x[1], reverse=True)
            print(f"{len(docs_score_sorted)} docs retrievaled.")

            if docs_score_sorted[0][1] >= self.crossencoder_normscore_cutoff:
                cutoff = self.crossencoder_normscore_cutoff
            else: #In case no strong correlated doc is retrived
                cutoff = self.crossencoder_normscore_cutoff_loose

            out_docs = []
            P = 0
            pS = 0.0
            for D, S in docs_score_sorted:
                tokennum = self.llm.get_num_tokens(D['context'])
                if S < cutoff or P + tokennum > self.doc_num_ctx:
                    break
                out_docs.append(D)
                P += tokennum
                pS = S

            print(f"{len(out_docs)} docs included. Lowest cross encoder score: {pS}")
            return out_docs
        
        # if not(self.QA_need_retrival(query, memory=memory)):
        #     return ("No context needed.", [])
        if retrival_option == 'PubMedOnly':
            docs = []
            scores = []
            out_docs = []
        else:
            squerys = self.QA_rewrite_question_for_retrival(query, memory=memory)
            if not squerys:
                return ("No context needed.", [])
        
            print("Local retrieval...")
            docs = []
            squerys_dup = []
            for squery in squerys:
                if not(squery.strip()):
                    continue
                t = self.retriever.invoke(squery)
                docs.extend(t)
                squerys_dup.extend([squery]*len(t))
            assert len(docs) == len(squerys_dup)

            docs = list(map(self.parse_doc, docs))
            scores = sigmoid(self.crossencoder.predict([(x, y['content']) for x, y in zip(squerys_dup, docs)]))
            out_docs = fill_docs_to_context(docs, scores)

        if retrival_option == 'localOnly':
            pass
        else:
            pubmed_query = self.QA_if_need_pubmed_search(query, memory, out_docs)
            if pubmed_query:
                pquery, description = pubmed_query
                print(f"PubMed query: {pquery}")
                print(f"Description: {description}")
                pm_ids = self.pubmed_abstract_search(pquery)
                pubmed_docs = []
                for pm_id in pm_ids:
                    pubmed_docs.extend(self.pubmed_fetch_and_chop(pm_id))

                #Strategy I: only compare to description
                pubmed_scores = sigmoid(self.crossencoder.predict([(description, y['content']) for y in pubmed_docs]))

                #Increase PubMed quary score
                s_pubmed_docs = []
                s_pubmed_scores = []
                t = sorted(zip(pubmed_docs, pubmed_scores), key=lambda x: x[1], reverse=True)
                for (i, (D, S)) in enumerate(t):
                    # I included at least 3 PubMed result in the context.
                    if i == 0: print(f"Highest PubMed relativity: {t[0][1]}")
                    if S > self.crossencoder_normscore_cutoff and i <= 3:
                        S += 1.0
                    s_pubmed_docs.append(D)
                    s_pubmed_scores.append(S)
                out_docs = fill_docs_to_context(docs + s_pubmed_docs, scores + s_pubmed_scores)
            
            # #Strategy II: compare to both description and previous RAG queries.
            # pquery_doc_pair = []
            # pubmed_docs_dup = []
            # for A in [description]+squerys:
            #     for B in pubmed_docs:
            #         pubmed_docs_dup.append(B)
            #         pquery_doc_pair.append((A, B['content']))
            # pubmed_scores = sigmoid(self.crossencoder.predict(pquery_doc_pair))
            # out_docs = fill_docs_to_context(docs + pubmed_docs_dup, scores + pubmed_scores)
        
        return ("\n".join([x['context'] for x in out_docs]), out_docs)
    #}}
    #{{ Unused code
        #{{ QA if need retrival (not used)
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
        #{{ QA if tetrivalled documents is relevant (not used)
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
        #{{ QA if a list of documents are relevant (not used)
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
        #{{ QA use cross-encoder (not used)
    class CrossEncoder:
        def __init__(self):
            self.model = CrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                                      device="cpu")
        
        def predict_score(self, query, docs):
            pairs = [(query, doc['context']) for doc in docs]
            scores = self.model.predict(pairs)
            # reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            # return [doc for doc, _ in reranked[:self.top_n]]
            return scores
    #}}
        #{{ QA 2nd retrival: rewrite user's question for retrival
    def QA_2nd_rewrite_question_for_retrival(self, quest:str, memory, previous_queries:str|list[str], retrieved_results:str|list[str]):
        system_template = """
You are a decision-making module that determines whether additional information retrieval is needed.

Use the following inputs:
- <ChatHistory>: The full conversation so far.
- <UserQuestion>: The user's most recent question.
- <PreviousQueries>: Search queries already made (if any).
- <RetrievedResults>: The content retrieved from previous queries.

Your task:
1. Decide whether to perform more retrieval.
2. If retrieval is needed, propose one or more new queries. The new queries should be explicit, unambiguous, and sufficiently detailed.
3. If not needed or hopeless, return an empty query list [].

Decision rules:
- "RETRIEVE": More or refined search is useful. Provide 1–3 new queries.
- "NO_ADDITIONAL_RETRIEVAL_NEEDED": Current information is sufficient.
- "RETRIEVAL_HOPELESS": No realistic query can improve the answer.

Output strictly in JSON format as a list of objects. No any other contents.
Output format example:
{{
  "decision": "RETRIEVE" | "NO_ADDITIONAL_RETRIEVAL_NEEDED" | "RETRIEVAL_HOPELESS",
  "queries": ["list of new search queries, if any"],
  "explanation": "brief reasoning for your decision (1–3 sentences)"
}}

Note:
- Never include any text outside the JSON object.
- The "explanation" is for internal reasoning only.
"""
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = """
<ChatHistory>
{chat_history}
</ChatHistory>

<UserQuestion>
{question}
</UserQuestion>

<PreviousQueries>
{previous_queries}
</PreviousQueries>
        
<RetrievedResults>
{retrieved_results}
</RetrievedResults>
    """
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_history = memory.load_memory_variables({})["chat_history"]
        
        if type(previous_queries) is list:
            previous_queries = "\n".join(previous_queries)
            
        if type(retrieved_results) is list:
            retrieved_results = "\n".join(retrieved_results)

        prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt]).format_prompt(question=quest, chat_history=chat_history, previous_queries=previous_queries, retrieved_results=retrieved_results)
        # print(prompt)
        tokennum = self.llm.get_num_tokens_from_messages(prompt.to_messages())
        print(f"Token number: {tokennum}")
        outstr = self.llm.invoke(prompt).content.strip()
        try:
            out = json.loads(outstr)
        except Exception as e:
            warnings.warn(f"Error in parse below JSON:\n{outstr}\nError: {e}")
            return None

        print(out['explanation'])
        if out['decision'] == "NO_ADDITIONAL_RETRIEVAL_NEEDED":
            return None
        elif out['decision'] == "RETRIEVAL_HOPELESS":
            print("Retrival hopeless. Give up.")
            return None
        else:
            return out['queries']
                
    #}}
    #}}
if __name__ == "__main__":
    rag=RAG()
    rag.ready()
    rag.chat()
