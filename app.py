import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

import ollama

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"

### LLM

local_llm = "llama3"

### Documents directory

documents_directory = "pdfs/"

### Index

def get_retriever(filenames = None):
    docs = [PyPDFLoader(documents_directory + filename).load() for filename in filenames]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
    )
    retriever = vectorstore.as_retriever()
    return retriever

### Retrieval Grader

def check_relevance(retriever = None, question = None):
    
    docs = retriever.invoke(question)
      
    doc_txt = docs[0].page_content
    
    system_prompt = """
        You are a grader assessing relevance of a retrieved document to a user question. 
        If the document contains keywords related to the user question, grade it as relevant. 
        It does not need to be a stringent test. 
        The goal is to filter out erroneous retrievals.
        Give a binary score 'relevant' or 'irrelevant' score to indicate whether the document is relevant to the question.
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """

    user_prompt = f"""
        Question: {question} 
        Context: {doc_txt}  
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = ollama.chat(model='llama3', messages=messages)['message']['content']

    return docs, response

### Generate

def retrieve_answer(retriever = None, question = None):

    docs = retriever.invoke(question)
    
    system_prompt = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Please answer the question in traditional Chinese.
    """

    user_prompt = f"""
        Question: {question} 
        Context: {docs}  
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return ollama.chat(model='llama3', messages=messages, stream=True)

### Hallucination Grader

def check_hallucination(docs = None, generation = None):
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'hallucinating' or 'clear-minded' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    response = hallucination_grader.invoke({"documents": docs, "generation": generation})
    return response

### Answer Grader

def check_usefulness(question = None, generation = None):
    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'useful' or 'useless' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | JsonOutputParser()
    response = answer_grader.invoke({"question": question, "generation": generation})
    return response

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def home():
    return render_template('chat.html')

@socketio.on('request')
def handle_request(data):
    message = data.get('message')

    docs, relevance = check_relevance(retriever = retriever, question = message)
    
    if relevance == 'irrelevant':
        emit('response', {'message': "The question is considered irrelevant.", 'references': "N/A"})
        return

    generation = ""
    for chunk in retrieve_answer(retriever = retriever, question = message):
        generation += chunk['message']['content']
        emit('response', {'message': generation, 'references': "N/A"})

    clear_mindedness = check_hallucination(docs = docs, generation = generation)
    if clear_mindedness == 'hallucinating':
        emit('response', {'message': "The assistant is hallucinating.", 'references': "N/A"})
        return

    usefulness = check_usefulness(question = message, generation = generation)
    if usefulness == 'useless':
        emit('response', {'message': "The response is useless.", 'references': "N/A"})
        return

    references = ""
    for doc in docs:
        doc_page, doc_source, doc_txt = doc.metadata["page"], doc.metadata["source"], doc.page_content
        references += f'{doc_source} 第{doc_page}頁:<br>\n'
        for line in doc.page_content.split('\n'):
            references += f"{line}<br>\n"
        references += "<br>\n"

    emit('response', {'message': generation, 'references': references})
    return

print("loading pdfs...")
filenames = next(os.walk(documents_directory), (None, None, []))[2]
retriever = get_retriever(filenames = filenames)
print("pdfs loaded.")
    
if __name__ == '__main__':
    socketio.run(app)
    # app.run(host='0.0.0.0', port=5000)