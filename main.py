import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

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
        chunk_size=250, chunk_overlap=0
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
    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'relevant' or 'irrelevant' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )

    retrieval_grader = prompt | llm | JsonOutputParser()
    docs = retriever.invoke(question)
    
    print("Referenced documents: ")
    for doc in docs:
        doc_page, doc_source, doc_txt = doc.metadata["page"], doc.metadata["source"], doc.page_content
        print(f'\tAt page {doc_page} of "{doc_source}" wrote: ')
        for line in doc.page_content.split('\n'):
            print(f"\t\t{line}")
    print()
    
    doc_txt = docs[0].page_content
    response = retrieval_grader.invoke({"question": question, "document": doc_txt})
    return docs, response

### Generate

def retrieve_answer(retriever = None, question = None):
    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )

    llm = ChatOllama(model=local_llm, temperature=0)


    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    docs = retriever.invoke(question)
    
    generation = rag_chain.invoke({"context": docs, "question": question})
    return generation


### Hallucination Grader

# LLM
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

filenames = next(os.walk(documents_directory), (None, None, []))[2]

def main():
    print("loading pdfs...")
    retriever = get_retriever(filenames = filenames)
    print("pdfs loaded.")

    while True:
        question = input("\nUser: ")
        print()
    
        if question.lower().strip() == "exit":
            print("Bye.")
            return
        
        docs, relevance = check_relevance(retriever = retriever, question = question)

        if relevance == 'irrelevant':
            print("The question is considered irrelevant.")
            continue

        generation = retrieve_answer(retriever = retriever, question = question)

        clear_mindedness = check_hallucination(docs = docs, generation = generation)

        if clear_mindedness == 'hallucinating':
            print("The assistant is hallucinating.")
            continue

        usefulness = check_usefulness(question = question, generation = generation)

        if usefulness == 'useless':
            print("The response is useless.")
            continue

        print(f"Assistant: {generation}")
        
if __name__ == "__main__":
    main()