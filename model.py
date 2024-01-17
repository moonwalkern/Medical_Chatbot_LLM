from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """Use the following piece of information to answer the user's question.
If you don't know the answer, just say that you dont know, dont try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrival for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,input_variables=['context','question'])
    return prompt


def reteriever_qa_llm(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        chain_type='stuff',
        reteriever = db.as_retriever(),
        return_source_documents = True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    llm = CTransformers.from_pretrained(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        max_length = 512,
        temperature = 0.5
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings.from_pretrained(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device':'cpu'})
    db = FAISS(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = reteriever_qa_llm(llm, qa_prompt, db)
    return qa
