from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import gradio as gr

DB_FAISS_PATH = "Medical_Chatbot_LLM/vectorstores/db_faiss"

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
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
        chain_type='stuff',
        retriever = db.as_retriever(),
        return_source_documents = True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model = "Medical_Chatbot_LLM/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        config={'max_new_tokens': 1024,
                'temperature': 0.01,
                'context_length' : 2048
                }
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device':'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = reteriever_qa_llm(llm, qa_prompt, db)
    return qa

def final_result(query, history):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    answer = response["result"]
    sources = response["source_documents"]

    if sources:
        answer += f"\n\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"
    return answer

def main():
    question = "What is cancer"
    response = final_result(question,None)
    print(response)

# if __name__ == "__main__":
#     gr.ChatInterface(
#         final_result,
#         chatbot=gr.Chatbot(height=500),
#         textbox=gr.Textbox(placeholder="Enter your question here",container=False, scale=7),
#         title="Search for Medical Questions",
#         theme="soft",
#         examples=[
#             "What is Abortion",
#             "What is First trimester abortions",
#             "What is MEDICAL ABORTIONS."
#         ],
#         cache_examples=False,
#         retry_btn=None,
#         undo_btn="Delete Previous",
#         clear_btn="Clear"
#     ).launch()


medical_chat = gr.ChatInterface(
        final_result,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Enter your question here",container=False, scale=7),
        title="Search for Medical Questions",
        theme="soft",
        examples=[
            "What is Abortion",
            "What is First trimester abortions",
            "What is MEDICAL ABORTIONS."
        ],
        cache_examples=False,
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear"
    )
