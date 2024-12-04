from dotenv import load_dotenv
import langchain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

from ai.prompt import contextualize_q_prompt, qa_prompt
from ai.db import db_obj
from ai.llm import llm

langchain.verbose = True
load_dotenv()
store = db_obj.get_store()

def query(chat_history, user_input, dummy=False):
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    try:
        if dummy:
            print("returning dummy result")
            return True, [
                HumanMessage(content=user_input),
                AIMessage(content="I am dummy answer")
            ], "I am dummy answer"
        else:
            print("invoking chain")
            result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
            print("got result")
            return True, [
                HumanMessage(content=user_input),
                AIMessage(content=result["answer"])
            ], result["answer"]
    except Exception as e:
        print("Got error from AI: retry sending the message " + str(e))
        return False, []
