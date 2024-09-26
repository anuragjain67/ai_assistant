from data import get_db
from dotenv import load_dotenv
import langchain

langchain.verbose = True
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def initialize_chain(data_name):
    print("--- getting db ---")
    db = get_db(data_name)
    print("--- got db ---")

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-1.5-pro",
    #     temperature=0,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    #     # other params...
    # )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    # Contextualize question prompt
    # This system prompt helps the AI understand that it should reformulate the question
    # based on the chat history to make it a standalone question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )

    # Create a prompt template for contextualizing questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a history-aware retriever
    # This uses the LLM to help reformulate the question based on chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    # Answer question prompt
    # This system prompt helps the AI understand that it should provide concise answers
    # based on the retrieved context and indicates what to do if the answer is unknown
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know."
        "\n\n"
        "{context}"
    )

    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a chain to combine documents for question answering
    # `create_stuff_documents_chain` feeds all retrieved context into the LLM
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create a retrieval chain that combines the history-aware retriever and the question answering chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def process_input(rag_chain, chat_history, user_input, dummy=False):
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

def continual_chat(rag_chain):
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Process the user's query through the retrieval chain
        try:
            result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        except Exception as e:
            print("Got error from AI: retry sending the message " + str(e))
            continue
        # Display the AI's response
        print(f"AI: {result['answer']}")
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"]))


if __name__ == '__main__':
    rag_chain = initialize_chain("personal_notion")
    continual_chat(rag_chain)
