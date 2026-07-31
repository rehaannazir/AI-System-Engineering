from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

load_dotenv()

brain = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are AI assistant. Assist the human with appropriate answers."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}"),
    ]
)


def get_session_history(session_id):

    return SQLChatMessageHistory(session_id=session_id, connection="sqlite:///chat.db")


chain = prompt | brain

chatbot = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="query",
    history_messages_key="history",
)

response = chatbot.invoke(
    {"query": "Who is the father of Pakistan?"},
    config={"configurable": {"session_id": "user_1"}},
)

print(response.content)
