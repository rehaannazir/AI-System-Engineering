from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory

load_dotenv()

brain = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

session_id = "user_1"
TOKEN_LIMIT = 500
FRESH_TOKENS = 250

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are AI assistant. Assist the human with appropriate answers."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}"),
    ]
)

summary_prompt = ChatPromptTemplate.from_template("""
Summarize the following conversation.
Keep:
- Important facts
- User preferences
- Goals
- Decisions
- Names

Conversation:
{conversation}
""")

summary_chain = summary_prompt | brain
chain = prompt | brain


store = {}


def get_session_history(session_id):

    if session_id not in store:
        store[session_id] = SQLChatMessageHistory(
            session_id=session_id, connection="sqlite:///chat.db"
        )

    return store[session_id]


def count_tokens(messages: list):
    return brain.get_num_tokens_from_messages(messages)


def history_check():

    history = get_session_history(session_id)
    history_tokens = count_tokens(history.messages)

    if history_tokens >= TOKEN_LIMIT:

        conversation = history.messages
        cur_tokens = 0
        recent_chat = []
        split_index = len(conversation)

        for i in reversed(conversation):

            cur_tokens += count_tokens([i])

            if cur_tokens <= FRESH_TOKENS:
                recent_chat.append(i)
                split_index -= 1

            else:
                break

        summary_chat = conversation[:split_index]
        history.clear()

        conversation_text = "\n".join(f"{m.type}: {m.content}" for m in summary_chat)

        summary = summary_chain.invoke({"conversation": conversation_text})

        history.add_message(SystemMessage(content=str(summary.content)))

        for msg in reversed(recent_chat):
            history.add_message(msg)


chatbot = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="query",
    history_messages_key="history",
)


while True:
    print("******************* <>CHAT STARTED<> ***********************")
    history_check()
    query = input("\n\n>>> ")

    if query.lower().strip() not in ["quit", "exit"]:

        response = chatbot.invoke(
            {"query": query},
            config={"configurable": {"session_id": session_id}},
        )
        print(f">>> {response.content}")
    else:
        print("******************* <>CHAT ENDED<> ***********************")
        break
