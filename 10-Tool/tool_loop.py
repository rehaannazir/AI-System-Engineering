import os
import json
import httpx
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage

load_dotenv()

# Getting path of JSON File
CRM_DATA_PATH = os.path.join(os.path.dirname(__file__), "crm_data.json")


# Making tools for LLM manually
@tool
def check_weather(city: str) -> dict:
    """Give me weather of provided city"""

    url = "http://api.weatherapi.com/v1/current.json"

    response = httpx.get(
        url=url, params={"key": os.getenv("WEATHER_API_KEY"), "q": city}
    )

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        return {"error": f"Could not find weather for city={city!r}"}

    response_json = response.json()

    required_output = {
        "humidity": response_json["current"]["humidity"],
        "wind_speed_kph": response_json["current"]["wind_kph"],
        "temprature_celcius": response_json["current"]["temp_c"],
    }
    return required_output


@tool
def check_crm(name: str, company: str) -> dict:
    """Give the data of a customer in json file"""

    with open(CRM_DATA_PATH, "r") as f:
        crm_data = json.load(f)

    for customer in crm_data["customers"]:
        if (
            customer["name"].lower() == name.lower()
            and customer["company"].lower() == company.lower()
        ):
            return customer

    return {"error": f"No customer found for name={name!r}, company={company!r}"}


@tool
def calculator(x: int, y: int, op: str):
    """To perform calculation on two integers"""

    try:
        if op.lower() in ["multiply", "multiplication", "*"]:
            return {"Multiply_Ans": x * y}

        elif op.lower() in ["division", "divide", "/"]:
            return {"Division_Ans": x / y}

        elif op.lower() in ["add", "addition", "+", "sum"]:
            return {"Add_Ans": x + y}

        elif op.lower() in ["minus", "subtract", "-", "subtraction"]:
            return {"Minus_Ans": x - y}

        else:
            return {"error": "The operation is not valid to calculator"}

    except ZeroDivisionError:
        return {"error": "Cannot divide by zero. Enter a valid value for y"}


# LLM with retries and tools act as agent
llm_agent = (
    ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    .bind_tools([check_weather, check_crm, calculator])
    .with_retry(stop_after_attempt=3)
)

SYSTEM_PROMPT = """You are a helpful assistant that answers user requests by reasoning \
step by step and calling tools whenever they are needed to get accurate, up-to-date \
information. Never guess or make up data that a tool could give you.

You have access to the following tools:

1. check_weather(city: str)
   - Returns current humidity, wind speed (kph), and temperature (Celsius) for a city.
   - Use this whenever the user asks about current weather conditions in a specific place.
   - If the user doesn't name a city, ask them for one instead of guessing.

2. check_crm(name: str, company: str)
   - Looks up a customer record by full name AND company.
   - Use this when the user asks for details about a specific customer (status, deal \
value, contact info, etc.).
   - Both name and company are required. If either is missing, ask the user for it \
before calling the tool.
   - If the tool returns an "error" field, tell the user no matching customer was found \
instead of inventing one.

3. calculator(x: int, y: int, op: str)
   - Performs add, subtract, multiply, or divide on two integers.
   - Use this for any arithmetic instead of computing it yourself.
   - If the tool returns an "error" field (e.g. division by zero or invalid operation), \
relay that clearly to the user.

Guidelines:
- Only call a tool when it's actually needed to answer the request; answer directly if \
you already have enough information.
- Call one tool at a time and use its result to decide your next step before calling \
another.
- If a tool result contains an "error" key, don't repeat the exact same call again. \
If you can spot a likely cause (e.g. a multi-word city name was passed without spaces, \
a typo, wrong casing), correct that specific argument and retry once. If you already \
retried with a corrected argument and it still failed, stop and explain the problem in \
plain language instead of retrying again.
- Once you have everything you need, give a clear, concise final answer grounded only \
in the tool outputs and the conversation so far."""

# Generating prompt runnable
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{query}"),
    ]
)

# Making Tools dict to use in recursive calling
tools = {
    "check_weather": check_weather,
    "check_crm": check_crm,
    "calculator": calculator,
}


# Helper query runnable to give required dict to prompt
def get_query(query):
    return {"query": query}


query = RunnableLambda(get_query)


# Actual Agent haveing tools when get query returns AI Message
def run_agent(user_query: str):
    chain = query | prompt
    messages = chain.invoke(user_query).to_messages()  # [SYSTEM Message, HUMAN Message]

    answer = llm_agent.invoke(messages)

    while True:

        if answer.tool_calls:  # If AI Message have un-empty tool_calls list

            messages.append(answer)  # To take history of LLM Responses (AI Messages)

            for i in answer.tool_calls:  # We may have multiple tools at one call

                tool_name = i["name"]
                tool_args = i["args"]

                messages.append(  # To make history of our responses (Tool Messages)
                    ToolMessage(
                        content=json.dumps(
                            tools[tool_name].invoke(tool_args)
                        ),  # function excuted by python given by LLM
                        tool_call_id=i["id"],
                    )
                )

            answer = llm_agent.invoke(
                messages
            )  # LLM excutes messages(including history to get better approach) until get final result

            continue

        else:

            return answer.content  # It will be empty until LLM make final response


if __name__ == "__main__":
    q = "What's about the weather of Paris and also New York city"
    print(run_agent(q))
