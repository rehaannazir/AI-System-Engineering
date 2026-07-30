from pydantic import BaseModel, Field
from typing import Optional, Literal
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Loading env variables
load_dotenv()


# Required output schema
class Lead(BaseModel):

    name: str = Field(description="Full name of the person contacting the company")
    company: str = Field(description="Name of the company the person represents")
    intent: str = Field(
        description="What the person wants (e.g. any service, demo, support, meeting or anything else means purpose of message)"
    )
    budget_signal: Optional[str] = Field(
        default=None, description="Any mention of budget or spend"
    )
    urgency: Literal["Normal", "Urgent", "Very Urgent"] = Field(
        description="Urgency inferred from tone/wording"
    )


# Preparing the structured LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", response_mime_type="application/json"
)
structured_llm = llm.with_structured_output(Lead)

# Making the Prompt Runnable
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are receptionist and CRM manager of a company. Just extract  name, company, intent, budget(if given), urgency(from msg intensity and emotions). Return only JSON, nothing else.",
        ),
        (
            "human",
            "The message of the person:- message : {email}. Now fetch valuable insights from it required by the company to fill CRM.",
        ),
    ]
)

# Just giving user interface a classy touch
print(
    """╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███╗   ██╗███████╗██╗  ██╗ █████╗ ██████╗  █████╗                          ║
║   ████╗  ██║██╔════╝╚██╗██╔╝██╔══██╗██╔══██╗██╔══██╗                         ║
║   ██╔██╗ ██║█████╗   ╚███╔╝ ███████║██████╔╝███████║                         ║
║   ██║╚██╗██║██╔══╝   ██╔██╗ ██╔══██║██╔══██╗██╔══██║                         ║
║   ██║ ╚████║███████╗██╔╝ ██╗██║  ██║██║  ██║██║  ██║                         ║
║   ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝                         ║
║                                                                              ║
║      AI Automation • Agents • RAG • LangGraph                                ║ 
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝"""
)

email = input("\nHow can we help you?\n")

# Generating chain
chain = prompt | structured_llm
answer = chain.invoke({"email": email})

print(f"\n\n {answer}")
