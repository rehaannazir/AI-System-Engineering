from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# Loading env variables
load_dotenv()


# Required output schemas
class Decision(BaseModel):
    decision: str = Field(
        description="What was decided, stated as a single clear sentence"
    )
    rationale: Optional[str] = Field(
        default=None, description="Why this decision was made, if mentioned"
    )


class ActionItem(BaseModel):
    task: str = Field(description="The specific action to be taken")
    owner: Optional[str] = Field(
        default=None, description="Person responsible, if named in the meeting"
    )
    due_date: Optional[str] = Field(
        default=None,
        description="Deadline mentioned, e.g. 'next Friday' or '2026-08-01'",
    )
    priority: Literal["low", "medium", "high"] = Field(
        default="medium", description="Urgency inferred from tone/context"
    )


class Risk(BaseModel):
    risk: str = Field(description="The risk or blocker raised")
    severity: Literal["low", "medium", "high"] = Field(
        description="Impact level inferred from discussion"
    )
    mitigation: Optional[str] = Field(
        default=None, description="Any proposed mitigation or next step"
    )


# Main Schema for LLM
class Summary(BaseModel):
    title: str = Field(description="A short title summarizing the meeting's main topic")
    decisions: List[Decision] = Field(
        description="All decisions made during the meeting"
    )
    action_items: List[ActionItem] = Field(
        description="All action items assigned during the meeting"
    )
    risks: List[Risk] = Field(
        default_factory=list, description="Risks, blockers, or concerns raised"
    )


# Preparing the structured LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", response_mime_type="application/json"
)
structured_llm = llm.with_structured_output(Summary)

# Making the Prompt Runnable
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an executive assistant who writes precise meeting summaries."
            "Extract only what is explicitly stated or clearly implied — do not invent names, dates, or owners."
            "Return only JSON, nothing else.",
        ),
        (
            "human",
            "Meeting transcript:\n{transcript}\n\n"
            "Summarize this into decisions, action items (with owner/due date/priority if mentioned),"
            "and risks (with severity and mitigation if mentioned).",
        ),
    ]
)


# As our prompt need dictionary
def get_notes(transcript):

    return {"transcript": transcript}


get_transcript = RunnableLambda(get_notes)

transcript = """
Q3 Budget Review Call

Alex: okay so first up, we're going with the AWS migration instead of staying on-prem, final call, no more debate on this one.
Dana: fine by me. but heads up — if we don't get the migration budget approved by finance before end of month, we're going to miss the Q3 window entirely. that's a big risk honestly, could push everything back a whole quarter.
Alex: yeah let's flag that to finance ASAP, I'll message the CFO directly today.
Dana: also can someone own actually writing the migration runbook? like end to end, needs to be done before we touch prod
Sam: I can take that, give me till the 20th
Alex: perfect. also we decided NOT to hire the extra contractor this quarter, budget's too tight, we'll revisit in Q4
Dana: one more risk — our lead DBA is going on leave starting next week, so if migration issues come up while she's out, we don't really have backup coverage. medium concern, not urgent-urgent but worth planning around
Sam: I'll loop in her backup and get a handoff doc going, low effort, should be quick
"""

# Generating chain
chain = get_transcript | prompt | structured_llm
answer = chain.invoke(transcript)

print(answer)
