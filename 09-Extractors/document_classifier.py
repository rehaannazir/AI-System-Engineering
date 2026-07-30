import fitz
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# Loading env variables
load_dotenv()


# Required output schema
class Classifier(BaseModel):
    document_type: Literal[
        "Invoice",
        "Resume",
        "Contract",
        "Purchase Order",
        "Bank Statement",
        "Research Paper",
        "Legal Notice",
        "Medical Report",
        "Complaint Email",
        "Other",
    ] = Field(description="The single best-fitting category for this document")

    confidence_score: int = Field(
        ge=0,
        le=10,
        description="Score from 0-10 for how confident you are in document_type",
    )

    reasoning: str = Field(
        description="One short sentence citing the specific text/clues that led to this classification"
    )


# Preparing the structured LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", response_mime_type="application/json"
)
structured_llm = llm.with_structured_output(Classifier)

# Making the Prompt Runnable
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a document classification system. Classify the given document "
            "into exactly one of the allowed categories. Base your decision only on "
            "content actually present in the document — if it does not clearly match "
            "any specific category, choose 'Other' rather than guessing. "
            "Return only JSON, nothing else.",
        ),
        (
            "human",
            "Document text:\n{document_text}\n\n"
            "Classify this document and provide your confidence score and reasoning.",
        ),
    ]

# As our prompt need dictionary (document -> text -> dictionary)
def fetch_text(document):
    with fitz.open(document) as doc_pages:
        doc_text = "".join(page.get_text() for page in doc_pages)

    return {"document_text": doc_text}

# Making function a runnable
text_extractor = RunnableLambda(fetch_text)

document = "---.pdf"
chain = text_extractor | prompt | structured_llm
answer = chain.invoke(document)
print(answer)
