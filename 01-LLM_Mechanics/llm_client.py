import os
import tenacity
import logging
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from pydantic import BaseModel, Field
from tenacity import retry

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class API(BaseModel):

    description: str = Field(description="The description of API")


class retryable_api_call(Exception):
    pass


class non_retryable_api_call(Exception):
    pass


def check_status(response):

    if response in [500, 502, 503, 504, 429]:

        raise retryable_api_call(
            f"RetryableAPICall Error Occur. Status code {response}"
        )

    if 400 < response < 500:

        raise non_retryable_api_call(
            f"NonRetryableAPICall Error Occur. Status code {response}"
        )


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prompt = "What is API? Expplain in 1000 words"
instruction = "You are API professional. Answer in formal but polite tone. Return JSON"

input_tokens = client.models.count_tokens(
    model="gemini-2.5-flash", contents=[prompt, instruction]
)

print(f"Total Input Tokens: {input_tokens.total_tokens}")
input_cost = (input_tokens.total_tokens / 1000000) * 0.30
print(f"Input Cost : $ {input_cost:.10f}")


@retry(
    retry=tenacity.retry_if_exception_type(
        (ConnectionError, TimeoutError, retryable_api_call)
    ),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=4),
    stop=tenacity.stop_after_attempt(2),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_client():

    try:
        stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=instruction,
                response_mime_type="application/json",
                response_schema=API,
            ),
        )
        return list(stream)

    except errors.APIError as e:
        check_status(e.code)
        raise


try:
    chunks = call_client()

    full_text = ""
    for chunk in chunks:
        if chunk.text:
            full_text += chunk.text
            for char in chunk.text:
                print(char, end="", flush=True)
                time.sleep(0.02)

    parsed = API.model_validate_json(full_text)
    usage = chunks[-1].usage_metadata

    print("\n *** Tokens Consumption Summary ***")
    print(f"Total Output Tokens: {usage.candidates_token_count}")
    print(f"Total Tokens: {usage.total_token_count}")

    print(f"\n Output Cost : ${(usage.candidates_token_count / 1000000) * 2.50}")

except (retryable_api_call, non_retryable_api_call) as e:
    logger.error("API call failed: %s", e)

except Exception:
    logger.exception("Unexpected error during call_client")
