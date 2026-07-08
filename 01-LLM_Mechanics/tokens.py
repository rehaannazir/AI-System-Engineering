# Layer one

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompt = "Name of fastest animal of world in one word"

input_tokens = client.models.count_tokens(model="gemini-2.5-flash", contents=prompt)
print("input tokens:", input_tokens.total_tokens)

output = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

print(output.text)
print("prompt tokens:", output.usage_metadata.prompt_token_count)
print("output tokens:", output.usage_metadata.candidates_token_count)

# Layer two - multi-turn with a persona set via system_instruction
# (a fake prior "model" turn is not how personas are set - system_instruction is)

output = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Can you explain marketing in 20 words",
    config=types.GenerateContentConfig(
        system_instruction="You are a marketing expert with $1 billion plus net worth",
    ),
)
print(output.text)

# To find model limit

model = client.models.get(model="gemini-2.5-flash")
print("input token limit:", model.input_token_limit)
print("output token limit:", model.output_token_limit)


# Context Caching
# Explicit caching requires a minimum ~4096 tokens of cached content,
# so the doc below is padded well past that threshold.

large_document = """
These values are generally not returned dynamically by the SDK. Instead:

Current request token counts → use count_tokens() or usage_metadata.
Maximum context window and maximum output tokens → check the official Gemini model documentation for the specific model you're using, as these are published model specifications rather than runtime values.

For most applications, the combination of:

count_tokens() (before the request), and
response.usage_metadata (after the request)

provides the information you need to manage prompt size and token usage.
""" * 40

doc_tokens = client.models.count_tokens(model="gemini-2.5-pro", contents=large_document)
print("document tokens:", doc_tokens.total_tokens)

cache = client.caches.create(
    model="gemini-3.5-flash",
    config=types.CreateCachedContentConfig(
        contents=[large_document],
        ttl="300s",
    ),
)
print("cache name:", cache.name)

output = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Summarize the cached document in one sentence",
    config=types.GenerateContentConfig(cached_content=cache.name),
)
print(output.text)
