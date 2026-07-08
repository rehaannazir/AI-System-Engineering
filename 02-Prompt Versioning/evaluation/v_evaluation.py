import os
import time
from dotenv import load_dotenv
from google import genai
from registry.prompt_reg import Prompt_Register
from validators.validator import validate_output

load_dotenv()
p = Prompt_Register()

topic = "AI is dangerous"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

versions = ["v1.0", "v1.1", "v2.0"]

for v in versions:

    print(f"**** {v} ****")

    for i in range(5):

        contents = p.render("summarizer", v, {"article": topic})

        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=contents
        )

        try:
            validate_output(response.text, ["summary", "keywords"])
            print("Valid Json")

        except ValueError as e:

            print("invalid json")

        time.sleep(20)
