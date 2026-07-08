summarize = {
    "v1.0": {
        "template": """
Using the given below text summarize it and find keywords.

Text:
{article}

Return JSON with:
- summary: string
- keywords: list of strings
""",
        "variables": ["article"],
        "required_fields": ["summary", "keywords"],
    },
    "v1.1": {
        "template": """
You are a text summarizing specialist.

From the below topic, give its summary and SEO keywords.

Topic:
{article}

Return JSON in the following format:

{{
    "summary": "...",
    "keywords": ["keyword1", "keyword2"]
}}
""",
        "variables": ["article"],
        "required_fields": ["summary", "keywords"],
    },
    "v2.0": {
        "template": """
You are a text summarizing expert with 20+ years of experience in SEO.

Analyze the following topic:

{article}

Constraints:
- The summary should not exceed 100 words.
- The tone should be formal but polite.
- The audience is tech students and learners.
- It must be json output only with given example format containing all 

Return JSON only:

{{
    "summary": "Containing summary from article",
    "keywords": ["SEO keyword 1", "SEO keyword 2"]
}}
""",
        "variables": ["article"],
        "required_fields": ["summary", "keywords"],
    },
}
