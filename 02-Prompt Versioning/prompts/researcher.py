research = {
    "v1.0": {
        "template": """
Providing you the book {book} whose author is {author}.

Give me:
- The main idea of the author in this book.
- The market price of the book.

Return JSON with:
- idea: string
- price: float

No additional explanation.
""",
        "variables": [
            "book",
            "author",
        ],
        "required_fields": [
            "idea",
            "price",
        ],
    },
    "v2.0": {
        "template": """
You are a book research specialist.

Book Details:

Name:
{book}

Author:
{author}

Provide:
- The main idea of the author in this book.
- The current market price in USD.

Constraints:
- The idea should not exceed 150 words.
- Explain the actual purpose/message of the author.
- Price must be in USD.

Return JSON only:

{{
    "idea": "...",
    "price": 50.5
}}
""",
        "variables": [
            "book",
            "author",
        ],
        "required_fields": [
            "idea",
            "price",
        ],
    },
}
