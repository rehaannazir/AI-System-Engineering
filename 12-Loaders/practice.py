from pathlib import Path
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredHTMLLoader,
    TextLoader,
)

# Fetching entered files
path = Path("docs")
required_formats = [".pdf", ".txt", ".md", ".html"]
files = [file for file in path.iterdir() if file.suffix in required_formats]
