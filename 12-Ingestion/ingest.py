import re
import hashlib
import unicodedata
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)


def fetch_docs(folder, formats):

    docs = [file for file in folder.iterdir() if file.suffix in formats.keys()]

    return docs


# Loading text of docs
def loading_text(docs, formats):
    texts = []

    for doc in docs:
        loader_class = formats[doc.suffix]
        loader = loader_class(str(doc))
        text = loader.load()
        texts.extend(text)

    return texts


def normalize(docs):

    new_docs = []

    for doc in docs:

        text = doc.page_content.strip()

        # same unicode
        text = unicodedata.normalize("NFKC", text)

        # remove spacings
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # remove artifacts
        text = re.sub(r"Page\s+\d+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"-{3,}", "", text)  # ------
        text = re.sub(r"_{3,}", "", text)  # ______

        doc.page_content = text

        new_docs.append(doc)

    return new_docs


def markdown_chunking(doc):
    headers_to_split = [("#", "H1"), ("##", "H2"), ("###", "H3")]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)

    return splitter.split_text(doc)


def html_chunking(doc):
    headers_to_split = [("h1", "H1"), ("h2", "H2"), ("h3", "H3")]

    splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split)

    return splitter.split_text(doc)


def text_chunking(doc):

    seperators_order = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(separators=seperators_order)

    return splitter.split_documents([doc])


def get_hash(doc):

    return hashlib.sha256(doc.page_content.encode()).hexdigest()


def hashing(docs: list, store: set):

    after_docs = []

    for doc in docs:

        if get_hash(doc) not in store:

            check_sum = get_hash(doc)
            store.add(check_sum)
            after_docs.append(doc)

        else:
            continue

    return after_docs


def enrich_metadata(docs):

    for doc in docs:

        source = doc.metadata.get("source", "")

        doc.metadata["source"] = source
        doc.metadata["doc_type"] = Path(source).suffix
        doc.metadata["page"] = doc.metadata.get("page")
        doc.metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()
        doc.metadata["checksum"] = get_hash(doc)

    return docs


def merge_metadata(chunks, source_metadata):

    for chunk in chunks:

        for key, value in source_metadata.items():

            # setdefault so header metadata (H1/H2/H3) from splitters
            # isn't clobbered by the parent doc's metadata
            chunk.metadata.setdefault(key, value)

    return chunks


def chunking(docs):

    chunks = []

    for doc in docs:

        suffix = Path(doc.metadata["source"]).suffix

        if suffix == ".md":

            doc_chunks = markdown_chunking(doc.page_content)

        elif suffix == ".html":

            doc_chunks = html_chunking(doc.page_content)

        elif suffix in [".txt", ".pdf"]:

            doc_chunks = text_chunking(doc)

        else:

            continue

        chunks.extend(merge_metadata(doc_chunks, doc.metadata))

    return chunks


def main():

    hash_store = set()
    path = Path("docs")

    required_formats = {
        ".pdf": PyPDFLoader,
        ".md": partial(TextLoader, encoding="utf-8"),
        ".html": partial(TextLoader, encoding="utf-8"),
    }

    docs = fetch_docs(folder=path, formats=required_formats)

    docs_text = loading_text(docs, required_formats)

    docs_normalize = normalize(docs_text)

    docs = hashing(docs_normalize, store=hash_store)

    docs = enrich_metadata(docs)

    chunks = chunking(docs)

    print(docs_normalize)

    print(chunks)


if __name__ == "__main__":
    main()
