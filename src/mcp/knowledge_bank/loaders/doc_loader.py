from pathlib import Path
from typing import List, Dict, Union
from langchain_community.document_loaders import TextLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain.schema import Document

def load_document(path: Union[str, Path]) -> List[Dict[str, Union[str, dict]]]:
    """
    Loads a document from the given path using the appropriate LangChain loader
    based on file extension. Returns a list of text chunks with metadata.

    Args:
        path (Union[str, Path]): Path to the document file.

    Returns:
        List[Dict[str, Union[str, dict]]]: A list of dicts, each containing:
            - 'text': Extracted text content
            - 'metadata': Metadata such as file path, page number, etc.
    """
    path = Path(path)
    ext = path.suffix.lower()

    loader_map = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".pdf": PyMuPDF4LLMLoader, # TODO: PyMuPDF4LLM Pro (paid)
    }

    if ext not in loader_map:
        raise ValueError(f"Unsupported file type: {ext}")

    loader = loader_map[ext](str(path))
    docs: List[Document] = loader.load()

    return [
        {"text": doc.page_content.replace("\n\n", "\n"), "metadata": doc.metadata}
        for doc in docs
    ]

if __name__ == "__main__":
    result = load_document(Path("C:/Users/USER/Documents/Work/CV.pdf"))
    print(len(result))
    
    for text in result:
        print(text['text'])
        print(text['metadata'])