import os
import subprocess
from pathlib import Path
from typing import List, Dict, Union

from langchain_community.document_loaders import TextLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain.schema import Document

def load_document(path: Union[str, Path]) -> List[Document]:
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

    # File type mapped to appropriate Loader
    loader_map = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".pdf": PyMuPDF4LLMLoader, # TODO: PyMuPDF4LLM Pro (paid)
    }

    # Convert unsupported types to pdf; then reprocess
    if ext not in loader_map:
        # TODO Check if pdf file exists (already converted reviously)
        try:
            new_pdf = convert_to_pdf(path)
            return load_document(new_pdf)
        except RuntimeError as e :
            print(e)
            return []

    # Select and invoke loader
    loader = loader_map[ext](str(path))
    docs: List[Document] = loader.load()

    for doc in docs:
        doc.page_content = doc.page_content.replace("\n\n", "\n")

    return docs

def convert_to_pdf(input_path: Path) -> str:
    """
    Uses pandoc cli subprocess to convert files to pdf.
    New pdf files a stored in ./_temp folder.
    """
    output_path = Path(f"{os.getcwd()}/loaders/_temp/{input_path.stem}.pdf")
    
    try:
        subprocess.run(
            ["pandoc", input_path, "-o", output_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Pandoc failed: {e.stderr.decode()}") from e

if __name__ == "__main__":
    results = load_document(Path("C:/Users/USER/Documents/Work/CV.docx"))

    formatted_results = [
        {"text": doc.page_content.replace("\n\n", "\n"), "metadata": doc.metadata}
        for doc in results
    ]
    
    for text in formatted_results:
        print(text['text'])
        print(text['metadata'])