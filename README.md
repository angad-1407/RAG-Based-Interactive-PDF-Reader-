# Interactive PDF Reader

A Streamlit + LangChain app for uploading a PDF, reading page text, and chatting with the document using retrieval-augmented generation.

## Features

- Upload any text-based PDF
- Browse extracted page text
- Build a FAISS vector index from PDF chunks
- Ask conversational questions about the document
- View cited source snippets for each answer
- Tune chunk size, overlap, retrieval count, and model temperature

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file. Gemini is the default provider:

```text
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CHAT_MODEL=gemini-2.5-flash
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
```

You can still switch the provider dropdown to OpenAI if you add:

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## Run

```powershell
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal.

## Notes

This works best with PDFs that contain selectable text. Scanned PDFs need OCR before they can be searched reliably.
