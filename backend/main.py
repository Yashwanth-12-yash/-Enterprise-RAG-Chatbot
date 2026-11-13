
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from io import BytesIO

from document_processor import split_text
from vector_store import create_vector_store, search_documents
from gemini_service import generate_answer


# ==========================================
# 1. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Enterprise RAG Chatbot",
    description="Company Document Question Answering System",
    version="1.0.0"
)


# ==========================================
# 2. CONFIGURE CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 3. HOME API
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Enterprise RAG Chatbot API is running",
        "status": "success"
    }


# ==========================================
# 4. HEALTH CHECK API
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "message": "Backend is working properly"
    }


# ==========================================
# 5. PDF UPLOAD API
# ==========================================

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    # Check whether a file was provided
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected"
        )

    # Validate PDF extension
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    # Read uploaded file
    contents = await file.read()

    # Check empty file
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty"
        )

    # Limit file size to 10 MB
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="PDF must be smaller than 10 MB"
        )

    try:

        # Read PDF from memory
        pdf = PdfReader(BytesIO(contents))

        # Check encrypted PDF
        if pdf.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Encrypted PDFs are not supported"
            )

        # ==================================
        # EXTRACT TEXT FROM PDF
        # ==================================

        extracted_text = ""

        for page_number, page in enumerate(
            pdf.pages,
            start=1
        ):

            page_text = page.extract_text() or ""

            extracted_text += (
                f"\n--- Page {page_number} ---\n"
                f"{page_text}"
            )

        # Validate extracted text
        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No readable text found in PDF"
            )

        # ==================================
        # SPLIT TEXT INTO CHUNKS
        # ==================================

        chunks = split_text(extracted_text)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text chunks could be created"
            )

        # ==================================
        # GENERATE EMBEDDINGS AND STORE
        # ==================================

        create_vector_store(chunks)

        # ==================================
        # SUCCESS RESPONSE
        # ==================================

        return {
            "filename": file.filename,
            "total_pages": len(pdf.pages),
            "total_chunks": len(chunks),
            "message": "PDF processed and stored successfully"
        }

    except HTTPException:
        raise

    except Exception as e:

        print("PDF processing error:", str(e))

        raise HTTPException(
            status_code=400,
            detail="Unable to process this PDF"
        )

    finally:
        await file.close()


# ==========================================
# 6. SEARCH REQUEST MODEL
# ==========================================

class SearchRequest(BaseModel):

    question: str


# ==========================================
# 7. SEARCH COMPANY DOCUMENTS API
# ==========================================

@app.post("/search")
def search_company_documents(request: SearchRequest):

    # Validate question
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:

        # Search relevant document chunks
        results = search_documents(
            question,
            top_k=3
        )

        return {
            "question": question,
            "total_results": len(results),
            "results": results
        }

    except FileNotFoundError:

        raise HTTPException(
            status_code=404,
            detail="No documents uploaded yet"
        )

    except Exception as e:

        print("Document search error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Document search failed"
        )

# ==========================================
# 8. GEMINI RAG CHAT API
# ==========================================

@app.post("/chat")
def chat_with_documents(request: SearchRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:

        # STEP 1: Retrieve relevant chunks from FAISS
        results = search_documents(
            question,
            top_k=10
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No relevant information found"
            )

        # STEP 2: Prepare document context
        context = "\n\n".join(
            [
                result["chunk"]
                for result in results
            ]
        )

        # STEP 3: Generate answer using Gemini
        answer = generate_answer(
            question,
            context
        )

        # STEP 4: Return answer and sources
        return {
            "question": question,
            "answer": answer,
            "sources": results
        }

    except HTTPException:
        raise

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Please upload a PDF document first"
        )

    except Exception as e:

        print("RAG Chat Error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer. Check the backend terminal."
        )

