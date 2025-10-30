from fastapi import FastAPI, File, UploadFile, HTTPException
from src.pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from src.langchain_utils import get_rag_chain
from src.db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record, initialize_database
from src.chroma_utils import index_document_to_chroma, delete_doc_from_chroma
import os
import uuid
import logging
import shutil

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Initialize database indexes on startup (non-fatal on failure)
@app.on_event("startup")
def on_startup():
    try:
        initialize_database()
    except Exception as e:
        logging.error(f"Database initialization failed on startup: {e}")

# Chat endpoint
@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")

    # Try fetching chat history safely
    try:
        chat_history = get_chat_history(session_id)
    except Exception as e:
        logging.error(f"Failed to fetch chat history for {session_id}: {e}")
        chat_history = []

    # Force only supported models
    model_value = query_input.model.value
    if model_value != "gemini-2.0-flash":
        logging.warning(f"Unsupported model '{model_value}' requested. Falling back to 'gemini-2.0-flash'.")
        model_value = "gemini-2.0-flash"

    # Try model candidates (future-proof fallback)
    model_candidates = [
        model_value,
        "gemini-2.0-flash",
    ]

    last_error = None
    answer = None

    for candidate in model_candidates:
        try:
            rag_chain = get_rag_chain(candidate)
            response = rag_chain.invoke({
                "input": query_input.question,
                "chat_history": chat_history
            })

            # Log the full raw response for debugging
            logging.info(f"RAG chain raw response for session {session_id}: {response}")

            # Safely extract the answer from possible formats
            if isinstance(response, dict):
                answer = response.get("answer") or response.get("output_text") or str(response)
            else:
                answer = str(response)

            model_value = candidate
            last_error = None
            break

        except Exception as e:
            last_error = e
            logging.warning(f"Model '{candidate}' failed for session {session_id}: {e}")

    # If all models failed, raise HTTP 500
    if last_error is not None or answer is None:
        logging.error(f"All model candidates failed for session {session_id}: {last_error}")
        raise HTTPException(status_code=500, detail=f"RAG invocation failed: {last_error}")

    # Try logging chat to DB (non-fatal if fails)
    try:
        insert_application_logs(session_id, query_input.question, answer, model_value)
    except Exception as e:
        logging.error(f"Failed to insert application log for {session_id}: {e}")

    # Log final answer
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=model_value)

# Document upload endpoint
@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")

    # Save the file to the data/documents directory
    documents_dir = "data/documents"
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)

    temp_file_path = os.path.join(documents_dir, f"temp_{file.filename}")

    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(temp_file_path, file_id)

        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# List documents endpoint
@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()

# Delete document endpoint
@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}