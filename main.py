from fastapi import FastAPI,UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
import os, uuid
from sqlalchemy import create_engine, text
from google import genai
from google.genai import types
from pypdf import PdfReader
import sys
import sqlite3

print("Local SQLite:", sqlite3.sqlite_version)

try:
    if sys.platform.startswith("linux"):
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ModuleNotFoundError:
    pass
import chromadb
from pydantic import BaseModel, EmailStr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta, UTC
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from jose import JWTError, jwt
import logging
from functools import lru_cache

#Load dotenv file
load_dotenv()

app=FastAPI()

origin_url=os.getenv("ORIGIN_URL")
if not origin_url:
    raise ValueError("ORIGIN_URL not found in .env")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

#Gemini API Key
@lru_cache(maxsize=1)
def get_gemini_client():
    return (genai.Client(api_key=os.getenv("GEMINI_API_KEY"))) 
model=os.getenv("GEMINI_MODEL")

#Chroma client

@lru_cache(maxsize=1)
def get_chroma_client():
    CHROMA_DIR = "chroma_db"
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return (chromadb.PersistentClient(path=CHROMA_DIR))

#Embedding model
@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(os.getenv("EMBEDDING_TRANSFORMER_MODEL"))

#Postgre engine
@lru_cache(maxsize=1)
def get_db_engine():
    return create_engine(os.getenv("POSTGRESQL_CONNECTION_STRING"))

pwd_context= CryptContext(schemes=["argon2"],deprecated="auto")

class RegisterRequest(BaseModel):
    username:str
    email: EmailStr
    password: str

def hash_password(password:str):
    return pwd_context.hash(password)

@app.post("/register")
def register(request:RegisterRequest):
    query=text(""" select user_id from user_table where email=:email""")
    engine=get_db_engine()
    with engine.begin() as conn:
        existing= conn.execute(query,{"email":request.email}).fetchone()
        if existing:
            raise HTTPException(status_code=409,detail="Email already exists.")
        hashed_password= hash_password(request.password)
        conn.execute(
            text(""" insert into user_table(user_name,email,password_hash) values(:user_name,:email,:password_hash)"""),
            {"user_name":request.username, "email":request.email, "password_hash":hashed_password}
        )
    return {"message":"User registered successfully."}

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM= os.getenv("JWT_ALGORITHM")
if not SECRET_KEY or not ALGORITHM:
    raise HTTPException(status_code=409, detail=" JWT config missing.")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES",60))

class LoginRequest(BaseModel):
    email:EmailStr
    password: str

def verify_password(plain_password:str, hashed_password:str):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data:dict):
    to_encode = data.copy()
    expire= datetime.now(UTC)+timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp":expire})
    return jwt.encode(to_encode,SECRET_KEY,algorithm=ALGORITHM)
def verify_access_token(token: str):
    try:
        payload= jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def get_current_user(authorization:str =Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing data")
    token= authorization.replace("Bearer", "").strip()
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload["user_id"]

@app.post("/login")
def login_user(request:LoginRequest):
    query=text(""" select user_id, user_name, email, password_hash from user_table where email=:email""")
    engine=get_db_engine()
    with engine.begin() as conn:
        user= conn.execute(query,{"email":request.email}).mappings().first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(request.password,user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"user_id":user["user_id"],"email":user["email"]})
    return{
        "access_token":token,
        "token_type": "bearer",
        "user_id":user["user_id"],
        "username":user["user_name"]
    }

class generalChatRequest(BaseModel):
    question: str

@app.post("/general-chat-request")
def general_chat_request(request:generalChatRequest, user_id:int = Depends(get_current_user)):
    history_query= text(""" select question, answer from general_chat_history where user_id=:user_id order by chat_id desc limit 3""")
    engine= get_db_engine()
    client= get_gemini_client()
    with engine.begin() as conn:
        rows= conn.execute(history_query,{"user_id":user_id}).mappings().all()
    chat_context=""
    for row in reversed(rows):
        chat_context += f"User: {row['question']}\n"
    transform_prompt=f"""
        You are a query rewriting assistant.
        Given the conversation history and the latest question, rewrite the latest question so that it becomes a complete standalone question.
        Conversation History:{chat_context}
        Latest Question: {request.question}
        Rules:
        -Preserve the original meaning.
        -Replace pronouns such as it, they, this, that and these.
        -Return only the rewritten question.
        -Don't provide explanations, labels, or additional text.
        -If the question is already standalone, return it unchanged.
    """
    rewrite_response= (client.models.generate_content(
        model=model,
        contents=transform_prompt
    ) )
    transform_question = rewrite_response.text.strip()
    print("Original:", request.question)
    print("Transformed:", transform_question)
    prompt=f"""
                    You are an expert in answering questions.

                    Question:
                    {transform_question}
                    Answer using EXACTLY this format:

                    • Point 1

                    • Point 2

                    • Point 3

                    • Point 4

                    Rules:
                    - Start every point with the bullet character "• ".
                    - After each point, insert TWO newline characters.
                    - Each point must be a complete sentence.
                    - Do not put multiple bullets on the same line.
                    - Do not add introductions, headings, or conclusions.
                    """
    def generate():
        stream=client.models.generate_content_stream(
            model=model,
            contents=prompt
        ) 
        answer=""
        for chunk in stream:
            if chunk.text:
                answer+=chunk.text
                yield chunk.text
        query=text(""" insert into general_chat_history(user_id,question,answer) values (:user_id,:question,:answer)""")
        with engine.begin() as conn:
            conn.execute(query,{"user_id":user_id,"question":request.question,"answer":answer})
    
    return StreamingResponse(generate(),media_type="text/plain")

@app.get("/general-chat-history")
def general_chat_history(user_id:int= Depends(get_current_user)):
    query=text("select question,answer from general_chat_history where user_id=:user_id order by chat_id")
    engine= get_db_engine()
    with engine.begin() as conn:
        response= conn.execute(query,{"user_id":user_id}).mappings().all()
    return response  

@app.delete("/delete-general-chat")
def delete_general_chat(user_id:int = Depends(get_current_user)):
    check_query=text("""Delete from general_chat_history where user_id=:user_id""")
    engine=get_db_engine()
    with engine.begin() as conn:
        response=conn.execute(check_query,{"user_id":user_id})
        if response.rowcount==0:
            raise HTTPException(
                status_code=404,
                detail="No chat history to delete."
            )
    return{
        "message":"Chat deleted successfully.",
        "user_id":user_id
    }

# Creating file upload directory
UPLOAD_DIR= "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

#Upload document
@app.post("/upload")
async def upload_document(user_id: int = Depends(get_current_user), file: UploadFile = File(...)):
    #Check whether the uploaded file is pdf or not.
    if ( not (file.filename.lower().endswith(".pdf")) or file.content_type != "application/pdf"):
        raise HTTPException(
            status_code= 400,
            detail="Only PDF files are allowed."
        )
    #Check whether document in db or not. If exists raise error
    check_query = text(""" select document_id from uploaded_documents where user_id=:user_id and document_name= :document_name""")
    engine= get_db_engine()
    with engine.begin() as conn:
        existing= conn.execute(check_query,{"user_id":user_id,"document_name":file.filename}).fetchone()
    if existing:
        raise HTTPException(
            status_code= 409,
            detail=" File already exists"
        )
    
    # If not exists then create file_path and file_name to store
    file_name = f"{uuid.uuid4()}_{file.filename}"
    file_path=os.path.join(UPLOAD_DIR,file_name)

    # Reads the uploaded file into memoty and creates a new file on disk and writes those bytes into it.
    with open(file_path,"wb") as f:
        content= await file.read()
        f.write(content)

    # Read the content
    reader= PdfReader(file_path)

    # Extract the text from content
    extracted_text=""
    for page in reader.pages:
        page_text=page.extract_text()
        if page_text:
            extracted_text+=page_text
    if not extracted_text:
        extracted_text = "Document has no readable text."

    summary=""
    # Inserting document name, path and summary into db.
    insert_query=text(""" insert into uploaded_documents(user_id,document_name, file_path, summary, created_at) values(:user_id,:document_name,:file_path,:summary,Now()) returning document_id""")
    with engine.begin() as conn:
        result= conn.execute(insert_query,{"user_id":user_id, "document_name":file.filename,"file_path":file_path,"summary":summary})
        document_id=result.scalar()
    chroma_client= get_chroma_client()
    # Creating collection with chroma client to store pdf chunks.
    collection = chroma_client.get_or_create_collection("pdf_chunks")
    # Splitting the text into chunks.
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks= text_splitter.split_text(extracted_text)
    # Embedding those chunks.
    embedding_model= get_embedding_model()
    embeddings= embedding_model.encode(chunks)
    # Storing chunks, embeddings and its metadata into collection.
    for idx,chunk in enumerate(chunks):
        collection.add(
            ids=[f"{document_id}_{idx}"], documents=[chunk], embeddings=[embeddings[idx].tolist()], metadatas=[{"document_id":document_id,"pdf_name":file.filename,"chunk_number":idx}]
        )
    return {
        "message": "File Uploaded Successfully",
        "document_name": file.filename,
        "stored_path":file_path,
        "document_id": document_id,
        "total_chunks":len(chunks),
        "summary":summary
    }

@app.get("/stream_summary/{document_id}")
def stream_summary(document_id:int, user_id:int = Depends(get_current_user)):
    query=text("""select file_path from uploaded_documents where document_id=:document_id and user_id=:user_id""")
    engine= get_db_engine()
    with engine.begin() as conn:
        result=conn.execute(query,{"document_id":document_id, "user_id": user_id}).mappings().first()
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path= result["file_path"]
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File missing on server")
    reader = PdfReader(file_path)
    text_content=""
    for page in reader.pages:
        page_text=page.extract_text()
        if page_text:
            text_content+=page_text
    summary_text=text_content[:6000]
    summary_prompt=f"""
        You are an expert document assistant.

        Read the following document:

        {summary_text}

        Provide:
        - 1-2 natural paragraphs.
        - Then 5 concise takeaway points.

        Write naturally like ChatGPT.
        Do not mention that you are summarizing"""
    client=get_gemini_client()
    def generate():
        full_summary=""
        stream=client.models.generate_content_stream(
            model=model,
            contents=summary_prompt
        ) 
        try:
            for chunk in stream:
                if chunk.text:
                    full_summary+=chunk.text
                    yield chunk.text
            update_query=text("""update uploaded_documents set summary=:summary where document_id=:document_id""")
            with engine.begin() as conn:
                conn.execute(update_query,{"summary":full_summary,"document_id":document_id})
        except Exception as stream_err:
            logging.error(f"Stream Failed: {stream_err}")
            return
    return StreamingResponse( generate(),media_type="text/plain" )

# Fetch available documents
@app.get("/documents")
def get_documents(user_id:int= Depends(get_current_user)):
    query= text(""" select document_id, document_name, summary, created_at from uploaded_documents where user_id=:user_id order by created_at desc""")
    engine= get_db_engine()
    with engine.begin() as conn:
        documents= conn.execute(query,{"user_id":user_id}).mappings().all()
    return documents

# Chat request model to use it while response to question in chat.
class ChatRequest(BaseModel):
    document_id: int
    question: str

# Chat: Sending a question and fetching the response and then storing it in sql.
@app.post("/chat")
def chat_request(request: ChatRequest, user_id: int= Depends(get_current_user)):
    # Get the created collection.
    chroma_client=get_chroma_client()
    collection= chroma_client.get_collection("pdf_chunks")
    # Convert the question to embedding.
    embedding_model= get_embedding_model()
    question_embedding= embedding_model.encode(request.question).tolist()
    # Fetching the similar chunks by passing the question to collection and if not there then raise error.
    result= collection.query(query_embeddings=[question_embedding],n_results=5, where={"document_id":request.document_id})
    if not result["documents"] or not result["documents"][0]:
        raise HTTPException(
            status_code=404,
            detail="No relevant chunk found"
        )
    # Combining those similar chunks.
    retrieved_chunks=result["documents"][0]
    context="\n\n".join(retrieved_chunks)
    
    # Prompt to pass it to llm.
    prompt = f"""
        You are an expert AI assistant for document-based question answering.

        You will be given a CONTEXT from a document and a USER QUESTION.

        -------------------------
        CONTEXT:
        {context}
        -------------------------

        USER QUESTION:
        {request.question}
        -------------------------

        INSTRUCTIONS:

        1. First, determine relevance:
        - If the question is NOT related to the context at all, respond ONLY with:
            "This question is not related to the provided document context."

        - If the question is partially related OR somewhat connected to the context:
            → Try to answer using the context as much as possible.

        2. Answering rules (VERY IMPORTANT):
        - Use ONLY the provided context.
        - Do NOT use external knowledge or assumptions.
        - If information is missing, clearly mention it is not available in the context.
        - Do NOT say "out of context" unless completely unrelated.

        3. Response style:
        - Answer in clear bullet points (4–6 points if possible).
        - Keep explanations simple, clear, and structured.
        - Use a helpful, ChatGPT-like tone.
        - Avoid unnecessary repetition or long paragraphs.

        4. Quality requirement:
        - Focus on key ideas, insights, and important details.
        - Make the answer easy to read and well-structured.

        FINAL OUTPUT FORMAT:
        - Bullet point answer only
        - Or single rejection sentence if fully unrelated

        """
    client= get_gemini_client()
    def generate():
        response=""
        stream=client.models.generate_content_stream(
            model=model,
            contents=prompt
            ) 
        for chunk in stream:
            if chunk.text:
                response+=chunk.text
                yield chunk.text
        # Inserting the chat that is user's question and llm's response into sql.
        insert_chat_query= text("""insert into chat_history(user_id, document_id, question, answer) values(:user_id,:document_id,:question,:answer)""")
        engine = get_db_engine()
        with engine.begin() as conn:
            conn.execute(insert_chat_query,{"user_id":user_id,"document_id":request.document_id,"question":request.question,"answer":response})
        
    return (StreamingResponse(generate(),media_type="text/plain"))

# Fetching the chat history of a document.
@app.get("/chat-history/{document_id}")
def get_chat_history(document_id:int, user_id:int = Depends(get_current_user)):
    query=text(""" select chat_id, question, answer, created_at from chat_history where document_id=:document_id and user_id=:user_id order by created_at asc""")
    engine= get_db_engine()
    with engine.begin() as conn:
        history=conn.execute(query,{"document_id":document_id, "user_id": user_id}).mappings().all()
    return history

# Deleting a document. To do this we need to delete from local, sql and chromadb.
@app.delete("/document/{document_id}")
def delete_document(document_id:int):
    # Check whether the doucment is there or not. If not raise error. If it's there then fetch it's file path.
    query=text(""" select file_path from uploaded_documents where document_id=:document_id""")
    engine = get_db_engine()
    with engine.begin() as conn:
        document= conn.execute(query,{"document_id":document_id}).mappings().first()
        if not document:
            raise HTTPException(
                status_code= 404,
                detail= "Document not found."
            )
        file_path=document["file_path"]

    # Get the chromadb collection and delete the particular document.
    chroma_client= get_chroma_client()
    collection = chroma_client.get_collection("pdf_chunks")
    try:
        collection.delete(where={"document_id":document_id})
    except Exception:
        pass

    # Delete the document from chat_history and uploaded_documents table.
    with engine.begin() as conn:
         conn.execute(
             text(""" delete from chat_history where document_id = :document_id"""),{"document_id":document_id}
         )
         conn.execute(
             text(""" delete from uploaded_documents where document_id=:document_id"""),{"document_id":document_id}
         )

    # Delete it from local storage. 
    if os.path.exists(file_path):
        os.remove(file_path)
    
    return {
        "message":"Document deleted successfully.",
        "document_id":document_id
    }




