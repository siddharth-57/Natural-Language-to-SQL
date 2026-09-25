from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI(title="Demo Web App Backend")

BENTOML_SERVICE_URL = "http://localhost:3000/query"

class QueryRequest(BaseModel):
    question: str
    columns: list[str]

@app.post("/ask")
async def ask(request: QueryRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            BENTOML_SERVICE_URL,
            json={"question": request.question, "columns": request.columns}
        )
        print(response.json())
        return response.json()
