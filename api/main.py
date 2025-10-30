from fastapi import FastAPI

app = FastAPI(title="Code-RAG MVP")

@app.get("/ping")
def ping():
    return {"ok": True}
