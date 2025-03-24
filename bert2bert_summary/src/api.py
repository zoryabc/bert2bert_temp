from fastapi import FastAPI
from inference import summarize

app = FastAPI()

@app.post("/summarize")
async def get_summary(text: str):
    return {"summary": summarize(text)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)