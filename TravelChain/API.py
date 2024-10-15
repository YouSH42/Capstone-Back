from fastapi import FastAPI, Request
from RAG.Module.indexing import update_index
import uvicorn

app = FastAPI()

def run():
    uvicorn.run(app, host="220.69.155.89", port=6000)

@app.post("/update-index")
async def update_index_handler(request: Request):
    jsonData: dict = request.json()
    update_index(jsonData)
    print(f"contentID.{jsonData["contentid"]}: Successfully update.")
    
    return {"message":f"contentID.{jsonData["contentid"]}: Successfully update."}
    