from fastapi import FastAPI, Request
from indexing import update_index

app = FastAPI()

@app.post("/update-index")
async def update_index_handler(request: Request):
    jsonData: dict = request.json()
    update_index(jsonData)
    print(f"contentID.{jsonData["contentid"]}: Successfully update.")
    
    return {"message":f"contentID.{jsonData["contentid"]}: Successfully update."}
    