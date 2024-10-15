import uvicorn
from RAG import API

if __name__ == "__main__":
    uvicorn.run(API.app, host="220.69.155.89", port=6000)