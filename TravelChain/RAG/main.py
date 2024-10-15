import uvicorn
from Module.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="220.69.155.89", port=6000)