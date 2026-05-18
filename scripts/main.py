import uvicorn

from scripts.api import app

if __name__ == "__main__":

    uvicorn.run(
        "scripts.api:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )

