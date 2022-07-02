from fastapi import FastAPI
import uvicorn
from src.message_model import MessageModel

app = FastAPI()


@app.get("/")
def root():
    return "api running!"


@app.post("/messageModel/")
async def send_message(message:str):
    message_model=MessageModel()
    pred=message_model.predict(message=message)
    
    return pred




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)