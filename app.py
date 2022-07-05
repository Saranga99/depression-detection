from fastapi import FastAPI
import uvicorn
from src.utils import MessageModel, VideoModel

app = FastAPI()


@app.get("/")
def root():
    return "api running!"


@app.post("/messageModel/")
async def message(message:str):
    message_model=MessageModel()
    pred=message_model.predict(message=message)
    return pred

@app.post("/video/")
async def video(path:str):
    video_model=VideoModel()
    pred=video_model.predict_using_video(video_path="/data/data.mp4")
    return pred

