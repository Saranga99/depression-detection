from fastapi import FastAPI
import uvicorn
from src.utils import MessageModel, VideoModel, AudioModel

app = FastAPI()


@app.get("/")
def root():
    return "api running!"


@app.post("/text/")
async def text(message:str):
    message_model=MessageModel()
    pred=message_model.predict(message=message)
    
    return pred

@app.post("/video/")
async def video():
    video_model=VideoModel()
    pred=video_model.predict_using_video(video_path="data/data.mp4")
    print(pred)
    print(type(pred))

    return pred

@app.post("/audio/")
async def audio():
    audio=AudioModel()
    response=audio.predict_audio("data/v1_audio.wav")
    print(response)

    return response

