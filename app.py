from fastapi import FastAPI
import uvicorn
from src.utils import MessageModel, VideoModel, AudioModel
import pandas as pd

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
    pie_chart = pred.groupby(['Human Emotions']).sum().plot(
                                                        kind='pie', 
                                                        y='Emotion Value from the Video',
                                                        figsize=(20,20),
                                                        title="Emations Percentage Values")
    
    pie_chart.get_figure().savefig("graphs/pie_chart.png")
    print(pie_chart)
    return pie_chart

@app.post("/audio/")
async def audio():
    audio=AudioModel()
    response=audio.predict_audio("data/v1_audio.wav")
    print(response)

    return response

