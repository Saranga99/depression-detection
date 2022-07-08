from fastapi import FastAPI
from src.utils import MessageModel, VideoModel, AudioModel
import pandas as pd
import shutil
import os

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
    df = pd.read_csv("data.csv")
    graph_1=df.plot.line(subplots=True, figsize=(20,20),title="Emotion Variations in Video")
    graph_2=df.plot(figsize=(20,20),title="Emotion Variations in Video")
    #save graphs
    graphs="graphs/"
    os.makedirs(graphs,exist_ok=True)
    graph_1.get_figure().savefig("graphs/graph_1.png")
    graph_2.get_figure().savefig("graphs/graph_2.png")
    pie_chart.get_figure().savefig("graphs/pie_chart.png")
    #check dipression
    angry=df[df['Human Emotions']=="Angry"]['Emotion Value from the Video'].values[0]
    sad=df[df['Human Emotions']=="Sad"]['Emotion Value from the Video'].values[0]
    happy=df[df['Human Emotions']=="Happy"]['Emotion Value from the Video'].values[0]
    output="Neutral"
    if ((angry+sad)/2) > happy:
        output="Depressed"
    else:
        output="Positive"
    shutil.rmtree("data.csv")

    return output


@app.post("/audio/")
async def audio():
    audio=AudioModel()
    response=audio.predict_audio("data/v1_audio.wav")
   
    return response

