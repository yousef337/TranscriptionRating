import whisper

def service(audio):
    model = whisper.load_model("base")
    result = model.transcribe(audio)
    return result["text"]

