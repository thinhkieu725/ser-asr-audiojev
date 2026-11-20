from audiojev_inference import AudioJev, Choice

model = AudioJev(device="cuda:0")
EMO = {
    "neutral": "The speaker sounds calm and emotionally neutral",
    "happy":   "The speaker sounds happy or joyful",
    "sad":     "The speaker sounds sad",
    "angry":   "The speaker sounds angry",
    "fear":    "The speaker sounds fearful or anxious",
    "disgust": "The speaker sounds disgusted",
    "surprise":"The speaker sounds surprised",
}
res = model.system_one(
    state="data/Crema/1001_DFA_ANG_XX.wav",
    questions={"emo": Choice(
        instructions="Judging by tone of voice, which emotion does the speaker express?",
        criteria=EMO)},
)
print(res["answers"]["emo"])