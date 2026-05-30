from faster_whisper import WhisperModel

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

def audio_to_text(audio_path):

    segments, info = model.transcribe(
        audio_path,
        beam_size=5
    )

    transcript = ""

    for segment in segments:
        transcript += segment.text + " "

    return transcript.strip()