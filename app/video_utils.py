from moviepy import VideoFileClip

def extract_audio(video_path, output_audio="temp.wav"):

    video = VideoFileClip(video_path)

    video.audio.write_audiofile(
        output_audio,
        logger=None
    )

    video.close()

    return output_audio