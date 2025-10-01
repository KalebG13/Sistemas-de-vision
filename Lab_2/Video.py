from yt_dlp import YoutubeDL
from pathlib import Path

def download_youtube_mp4(url: str, out_dir: str = "videos") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "outtmpl": str(Path(out_dir) / "%(title)s.%(ext)s"),
        # trata de obtener MP4 720p (o el mejor disponible) y fusiona en MP4
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)
        # Asegurar extensión .mp4 si se fusionó
        if not filepath.endswith(".mp4"):
            filepath = Path(filepath).with_suffix(".mp4")
        return str(filepath)

# Ejemplo:
video_path = download_youtube_mp4("https://www.youtube.com/watch?v=b-WViLMs_4c")
print(video_path)
