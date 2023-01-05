import subprocess

class Controller:
  def __init__(self, vid_stream):
    self.stream = vid_stream

  def stream(self):
    self.publish_stream()

  def publish_stream(self):
    command = f"ffmpeg -re -stream_loop -1 -i {self.stream} -c copy -f rtsp rtsp://localhost:8554/mystream"
    subprocess.call(command, shell=True)
