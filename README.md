
## Start and run the app
```bash
# run the server
$ cd backend && ./rtsp-simple-server
# start the app
$ cd frontend && streamlit run app.py
# receive the stream
$ ffmpeg -i rtsp://localhost:8554/mystream -c copy video.mp4
```
This `video.mp4` file is the received video of by the server. So you can process it and make new
video (`output.mp4`) with details.