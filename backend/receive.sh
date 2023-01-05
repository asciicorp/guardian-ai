#!/bin/sh
ffmpeg -i rtsp://localhost:8554/mystream -c copy video.mp4