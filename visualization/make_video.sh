#!/usr/bin/env bash

# JUST FOR DEBUGGING
rm -rf video
mkdir video

cd jpg
for file in *.jpg; do
  convert $file -crop 1500x800+400+1 ../video/$file
done
cd ..

cd video
ffmpeg -framerate 1 -i %04d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ../network_visualization.mp4
cd ..

rm -rf video
