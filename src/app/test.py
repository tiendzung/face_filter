# import cv2
# import time

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,700) #640
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,500) #480
# cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# prev_time = time.time()

# while cap.isOpened():
#     cur_time = time.time()
#     print(1 / (cur_time - prev_time))
#     prev_time = cur_time
#     isSuccess, frame = cap.read()
#     cv2.imshow('Cam', frame)
#     if cv2.waitKey(1)&0xFF == 27:
#         print("YES")
#         break


import gradio as gr

def process_video(video_path):
    # Your video processing code here
    return video_path  # return the video path

iface = gr.Interface(
    fn=process_video, 
    inputs=gr.Video(), 
    outputs='video'
)
iface.launch()

