import pyrootutils
import gradio as gr
from gradio import components as gr_comp  # Import from gradio.components

import cv2
# from apply_filter.src.ApplyFilter import apply_filter_on_image, apply_filter_on_video
from deploy_function import apply_filter_on_image, apply_filter_on_video, apply_filter_on_webcam
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# def apply_filter_on_image(image, filter_name):
#     return cv2.imread("/Users/tiendzung/Project/facial_landmarks-wandb/notebooks/output2_image.jpg")

# examples
package_dir = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
example_1_dir = str("/Users/tiendzung/Project/facial_landmarks-wandb/notebooks/anonymous_example.png")
example_2_dir = str("/Users/tiendzung/Project/facial_landmarks-wandb/notebooks/squid_game_example.png")
example_3_dir = str("/Users/tiendzung/Project/facial_landmarks-wandb/notebooks/anonymous_example.png")
example_list = [[example_1_dir],[example_2_dir],[example_3_dir]]

# for app
title = "Filter app"
description = "Using simple resnet 18 to detect landmarks and then applying filter on faces"
article = "Created by Dzung"
filter_names = ["squid_game_front_man", "anonymous", "dog", "cat"]

# Create the Gradio demo
image_tab = gr.Interface(
    fn=apply_filter_on_image,
    inputs=[
        gr.Image(),
        gr_comp.Radio(choices=filter_names, label="Select a filter:")
    ],
    outputs=gr.Image(),
    examples=example_list,
    title=title,
    description=description,
    article=article
)

video_tab = gr.Interface(
    fn=apply_filter_on_video,
    inputs=[
        gr_comp.Video(),
        gr_comp.Radio(choices=filter_names, label="Select a filter:")
    ],
    # outputs=gr.Video(type='file'),
    outputs="video",
    title=title,
    description=description,
    article=article
)

webcam_tab = gr.Interface(
    fn=apply_filter_on_webcam,
    inputs=[
    #     gr.Video(),
        gr_comp.Radio(choices=filter_names, label="Select a filter:")
    ],
    # outputs=gr.Video(type='file'),
    # inputs=[],
    outputs="webcam",
    title=title,
    description=description,
    article=article
)

demo = gr.TabbedInterface([image_tab, video_tab, webcam_tab], ["Image", "Video", "Webcam"]) ##video_tab

# Launch the demo!
demo.queue().launch(debug=False, # print errors locally?
            share=False) # generate a publically shareable URL?

# launch the demo with docker
# demo.launch(server_name="0.0.0.0", server_port=7000, debug=True)