#!/usr/bin/env python

# from setuptools import find_packages, setup

# setup(
#     name="src",
#     version="0.0.1",
#     description="Describe Your Cool Project",
#     author="",
#     author_email="",
#     url="https://github.com/user/project",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
#     install_requires=["pytorch-lightning", "hydra-core"],
#     packages=find_packages(),
# )

from moviepy.editor import VideoFileClip
videoClip = VideoFileClip("/Users/tiendzung/Project/facial_landmarks-wandb/outputs/output.mp4")
videoClip.write_gif("final_demo.gif")