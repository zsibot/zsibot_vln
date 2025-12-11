from moviepy.editor import ImageSequenceClip
import numpy as np
import cv2
import os

def save_video(image_list, video_path, fps=30, resize_factor=1.0, input_color_space="RGB"):
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    first_frame = image_list[0]
    original_height, original_width = first_frame.shape[:2]

    new_width = int(original_width * resize_factor)
    new_height = int(original_height * resize_factor)

    new_width = new_width if new_width % 2 == 0 else new_width - 1
    new_height = new_height if new_height % 2 == 0 else new_height - 1

    new_width = max(2, new_width)
    new_height = max(2, new_height)

    resized_images = []
    for img in image_list:
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        if img.shape[-1] == 3:
            if input_color_space == "BGR":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif input_color_space == "RGB":
                pass
            else:
                raise ValueError("Unsupported input_color_space. Use 'RGB' or 'BGR'.")

        resized_img = cv2.resize(
            img,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )
        resized_images.append(resized_img)

    clip = ImageSequenceClip(resized_images, fps=fps)
    clip.write_videofile(
        video_path,
        codec="libx264",
        audio=False,
        threads=4,
        ffmpeg_params=["-pix_fmt", "yuv420p"]
    )