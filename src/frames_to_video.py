import cv2
import os

def generate_video(path_to_frames, output_video_path, fps):
    # path to the directory containing frames
    frames_dir = path_to_frames
    output_video_path = output_video_path

    # get the list of all image files (sorted to maintain the sequence)
    images = [img for img in os.listdir(frames_dir) if img.endswith((".png",".jpg","jpeg"))]
    images.sort()

    print("?>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ",cv2.VideoWriter_fourcc(*'mp4v'))
    # Check if we have images
    if not images:
        raise ValueError("No image frames found in the directory.")

    # read the first image to get width and height
    first_frame = cv2.imread(os.path.join(frames_dir, images[0]))
    height, width, layers = first_frame.shape

    # define the codec and create VideoWriter object 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = fps
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Loop through images and write to video
    for image in images:
        img_path = os.path.join(frames_dir, image)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Skipping unreadable image: {img_path}")
            continue
        video.write(frame)

    # Release everthing
    video.release()
    cv2.destroyAllWindows()

    print(f"Video saved to {output_video_path}")

generate_video(r"data\Crowd_PETS09\S2\L1\Time_12-34\View_001", r"videos\video_001.mp4", 10)
generate_video(r"data\Crowd_PETS09\S2\L1\Time_12-34\View_005", r"videos\video_005.mp4", 10)