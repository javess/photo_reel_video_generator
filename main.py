import os
from argparse import ArgumentParser
import cv2
from datetime import datetime


def crop_based_on_face(image_path, face, target_face_width=400, target_face_height=400, target_image_width=4000, target_image_height=4000):
    # Load the image
    image = cv2.imread(image_path)

    (ff_x, ff_y, ff_w, ff_h) = face

    y = max(0, ff_y - 200)
    x = ff_x + (ff_w//2) - (target_image_width//2)
    cropped_image = image[y:y+target_image_height, x:x+target_image_width]
    cropped_height, cropped_width, _ = cropped_image.shape

    resized_height = 1080
    resized_width = (resized_height * cropped_width) // cropped_height

    resized_image = cv2.resize(cropped_image, (resized_width, resized_height))
    # resized_image_2 = cv2.resize(
    #     resized_image, (resized_width*2, resized_height*2))
    # Draw rectangles around the detected faces
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Cropped Image', cropped_image)
    # cv2.imshow('Resize Image', resized_image)
    # cv2.imshow('Resize Image 2', resized_image_2)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return resized_image


def detect_faces(image_path, max_y):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = [(x, y, w, h) for (x, y, w, h) in face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=9, minSize=(250, 250)) if y <= max_y]

    # Draw rectangles around the detected faces
    min_y = 1000000
    face = None
    for (x, y, w, h) in faces:
        if (y < min_y):
            min_y = y
            face = (x, y, w, h)
        # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the image with detected faces
    # cv2.imshow('Detected Faces', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return face


def get_cropped_image(image_path, max_y):
    face_anchor = detect_faces(image_path, max_y)
    return crop_based_on_face(image_path, face_anchor)
# Function to create a video from a series of images


def get_alpha_sigmoid(alpha_value):
    if alpha_value < 20:
        return 0.0

    if alpha_value > 80:
        return 1.0

    return (alpha_value - 20)/60.0


def create_video(image_folder, fps=25):
    images = sorted([img for img in os.listdir(
        image_folder) if img.endswith(".jpg")])

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width = None, None
    # Get the current date and time
    current_datetime = datetime.now()

    # Convert the datetime object to a string
    current_datetime_string = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")

    img1, img2 = None, None
    for i in range(len(images) - 1):
        print(f"image {i}/{len(images)}")
        img1 = get_cropped_image(os.path.join(
            image_folder, images[i]), 1500)
        img2 = get_cropped_image(os.path.join(
            image_folder, images[i+1]), 1500)

        if height is None:
            height, width, _ = img1.shape
            video = cv2.VideoWriter(f'./out/date_{current_datetime_string}__fps_{fps}.mp4', cv2.VideoWriter_fourcc(
                *'mp4v'), fps, (width, height))

        # Resize images to ensure they have the same dimensions
        img1_resized = cv2.resize(img1, (width, height))
        img2_resized = cv2.resize(img2, (width, height))

        # Apply cross dissolve effect
        for alpha in range(0, 101):
            blended = cv2.addWeighted(
                img1_resized, get_alpha_sigmoid(100 - alpha), img2_resized, get_alpha_sigmoid(alpha), 0)
            video.write(blended)

    # Pin some frames time on the final video
    for i in range(30):
        video.write(img2)

    cv2.destroyAllWindows()
    video.release()


parser = ArgumentParser()
parser.add_argument("-d", "--dir", dest="directory",
                    help="Directory containing photos", metavar="DIR", required=True)
parser.add_argument("-r", "--rename_files",
                    action="store_true", dest="rename_files", default=False,
                    help="don't print status messages to stdout")

args = parser.parse_args()

# Define the folder path
folder_path = args.directory

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Get creation time for each file and sort them based on creation time
files_with_creation_time = [(file, os.path.getctime(
    os.path.join(folder_path, file))) for file in files]
files_sorted_by_creation_time = sorted(
    files_with_creation_time, key=lambda x: x[1], reverse=True)

idx = 1
# Print the sorted list of files
if (args.rename_files):
    print("renaming all files to follow indexed pattern")
    for file, creation_time in files_sorted_by_creation_time:
        print(f'{file}: {creation_time} -> {idx:03d}.jpg')
        # Create the new file path by joining the directory of the current file with the new file name
        new_file_path = os.path.join(folder_path, f'{idx:03d}.jpg')

        # Rename the file
        os.rename(os.path.join(folder_path, file), new_file_path)
        idx = idx + 1

# Example usage
image_folder = args.directory  # Specify the folder containing your images
create_video(image_folder, fps=90)
