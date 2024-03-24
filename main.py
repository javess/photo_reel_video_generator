import os
from argparse import ArgumentParser
import cv2
from datetime import datetime
from typing import Tuple, List, Optional


def crop_based_on_face(image_path: str, face: Tuple[int, int, int, int], target_face_width: int = 400, target_face_height: int = 400, target_image_width: int = 4000, target_image_height: int = 4000) -> np.ndarray:
    image = cv2.imread(image_path)
    (ff_x, ff_y, ff_w, ff_h) = face

    y = max(0, ff_y - 200)
    x = ff_x + (ff_w // 2) - (target_image_width // 2)
    cropped_image = image[y:y + target_image_height, x:x + target_image_width]
    cropped_height, cropped_width, _ = cropped_image.shape

    resized_height = 1080
    resized_width = (resized_height * cropped_width) // cropped_height
    resized_image = cv2.resize(cropped_image, (resized_width, resized_height))

    return resized_image


def detect_faces(image_path: str, max_y: int) -> Optional[Tuple[int, int, int, int]]:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = [(x, y, w, h) for (x, y, w, h) in face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=9, minSize=(250, 250)) if y <= max_y]

    min_y = float('inf')
    face = None
    for (x, y, w, h) in faces:
        if y < min_y:
            min_y = y
            face = (x, y, w, h)

    return face


def get_cropped_image(image_path: str, max_y: int) -> np.ndarray:
    face_anchor = detect_faces(image_path, max_y)
    return crop_based_on_face(image_path, face_anchor)


def get_alpha_sigmoid(alpha_value: int) -> float:
    return 0.0 if alpha_value < 20 else 1.0 if alpha_value > 80 else (alpha_value - 20) / 60.0


def create_video(image_folder: str, fps: int) -> None:
    images = sorted([img for img in os.listdir(
        image_folder) if img.endswith(".jpg")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width = frame.shape[:2]
    current_datetime_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    video = cv2.VideoWriter(f'./out/date_{current_datetime_string}__fps_{
                            fps}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i in range(len(images) - 1):
        print(f"image {i}/{len(images)}")
        img1 = get_cropped_image(os.path.join(image_folder, images[i]), 1500)
        img2 = get_cropped_image(os.path.join(
            image_folder, images[i + 1]), 1500)

        for alpha in range(0, 101):
            blended = cv2.addWeighted(img1, get_alpha_sigmoid(
                100 - alpha), img2, get_alpha_sigmoid(alpha), 0)
            video.write(blended)

    for _ in range(30):
        video.write(img2)

    video.release()
    cv2.destroyAllWindows()


def rename_files(folder_path: str) -> None:
    files_with_creation_time = [(file, os.path.getctime(
        os.path.join(folder_path, file))) for file in os.listdir(folder_path)]
    files_sorted_by_creation_time = sorted(
        files_with_creation_time, key=lambda x: x[1], reverse=True)

    for idx, (file, _) in enumerate(files_sorted_by_creation_time, start=1):
        new_file_path = os.path.join(folder_path, f'{idx:03d}.jpg')
        os.rename(os.path.join(folder_path, file), new_file_path)


def parse_arguments() -> argparse.Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", dest="directory",
                        help="Directory containing photos", metavar="DIR", required=True)
    parser.add_argument("-r", "--rename_files", action="store_true", dest="rename_files",
                        default=False, help="Rename files in the directory to be in order of creation time")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    folder_path = args.directory

    if args.rename_files:
        print("Renaming all files to follow indexed pattern")
        rename_files(folder_path)

    create_video(folder_path, fps=90)


if __name__ == "__main__":
    main()
