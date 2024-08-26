from flask import Flask, request, send_file
import os
from ultralytics import YOLO
import cv2
import numpy as np
import math

model = YOLO("yolov8n-pose.pt")


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/vedio', methods=['GET'])
def vedio():
    return send_file('videos/workouts.mp4', as_attachment=False)

@app.route('/uploadVideo', methods=["POST"])
def uploadVideo():
    video = request.files['video'].stream.read()
    name = request.form['fileName']

    # 检查 video 目录是否存在,如果不存在则创建
    images_dir = os.path.join(os.getcwd(), 'videos')
    if not os.path.exists(images_dir):
        try:
            os.makedirs(images_dir)
        except OSError as e:
            # 处理创建目录失败的情况
            print(f"Error creating 'images' directory: {e}")
            return "Error uploading file", 500

    if not files_exists(name, 1):
        file_path = os.getcwd() + '/videos/' + name
        with open(file_path, 'ab') as f:
            f.write(video)
        # count.py
        cap = cv2.VideoCapture("videos/videoTest.mp4")
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        video_writer = cv2.VideoWriter("videos/workouts.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        count = 0
        pose = 3

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            im1 = im0
            results = model(source=im0, show=False, conf=0.3, save=False)

            keypoints = results[0].keypoints.data
            ankle1 = keypoints[0, 15, :2]
            ankle2 = keypoints[0, 16, :2]
            dis_ankle = math.sqrt((ankle1[0] - ankle2[0]) ** 2 + (ankle1[1] - ankle2[1]) ** 2)
            if dis_ankle > 400:
                pose = 1  # gb
            if dis_ankle < 200 and pose == 1:
                pose = 0  # szzs
                count += 1

            mid_x = (ankle1[0] + ankle2[0]) // 2
            mid_y = (ankle1[1] + ankle2[1]) // 2

            cv2.circle(im1, tuple(map(int, ankle1)), 5, (0, 0, 255), 2)  # 绘制ankle1
            cv2.circle(im1, tuple(map(int, ankle2)), 5, (0, 0, 255), 2)  # 绘制ankle2
            cv2.line(im1, tuple(map(int, ankle1)), tuple(map(int, ankle2)), (255, 255, 0), 2)
            cv2.putText(im1, f"{dis_ankle:.2f}°", tuple(map(int, ankle2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(im1, f"count = {count}", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            video_writer.write(im1)

        cv2.destroyAllWindows()
        video_writer.release()

        return 'upload success'

@app.route('/testCount')
def testCount():
    cap = cv2.VideoCapture("videos/videoTest.mp4")
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter("videos/workouts.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    count = 0
    pose = 3

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        im1 = im0
        results = model(source=im0, show=False, conf=0.3, save=False)

        keypoints = results[0].keypoints.data
        ankle1 = keypoints[0, 15, :2]
        ankle2 = keypoints[0, 16, :2]
        dis_ankle = math.sqrt((ankle1[0] - ankle2[0]) ** 2 + (ankle1[1] - ankle2[1]) ** 2)
        if dis_ankle > 400:
            pose = 1  # gb
        if dis_ankle < 200 and pose == 1:
            pose = 0  # szzs
            count += 1

        mid_x = (ankle1[0] + ankle2[0]) // 2
        mid_y = (ankle1[1] + ankle2[1]) // 2

        cv2.circle(im1, tuple(map(int, ankle1)), 5, (0, 0, 255), 2)  # 绘制ankle1
        cv2.circle(im1, tuple(map(int, ankle2)), 5, (0, 0, 255), 2)  # 绘制ankle2
        cv2.line(im1, tuple(map(int, ankle1)), tuple(map(int, ankle2)), (255, 255, 0), 2)
        cv2.putText(im1, f"{dis_ankle:.2f}°", tuple(map(int, ankle2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(im1, f"count = {count}", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        video_writer.write(im1)

    cv2.destroyAllWindows()
    video_writer.release()
    return send_file('videos/workouts.mp4', as_attachment=False)


@app.route('/uploadImage', methods=["POST"])
def uploadImage():
    video = request.files['photo'].stream.read()
    name = request.form['fileName']

    # 检查 images 目录是否存在,如果不存在则创建
    images_dir = os.path.join(os.getcwd(), 'images')
    if not os.path.exists(images_dir):
        try:
            os.makedirs(images_dir)
        except OSError as e:
            # 处理创建目录失败的情况
            print(f"Error creating 'images' directory: {e}")
            return "Error uploading file", 500
    file_path = os.getcwd() + '/images/' + name
    print(file_path)
    with open(file_path, 'ab') as f:
        f.write(video)
    results = model(source=file_path, show=False, conf=0.3, save=False)
    keypoints = results[0].keypoints.data
    print(keypoints)
    img = cv2.imread(file_path)
    szzs = 1
    p1 = keypoints[0, 6, :2] # 肩
    p4 = keypoints[0, 5, :2] # 肩
    p2 = keypoints[0, 8, :2] #  肘
    p11 = keypoints[0, 7, :2] # 肘
    p12 = keypoints[0, 9, :2] # 手
    p3 = keypoints[0, 10, :2]  # 手
    p9 = keypoints[0, 12, :2] # 胯
    p10 = keypoints[0, 11, :2] # 胯
    p5 = keypoints[0, 14, :2] # 膝
    p7 = keypoints[0, 13, :2] # 膝
    p6 = keypoints[0, 16, :2] # 踝
    p8 = keypoints[0, 15, :2] # 踝
    v1 = p2 - p1  # 第7行和第9行形成的向量
    v2 = p2 - p3  # 第9行和第11行形成的向量
    dot = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    angle = np.arccos(dot / (mag1 * mag2))
    angle_deg = np.rad2deg(angle)
    ankle1 = keypoints[0, 15, :2]
    ankle2 = keypoints[0, 16, :2]
    dis_ankle = math.sqrt((ankle1[0] - ankle2[0]) ** 2 + (ankle1[1] - ankle2[1]) ** 2)
    shoulder1 = keypoints[0, 5, :2]
    shoulder2 = keypoints[0, 6, :2]
    dis_shoulder = math.sqrt((shoulder1[0] - shoulder2[0]) ** 2 + (shoulder1[1] - shoulder2[1]) ** 2)

    cv2.circle(img, tuple(map(int, p1)), 5, (255, 0, 0), 2)  # 绘制p1
    cv2.circle(img, tuple(map(int, p2)), 5, (0, 255, 0), 2)  # 绘制p2
    cv2.circle(img, tuple(map(int, p3)), 5, (0, 0, 255), 2)  # 绘制p3
    cv2.circle(img, tuple(map(int, p4)), 5, (255, 10, 0), 2)
    cv2.circle(img, tuple(map(int, p5)), 5, (0, 255, 10), 2)
    cv2.circle(img, tuple(map(int, p6)), 5, (0, 0, 255), 2)
    cv2.circle(img, tuple(map(int, p7)), 5, (255, 0, 0), 2)
    cv2.circle(img, tuple(map(int, p8)), 5, (0, 255, 0), 2)
    cv2.circle(img, tuple(map(int, p9)), 5, (0, 0, 255), 2)
    cv2.circle(img, tuple(map(int, p10)), 5, (0, 0, 255), 2)
    cv2.circle(img, tuple(map(int, p11)), 5, (0, 0, 255), 2)
    cv2.circle(img, tuple(map(int, p12)), 5, (0, 0, 255), 2)
    cv2.line(img, tuple(map(int, p1)), tuple(map(int, p2)), (255, 255, 0), 2)  # 连接p1和p2
    cv2.line(img, tuple(map(int, p2)), tuple(map(int, p3)), (255, 255, 0), 2)  # 连接p2和p3
    cv2.line(img, tuple(map(int, p4)), tuple(map(int, p11)), (255, 255, 0), 2)  # 连接p1和p2
    cv2.line(img, tuple(map(int, p11)), tuple(map(int, p12)), (255, 255, 0), 2)
    cv2.line(img, tuple(map(int, p9)), tuple(map(int, p5)), (255, 255, 0), 2)
    cv2.line(img, tuple(map(int, p5)), tuple(map(int, p6)), (255, 255, 0), 2)
    cv2.line(img, tuple(map(int, p10)), tuple(map(int, p7)), (255, 255, 0), 2)
    cv2.line(img, tuple(map(int, p7)), tuple(map(int, p8)), (255, 255, 0), 2)
    cv2.putText(img, f"{angle_deg:.2f}°", tuple(map(int, p2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    cv2.imwrite("images/done.jpg", img)  # 保存图像

    return 'success'



@app.route('/static')
def static_demo():
    return send_file("images/done.jpg", mimetype='image/jpeg')

def files_exists(file_name, choice):
    if choice == 1:
        path = os.getcwd() + '\\videos\\'
        video_path = os.path.join(path, file_name)
        return os.path.isfile(video_path)
    else:
        path = os.getcwd() + '\\images\\'
        image_path = os.path.join(path, file_name)
        return os.path.isfile(image_path)

@app.route('/home')
def home_image():
    return send_file("images/home.png", mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
