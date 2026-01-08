from roarm_sdk.roarm import roarm
import pygame
import time
import cv2
import numpy as np
from collections import deque
from datetime import datetime

pygame.init()
pygame.joystick.init()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("cam not connected")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()


arm = roarm(
    roarm_type="roarm_m2",
    port="COM6",
    baudrate=115200
)

pose = [300, 0, 200, 0]
STEP = 5
UPDATE_RATE = 30
dt = 1.0 / UPDATE_RATE
running = True
recording = False

arm.pose_ctrl(pose)
time.sleep(2)



frame_stack = deque(maxlen=4)

images_data = []
poses_data = []
actions_data = []
timestamps_data = []

def save_session():
    if len(images_data) == 0:
        print("no data for save")
        return

    filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez_compressed(
        filename,
        images=np.array(images_data, dtype=np.uint8),
        poses=np.array(poses_data, dtype=np.float32),
        actions=np.array(actions_data, dtype=np.float32),
        timestamps=np.array(timestamps_data, dtype=np.float64),
    )
    print(f"Saved to: {filename}")

while running:
    start = time.time()
    pygame.event.pump()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (256, 256))
    frame_stack.append(frame)

    cv2.putText(
        frame,
        "REC" if recording else "IDLE",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255) if recording else (255, 255, 255),
        2,
    )
    cv2.imshow("Camera", frame)

    action = [0, 0, 0, 0, 0, 0, 0]
    moved = False

    hat_x, hat_y = joystick.get_hat(0)

    if joystick.get_axis(0) <-0.5:

        pose[1] += STEP
        action[0] = 1
        moved = True
    if joystick.get_axis(0)>0.5:
        pose[1] -= STEP
        action[1] = 1
        moved = True

    if joystick.get_axis(1)<-0.5:
        pose[0] += STEP
        action[2] = 1
        moved = True
    if joystick.get_axis(1)>0.5:
        pose[0] -= STEP
        action[3] = 1
        moved = True

    if joystick.get_axis(3)<-0.5:
        pose[2] += STEP
        action[4] = 1
        moved = True
    if joystick.get_axis(3)>0.5:
        pose[2] -= STEP
        action[5] = 1
        moved = True
    if joystick.get_button(1):
        pose=[300,0,200,0]
        moved = True

    for event in pygame.event.get():
        if event.type == pygame.JOYBUTTONDOWN:

            if event.button == 2:
                recording = not recording

                if recording:
                    print("Recording")
                    images_data.clear()
                    poses_data.clear()
                    actions_data.clear()
                    timestamps_data.clear()
                else:
                    print("stopped recording")
                    save_session()

            if event.button == 0:
                pose[3] = 0 if pose[3] == 90 else 90
                action[6] = 1
                moved = True

    pose[2] = max(-100, min(300, pose[2]))

    if moved:
        arm.pose_ctrl(pose)

    if recording and len(frame_stack) == 4:
        images_data.append(np.stack(frame_stack, axis=0))
        poses_data.append(pose.copy())
        actions_data.append(action.copy())
        timestamps_data.append(time.time())

    if joystick.get_button(6):
        running = False

    elapsed = time.time() - start
    if elapsed < dt:
        time.sleep(dt - elapsed)

cap.release()
pygame.quit()
cv2.destroyAllWindows()
