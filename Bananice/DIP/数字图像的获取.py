import cv2
# 打开摄像头 #
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('video', 640, 480)
cap = cv2.VideoCapture(0)
# 循环读取摄像头的每一帧
while True:
    flag, frame = cap.read()
    if not flag:
        print("没读到数据！退出......")
        break

    # 显示数据
    else:
        cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
