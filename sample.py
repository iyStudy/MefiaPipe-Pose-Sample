import cv2
import mediapipe as mp

# mediapipeの描画ユーティリティと姿勢推定モデルを読み込む
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ウェブカメラからの入力を開始する
cap = cv2.VideoCapture(0)
# pose（姿勢）モデルを用いて画像から姿勢を推定する。最小検出信頼度と最小追跡信頼度はともに0.5に設定
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # カメラがオープンしている限りループを続ける
    while cap.isOpened():
        # カメラからフレームを読み込む
        ret, frame = cap.read()
        # カメラからのフレームが空だった場合は無視する
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # セルフィービュー（左右反転）表示のために画像を水平に反転させ、BGR画像をRGBに変換する
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # パフォーマンスを向上させるため、参照渡しを行うために画像を書き込み不可に設定する
        image.flags.writeable = False
        # 姿勢推定を行う
        results = pose.process(image)

        # 画像に推定結果のアノテーション（ランドマークや姿勢の接続）を描画する
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 画像を表示する
        cv2.imshow('MediaPipe Pose', image)
        # 'ESC'キーが押されるとループを抜ける
        if cv2.waitKey(5) & 0xFF == 27:
            break

# カメラへの接続を解除し、ウィンドウを全て閉じる
cap.release()
cv2.destroyAllWindows()
