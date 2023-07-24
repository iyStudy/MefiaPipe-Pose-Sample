import cv2
import mediapipe as mp
import random
import time
import numpy as np

# mediapipeの描画ユーティリティと姿勢推定モデルを読み込む
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 使用する指示のリストを設定
instructions = ['Raise your right hand', 'Raise your left hand', 'Lower your right hand', 'Lower your left hand']
# 最初の指示をランダムに選択
current_instruction = random.choice(instructions)
# 最初の指示の色をランダムに設定
instruction_color = np.random.randint(0, high=256, size=3).tolist() 
# 右手と左手が上げられているかの状態を追跡
right_hand_raised = False
left_hand_raised = False
# OKメッセージと評価のタイミングを追跡
ok_time = None
eval_time = None

# ウェブカメラからの入力を開始する
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
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

        # 姿勢のランドマークが得られた場合、指示に従ったかを判断するロジックを実行する
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # 指示文を画面上に表示するための矩形を描画
            cv2.rectangle(image, (0, 0), (640, 75), (255,255,255), -1)

            # 現在の指示を画面上に表示
            cv2.putText(image, current_instruction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, instruction_color, 2)

            # 初回の評価時刻を設定
            if eval_time is None:
                eval_time = time.time()

            # 指示が出されてから2秒後に、ユーザーの姿勢が指示に従っているかを評価
            if time.time() - eval_time > 2: 
                if current_instruction == 'Raise your right hand' and right_wrist.y < right_shoulder.y:
                    right_hand_raised = True

                if current_instruction == 'Raise your left hand' and left_wrist.y < left_shoulder.y:
                    left_hand_raised = True

                if current_instruction == 'Lower your right hand' and right_wrist.y > right_shoulder.y:
                    right_hand_raised = False

                if current_instruction == 'Lower your left hand' and left_wrist.y > left_shoulder.y:
                    left_hand_raised = False

                # 評価の結果が正しい場合、画面上に「OK」メッセージを表示
                if (current_instruction == 'Raise your right hand' and right_hand_raised) or \
                (current_instruction == 'Raise your left hand' and left_hand_raised) or \
                (current_instruction == 'Lower your right hand' and not right_hand_raised) or \
                (current_instruction == 'Lower your left hand' and not left_hand_raised):
                    cv2.rectangle(image, (0, 80), (150, 130), (255,255,255), -1)
                    cv2.putText(image, 'OK', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    if ok_time is None:
                        ok_time = time.time()

                # 「OK」メッセージが表示されてから2秒後、新しい指示をランダムに選択
                if ok_time is not None and time.time() - ok_time > 2:
                    current_instruction = random.choice(instructions)
                    instruction_color = np.random.randint(0, high=256, size=3).tolist() 
                    ok_time = None
                    eval_time = None

        # 画像を表示する
        cv2.imshow('MediaPipe Pose', image)
        # 'ESC'キーが押されるとループを抜ける
        if cv2.waitKey(5) & 0xFF == 27:
            break

# カメラへの接続を解除し、ウィンドウを全て閉じる
cap.release()
cv2.destroyAllWindows()
