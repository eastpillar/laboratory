import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image
import torch


def main():
    # ==== [사용자 설정 부분] ====
    list_path = "/home/aivs/바탕화면/hdd/adj_test/dataset/img_list2.txt"
    output_dir = "/home/aivs/바탕화면/hdd/adj_test/dataset/crop_image"
    img_size = (2048, 1536)   # (width, height) - 현재 저장 리사이즈는 (224,224) 그대로 유지함
    ad = 0.6
    min_conf = 0.9
    # =========================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)
    detector = MTCNN(keep_all=True, device=DEVICE)  # 모든 얼굴 받아서 후처리로 선택
    isPlot = False

    with open(list_path, 'r', encoding='utf-8') as f:
        img_paths = [ln.strip() for ln in f if ln.strip()]

    # 폴백용 상태: 같은 시퀀스에서만 prev_box 사용
    prev_box = None                  # np.array([x1,y1,x2,y2]) in float
    prev_key = None                  # (day, num, video_name)

    print(f"총 {len(img_paths)}장의 이미지를 처리합니다.")
    for j, img_full_path in enumerate(tqdm(img_paths, desc="Cropping")):
        # 경로 파싱: 시퀀스 키 추출
        temp = img_full_path.split('/')
        if len(temp) < 5:
            print(f"[경고] 경로 구조가 예상과 다릅니다: {img_full_path}")
            # 시퀀스 키를 알 수 없으면 prev_box를 초기화
            prev_box = None
            prev_key = None
            # 계속 진행
        day, num, video_name, img_name = temp[-5], temp[-4], temp[-2], temp[-1]
        cur_key = (day, num, video_name)

        cur_idx = int(os.path.splitext(img_name)[0])
        is_first_frame = (cur_idx == 0)

        # 시퀀스 변경 시 prev_box 초기화
        if prev_key is None or cur_key != prev_key:
            prev_box = None
            prev_key = cur_key

        img = cv2.imread(img_full_path)
        if img is None:
            print(f"[경고] 이미지를 읽지 못했습니다: {img_full_path}")
            continue

        img_h, img_w = img.shape[:2]
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # === (1) 얼굴 탐지 ===
        boxes, probs, landmarks = detector.detect(img_pil, landmarks=True)

        use_prev = False
        best_box = None
        best_conf = None

        if boxes is None or len(boxes) == 0:
            # 탐지 실패 → 같은 시퀀스라면 prev_box 폴백
            if prev_box is not None:
                use_prev = True
                best_box = prev_box.copy()
                best_conf = None
            else:
                print(f"[경고] 얼굴을 찾지 못했습니다: {img_full_path}")
                continue
        else:
            # === (2) 여러 얼굴 중 '아이 우선' 선택: 크기 + 화면 중앙 근접 점수 ===
            areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=float)

            # 얼굴 중심 좌표 계산
            x_centers = np.array([(b[0] + b[2]) / 2.0 for b in boxes], dtype=float)
            y_centers = np.array([(b[1] + b[3]) / 2.0 for b in boxes], dtype=float)

            # 상대 크기 정규화
            area_n = areas / float(img_w * img_h)  # [0,1] 대략

            # X, Y 모두 프레임 중앙에 가까울수록 값이 큰 척도 [0,1]
            x_center_closeness = 1.0 - np.abs(x_centers - (img_w / 2.0)) / (img_w / 2.0)
            y_center_closeness = 1.0 - np.abs(y_centers - (img_h / 2.0)) / (img_h / 2.0)

            # 두 축의 중앙 근접도를 평균 (또는 필요 시 가중평균)
            center_closeness = (x_center_closeness + y_center_closeness) / 2.0
            center_closeness = np.clip(center_closeness, 0.0, 1.0)

            # 최종 점수: 크기(가까움) + 중앙 근접도
            alpha, beta = 0.65, 0.35  # alpha: 크기 비중, beta: 중앙 비중
            scores = alpha * area_n + beta * center_closeness

            best_idx = int(np.argmax(scores))
            best_box = boxes[best_idx]
            best_conf = probs[best_idx]

            # 신뢰도 체크 (첫 프레임은 0.9 미달 시 0.7로 한 번 더 허용, prev_box는 쓰지 않음)
            if is_first_frame:
                # 첫 프레임은 prev_box 사용 금지
                if best_conf is None or best_conf < 0.9:
                    # "다시 detection"은 facenet_pytorch에선 결과가 동일할 수 있으므로
                    # 실질적으로 허용 임계만 0.7로 한 번 낮춰서 수용 여부 판단
                    if best_conf is not None and best_conf >= 0.7:
                        # accept as is (prev_box 사용 안 함)
                        use_prev = False
                    else:
                        print(f"[무시] 첫 프레임 신뢰도 낮음 ({best_conf if best_conf is not None else 'None'}): {img_full_path}")
                        continue
            else:
                # 첫 프레임이 아니면 원래 기준(0.9)로 판단, 필요시 prev_box 폴백 허용
                if best_conf is None or best_conf < min_conf:
                    if prev_box is not None:
                        use_prev = True
                        best_box = prev_box.copy()
                        best_conf = None
                    else:
                        print(f"[무시] 신뢰도 낮음 ({best_conf if best_conf is not None else 'None'}): {img_full_path}")
                        continue

        # === (3) bbox 확장 및 크롭 ===
        x1, y1, x2, y2 = [int(v) for v in best_box]
        w = x2 - x1
        h = y2 - y1
        xw1 = max(int(x1 - ad * w), 0)
        yw1 = max(int(y1 - ad * h), 0)
        xw2 = min(int(x2 + ad * w), img_w - 1)
        yw2 = min(int(y2 + ad * h), img_h - 1)

        # 비정상 bbox 방어
        if xw2 <= xw1 or yw2 <= yw1:
            # 폴백으로 온 경우라면 스킵, 아니면 prev_box 초기화 후 스킵
            if use_prev:
                print(f"[경고] 폴백 bbox가 유효하지 않습니다: {img_full_path}")
            else:
                prev_box = None
            continue

        cropped = img[yw1:yw2 + 1, xw1:xw2 + 1, :]
        # 저장 해상도는 기존과 동일하게 (224,224)
        cropped = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_NEAREST)

        # 저장 경로(원래 로직 유지)
        save_path_dir = f'/home/aivs/바탕화면/hdd/adj_test/dataset/PNU_selected/{day}/{num}/cropImage/{video_name}'
        os.makedirs(save_path_dir, exist_ok=True)
        save_path = f'{save_path_dir}/{img_name}'
        cv2.imwrite(save_path, cropped)

        # 성공적으로 저장했으면 prev_box 업데이트 (탐지 성공이면 현재, 폴백이면 그대로 유지)
        if not use_prev:
            prev_box = np.array(best_box, dtype=float)

        if isPlot:
            cv2.imshow("crop", cropped)
            cv2.waitKey(10)

    print(f"모든 크롭된 얼굴 이미지가 '{output_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
