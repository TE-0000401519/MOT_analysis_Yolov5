#仮想環境構築を確認するためのテスト用ファイル
import math

import torch
import cv2

# モデルの読み込み
model = torch.hub.load("yolov5", "custom", path="yolov5/runs/train/exp3/weights/best.pt", source="local")
#model = torch.hub.load("yolov5", "yolov5s", source="local")

# 入力画像の読み込み
img = cv2.imread("yolov5/data/test/images/7_DSC00533.JPG")
#img = cv2.imread("yolov5/data/images/bus.jpg")
# BGRではなくRGBでないとdetect.pyと同じ結果が出ず精度が落ちるのでRGB変換する
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# 検出の閾値設定
model.conf = 0.25

# 物体検出
# 検出したboxが複数ある場合、resultにはboxの座標データリストの複数リストが格納される
result = model(img)
print(result.pandas().xyxyn[0])

# バウンディングボックスを取得し画像をクリップ
for idx, row in enumerate(result.pandas().xyxyn[0].itertuples()):
    height, width = img.shape[:2]

    xmin = math.floor(width * row.xmin)
    xmax = math.floor(width * row.xmax)
    ymin = math.floor(height * row.ymin)
    ymax = math.floor(height * row.ymax)

    res_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    clipped_img = res_img[ymin:ymax, xmin:xmax]

    # f文字列 フォーマットを追加で指定できる
    cv2.imwrite(f"clipped_image/clip_{idx}.jpg", clipped_img)