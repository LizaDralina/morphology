import cv2
from matplotlib import pyplot as plt

img = cv2.imread("stop.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

stop_data = cv2.CascadeClassifier('stop_data.xml')

found = stop_data.detectMultiScale(img_gray, minSize=(20, 20))

amount_found = len(found)

if amount_found != 0:
    for (x, y, width, height) in found:
        cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (0, 255, 0), 5)

    text = f"STOP signs found: {amount_found}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 4
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (img_rgb.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 40

    cv2.putText(img_rgb, text, (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)

plt.figure(figsize=(12, 10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()