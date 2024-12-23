import sys
import cv2 as cv
import numpy as np


def adjust_color_channels(frame, blue=127, green=127, red=127):
    blue = int(blue - 127)
    green = int(green - 127)
    red = int(red - 127)

    b, g, r = cv.split(frame)

    b = cv.add(b, blue)
    g = cv.add(g, green)
    r = cv.add(r, red)

    b = np.clip(b, 0, 255)
    g = np.clip(g, 0, 255)
    r = np.clip(r, 0, 255)

    adjusted_frame = cv.merge([b, g, r])

    return adjusted_frame


def optimize_histogram(frame):
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        channels = cv.split(frame)
        eq_channels = [cv.equalizeHist(channel) for channel in channels]
        optimized_frame = cv.merge(eq_channels)
    else:
        optimized_frame = cv.equalizeHist(frame)

    return optimized_frame


def calculate_optimal_channel_value(channel):
    hist = cv.calcHist([channel], [0], None, [256], [0, 256])

    hist = hist / hist.sum()

    cumsum = np.cumsum(hist)

    optimal_value = np.argmax(cumsum > 0.5)
    return optimal_value


if __name__ == '__main__':
    video = cv.VideoCapture(0)
    video.set(cv.CAP_PROP_FPS, 15)
    if not video.isOpened():
        print("Не удалось открыть видео")
        sys.exit()

    cv.namedWindow('Filter')
    cv.createTrackbar('Blue', 'Filter', 127, 255, lambda x: None)
    cv.createTrackbar('Green', 'Filter', 127, 255, lambda x: None)
    cv.createTrackbar('Red', 'Filter', 127, 255, lambda x: None)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        blue = cv.getTrackbarPos('Blue', 'Filter')
        green = cv.getTrackbarPos('Green', 'Filter')
        red = cv.getTrackbarPos('Red', 'Filter')

        frame = adjust_color_channels(frame, blue, green, red)

        cv.imshow('Filter', frame)

        key = cv.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            optimized_frame = optimize_histogram(frame)

            b, g, r = cv.split(optimized_frame)
            optimal_blue = calculate_optimal_channel_value(b)
            optimal_green = calculate_optimal_channel_value(g)
            optimal_red = calculate_optimal_channel_value(r)

            cv.setTrackbarPos('Blue', 'Filter', optimal_blue)
            cv.setTrackbarPos('Green', 'Filter', optimal_green)
            cv.setTrackbarPos('Red', 'Filter', optimal_red)

    video.release()
    cv.destroyAllWindows()
