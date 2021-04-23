# Heart-rate-detection-using-Fingertip-video-Face-detection #
Django web based, Face detection using opencv and Heart rate detection using fingertip video

**Requirements :**

django <br>
python <br>
pip <br>
numpy <br>
cv2 <br>
matplotlib <br>
scipy <br>
maxflow <br>
imutils <br>

>Project demo : -

![demo](https://user-images.githubusercontent.com/49883798/115822402-1dc19000-a422-11eb-9ed5-e30b825d8739.gif)




```python
faceCascade = cv2.CascadeClassifier(CASCADE_PATH)

    colorSig = [] # Will store the average RGB color values in each frame's ROI
    heartRates = [] # Will store the heart rate calculated every 1 second
    previousFaceBox = None
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()
        if not ret:
            break

        previousFaceBox, roi = getBestROI(frame, faceCascade, previousFaceBox)

        if (roi is not None) and (np.size(roi) > 0):
            colorChannels = roi.reshape(-1, roi.shape[-1])
            avgColor = colorChannels.mean(axis=0)
            colorSig.append(avgColor)

        # Calculate heart rate every one second (once have 30-second of data)
        if (len(colorSig) >= WINDOW_SIZE) and (len(colorSig) % np.ceil(FPS) == 0):
            windowStart = len(colorSig) - WINDOW_SIZE
            window = colorSig[windowStart : windowStart + WINDOW_SIZE]
            lastHR = heartRates[-1] if len(heartRates) > 0 else None
            heartRates.append(getHeartRate(window, lastHR))
            # print("heartrates", heartRates)

        if np.ma.is_masked(roi):
            roi = np.where(roi.mask == True, 0, roi)
        cv2.imshow('Measuring...', roi)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break	

    print(heartRates)
```

>for more detail go through this pdf  :- [link](https://drive.google.com/file/d/1hy8q0nHDuTLIcw5mDD24FYOPcbvApdsj/view?usp=sharing "pdf link")
```

Thank you, if you are here!
