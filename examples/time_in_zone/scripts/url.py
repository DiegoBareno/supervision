import yt_dlp
import cv2

url = "https://www.youtube.com/watch?v=Cp4RRAEgpeU"

# Extraer URL directa con yt-dlp
ydl_opts = {"format": "best"}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(url, download=False)
    video_url = info_dict["url"]

print("Enlace de streaming directo:", video_url)

# Usar OpenCV para abrir el video
cap = cv2.VideoCapture(video_url)
if not cap.isOpened():
    print("⚠️ No se pudo abrir el stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer el frame.")
        break

    cv2.imshow("YouTube Live en OpenCV", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
