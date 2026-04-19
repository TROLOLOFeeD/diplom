import cv2
import numpy as np
import mss
import win32gui
import win32process
import psutil

def get_dota2_hwnd():
    """Надежный поиск окна dota2.exe через PID"""
    target_pid = None
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] and proc.info['name'].lower() == 'dota2.exe':
            target_pid = proc.info['pid']
            break
            
    if not target_pid:
        return None

    hwnds = []
    def callback(hwnd, _):
        try:
            if win32gui.IsWindowVisible(hwnd):
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid == target_pid:
                    hwnds.append(hwnd)
        except Exception:
            pass
        return True

    try:
        win32gui.EnumWindows(callback, None)
    except Exception as e:
        print(f"⚠️ Ошибка перечисления окон: {e}")

    return hwnds[0] if hwnds else None

# 1. Находим окно
hwnd = get_dota2_hwnd()
if not hwnd:
    print("❌ Dota 2 не найдена. Убедитесь, что игра запущена в режиме 'Окно без рамки' (Borderless).")
    exit()

rect = win32gui.GetWindowRect(hwnd)
w, h = rect[2] - rect[0], rect[3] - rect[1]
print(f"✅ Окно найдено. Разрешение: {w}x{h}")

# 2. Скриншот окна
with mss.mss() as sct:
    frame = cv2.cvtColor(np.array(sct.grab({"top": rect[1], "left": rect[0], "width": w, "height": h})), cv2.COLOR_BGRA2BGR)

# 3. Создаём окно БЕЗ рамки
window_name = "Dota 2 Calibration"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # ✅ Позволяет менять размер и убирать рамку

cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 4. Калибровка по кликам
crop_w, crop_h = 60, 45  # Размер кропа (настройте под свои нужды)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cx, cy = x - crop_w//2, y - crop_h//2
        print(f"✅ Готовый кортеж: ({cx}, {cy}, {crop_w}, {crop_h})")
        
        clone = frame.copy()
        cv2.rectangle(clone, (cx, cy), (cx + crop_w, cy + crop_h), (0, 0, 255), 2)
        cv2.imshow(window_name, clone)

cv2.setMouseCallback(window_name, click_event)
cv2.imshow(window_name, frame)

print("🖱 Кликайте по центрам элементов. Горячие клавиши:")
print("   F11 — переключить полноэкранный режим")
print("   R — сбросить размер/позицию окна")
print("   Q — выход")

fullscreen = False
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 0x7A:  # F11
        fullscreen = not fullscreen
        mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, mode)
        print(f"🔲 Полноэкранный режим: {'ВКЛ' if fullscreen else 'ВЫКЛ'}")
    elif key == ord('r'):
        cv2.resizeWindow(window_name, min(w, 800), min(h, 600))
        cv2.moveWindow(window_name, screen_w - 820, 10)
        print("🔄 Окно возвращено в правый верхний угол")

cv2.destroyAllWindows()
