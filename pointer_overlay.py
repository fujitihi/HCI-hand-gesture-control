import tkinter as tk
import json
import os
import ctypes
from ctypes import windll

top = tk.Tk()
top.overrideredirect(True)
top.attributes('-topmost', True)

def make_window_transparent_by_color(hwnd, color_key=(0, 0, 0)):
    r, g, b = color_key
    colorref = r | (g << 8) | (b << 16)
    style = windll.user32.GetWindowLongW(hwnd, -20)
    style |= 0x80000
    windll.user32.SetWindowLongW(hwnd, -20, style)
    windll.user32.SetLayeredWindowAttributes(hwnd, colorref, 255, 0x1)

top.update_idletasks()
hwnd = windll.user32.GetParent(top.winfo_id())
make_window_transparent_by_color(hwnd, color_key=(0, 0, 0))

top.geometry(f"{top.winfo_screenwidth()}x{top.winfo_screenheight()}+0+0")

top.config(bg='black')
canvas = tk.Canvas(top, bg='black', highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)

pointer_left = canvas.create_oval(0, 0, 0, 0, fill='blue', outline='')
pointer_right = canvas.create_oval(0, 0, 0, 0, fill='red', outline='')

sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
x_min = (sw - 1000) // 2
x_max = x_min + 1000
y_min = (sh - 350) // 2
y_max = y_min + 350

def update():
    try:
        if os.path.exists("keyboard_pointer.json"):
            with open("keyboard_pointer.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            if "Left" in data:
                x = int(data["Left"]["x"] * sw)
                y = int(data["Left"]["y"] * sh)
                x = max(x_min, min(x, x_max))
                y = max(y_min, min(y, y_max))
                canvas.coords(pointer_left, x-10, y-10, x+10, y+10)

            if "Right" in data:
                x = int(data["Right"]["x"] * sw)
                y = int(data["Right"]["y"] * sh)
                x = max(x_min, min(x, x_max))
                y = max(y_min, min(y, y_max))
                canvas.coords(pointer_right, x-10, y-10, x+10, y+10)

    except Exception as e:
        print(f"[Overlay Error] {e}")
    top.after(30, update)

update()
top.mainloop()
