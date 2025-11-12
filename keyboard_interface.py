# -*- coding: utf-8 -*-
import os
import tkinter as tk
import json

if not os.path.exists("keyboard_pointer.json"):
    with open("keyboard_pointer.json", "w", encoding="utf-8") as f:
        json.dump({}, f)

root = tk.Tk()
root.title("virtual keyboard")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

keyboard_width = screen_width // 2
keyboard_height = screen_height // 3

#x_pos = (screen_width - keyboard_width) // 2
#y_pos = screen_height - keyboard_height

x = (screen_width - 1000) // 2
y = (screen_height - 350) // 2

#root.geometry(f"{keyboard_width}x{keyboard_height}+{x_pos}+{y_pos}")
root.geometry(f"1000x350+{x}+{y}")

is_shift = False
is_caps = False
is_shortcut_mode = False
input_buffer = ""

shift_map = {
    '1': '!', '2': '@', '3': '#', '4': '$',
    '5': '%', '6': '^', '7': '&', '8': '*',
    '9': '(', '0': ')', '-': '_', '=': '+',
    '[': '{', ']': '}', '\\': '|', ';': ':',
    "'": '"', ',': '<', '.': '>', '/': '?',
}

def key_press(k):
    global is_shift, is_caps, is_shortcut_mode, shortcut_keys, input_buffer

    if k == "Shortcut":
        if is_shortcut_mode:
            write_shortcut_to_json(shortcut_keys)
            toggle_shortcut_mode()
        else:
            toggle_shortcut_mode()
        return

    if is_shortcut_mode:
        if k not in shortcut_keys:
            shortcut_keys.append(k)
        shortcut_label.config(text=" + ".join(shortcut_keys))
        return

    if k == "Shift":
        toggle_shift()
        return
    if k == "Caps":
        toggle_caps()
        return

    if k == "Backspace":
        input_buffer = input_buffer[:-1]
    elif k == "Space":
        input_buffer += " "
    elif k == "Tab":
        input_buffer += "\t"
    elif k == "Enter":
        write_text_to_json(input_buffer)
        input_buffer = ""
    else:
        char = shift_map[k] if is_shift and k in shift_map else k
        if len(char) == 1 and char.isalpha():
            if is_caps ^ is_shift:
                char = char.upper()
            else:
                char = char.lower()
        input_buffer += char

    if is_shift:
        is_shift = False
        shift_button.config(relief="raised")
        update_key_labels()

    buffer_label.config(text=input_buffer)

def toggle_shift():
    global is_shift
    is_shift = not is_shift
    shift_button.config(relief="sunken" if is_shift else "raised")
    update_key_labels()

def toggle_caps():
    global is_caps, is_shift
    if is_shift:
        is_caps = not is_caps
        caps_button.config(relief="sunken" if is_caps else "raised")
    is_shift = False
    shift_button.config(relief="raised")
    update_key_labels()

def update_key_labels():
    for key, btn in button_dict.items():
        label = shift_map.get(key, key) if is_shift else key
        if len(label) == 1 and label.isalpha():
            if is_caps ^ is_shift:
                label = label.upper()
            else:
                label = label.lower()
        btn.config(text=label)

def toggle_shortcut_mode():
    global is_shortcut_mode, shortcut_keys
    is_shortcut_mode = not is_shortcut_mode
    shortcut_keys = []
    shortcut_button.config(relief="sunken" if is_shortcut_mode else "raised")
    shortcut_label.config(text="")

def write_text_to_json(text):
    data = {"type": "text", "value": text}
    with open("keyboard_event.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

def write_shortcut_to_json(keys):
    data = {"type": "shortcut", "keys": [k.lower() for k in keys]}
    with open("keyboard_event.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

keyboard_layout = [
    ['Esc', '1','2','3','4','5','6','7','8','9','0','-','=', 'Backspace'],
    ['Tab', 'Q','W','E','R','T','Y','U','I','O','P','[',']','\\'],
    ['Caps', 'A','S','D','F','G','H','J','K','L',';','"', 'Enter'],
    ['Shift', 'Z','X','C','V','B','N','M',',','.','/','Shift'],
    ['Ctrl', 'Alt', 'Space', 'Alt', 'Ctrl', 'Shortcut']
]

width_map = {
    'Backspace': 10, 'Tab': 7, 'Caps': 8, 'Enter': 10, 'Shift': 10,
    'Ctrl': 7, 'Alt': 7, 'Space': 40, 'Esc': 5, 'Shortcut': 10
}

button_dict = {}
shift_button = None
caps_button = None
shortcut_button = None
shortcut_keys = []
shortcut_label = tk.Label(root, text="", font=("Consolas", 12), fg="blue")
shortcut_label.pack()

buffer_label = tk.Label(root, text="", font=("Consolas", 12), fg="green")
buffer_label.pack()

for row_keys in keyboard_layout:
    row_frame = tk.Frame(root)
    row_frame.pack(pady=1)
    for key in row_keys:
        w = width_map.get(key, 5)
        btn = tk.Button(row_frame, text=key, width=w, height=2,
                        font=("Consolas", 12),
                        command=lambda k=key: key_press(k))
        btn.pack(side=tk.LEFT, padx=1)
        if key == 'Shift' and not shift_button:
            shift_button = btn
        elif key == 'Caps':
            caps_button = btn
        elif key == "Shortcut":
            shortcut_button = btn
        else:
            button_dict[key] = btn

root.mainloop()
