# modules/GUI.py
import os
from tkinter import Tk, Label, Button, filedialog

def GUI_For_User():
    results = {"oldimage": None, "newimage": None}

    def pick_old():
        p = filedialog.askopenfilename(title="Select OLD image", filetypes=[("Images","*.tif;*.tiff;*.png;*.jpg;*.jpeg")])
        if p: results["oldimage"] = p; btn_old.config(text=os.path.basename(p))

    def pick_new():
        p = filedialog.askopenfilename(title="Select NEW image", filetypes=[("Images","*.tif;*.tiff;*.png;*.jpg;*.jpeg")])
        if p: results["newimage"] = p; btn_new.config(text=os.path.basename(p))

    def finish(): root.quit()

    root = Tk(); root.title("Pick images"); root.geometry("520x160"); root.resizable(False, False)
    Label(root, text="OLD image:").grid(row=0, column=0, padx=10, pady=8, sticky="e")
    btn_old = Button(root, text="Choose…", width=35, command=pick_old); btn_old.grid(row=0, column=1, pady=8)
    Label(root, text="NEW image:").grid(row=1, column=0, padx=10, pady=8, sticky="e")
    btn_new = Button(root, text="Choose…", width=35, command=pick_new); btn_new.grid(row=1, column=1, pady=8)
    Button(root, text="Run", width=18, command=finish).grid(row=3, column=0, columnspan=2, pady=12)
    root.mainloop(); root.destroy()
    return results["oldimage"], results["newimage"]
