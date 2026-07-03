#!/usr/bin/env python3
"""
Drawing Comparison — GUI

Pick an OLD and a NEW drawing, click Compare, and it generates a NEW image that
highlights the differences (green = added in NEW, red = removed from OLD, orange
boxes = change regions). The two source drawings are never modified.

Engine: modules/diff_blockmatch.py (translation-tolerant; ignores parts/text that
merely moved, flags genuine add/remove/content changes).

Run: double-click "Compare Drawings.bat", or  py -3.13 gui_app.py
"""
import os
import sys
import threading
import queue
import traceback

# --- make the bundled .venv packages importable regardless of interpreter -----
_HERE = os.path.dirname(os.path.abspath(__file__))
_SP = os.path.join(_HERE, ".venv", "Lib", "site-packages")
if os.path.isdir(_SP) and _SP not in sys.path:
    sys.path.insert(0, _SP)
sys.path.insert(0, _HERE)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from modules import diff_blockmatch as bm

IMG_TYPES = [("Drawings", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")]
PREVIEW_MAX = (1000, 640)
SIMILARITY_WARN = 0.40   # below this, warn that the two images may be unrelated


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.old_path = tk.StringVar()
        self.new_path = tk.StringVar()
        self.out_path = tk.StringVar()
        self.status = tk.StringVar(value="Pick an OLD and a NEW drawing.")
        self.q: "queue.Queue" = queue.Queue()
        self._preview_img = None      # keep a ref so Tk doesn't GC it
        self._result = None
        self._anim = 0
        self._anim_base = ""
        self._job = None

        root.title("Drawing Comparison")
        root.geometry("1120x820")
        root.minsize(900, 640)

        self._build()

    # ---------------------------------------------------------------- UI ----
    def _build(self):
        pad = dict(padx=8, pady=6)
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="OLD drawing:", width=13).grid(row=0, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.old_path).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="Browse…", command=self._pick_old).grid(row=0, column=2)

        ttk.Label(top, text="NEW drawing:", width=13).grid(row=1, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.new_path).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="Browse…", command=self._pick_new).grid(row=1, column=2)

        ttk.Label(top, text="Save diff to:", width=13).grid(row=2, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.out_path).grid(row=2, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="Save as…", command=self._pick_out).grid(row=2, column=2)
        top.columnconfigure(1, weight=1)

        # options
        opt = ttk.LabelFrame(self.root, text="Options")
        opt.pack(fill="x", **pad)
        self.v_search = tk.IntVar(value=640)
        self.v_tol = tk.IntVar(value=3)
        self.v_minarea = tk.IntVar(value=60)
        self.v_leaders = tk.BooleanVar(value=True)
        self.v_mode = tk.StringVar(value="classic")
        self.v_color = tk.BooleanVar(value=False)

        # row 0: drawing type / mode
        ttk.Label(opt, text="Drawing type:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Radiobutton(opt, text="B&W line art (classic)", variable=self.v_mode,
                        value="classic", command=self._sync_mode).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(opt, text="Any / colored (general)", variable=self.v_mode,
                        value="general", command=self._sync_mode).grid(row=0, column=2, sticky="w")
        self.chk_color = ttk.Checkbutton(opt, text="Also flag color changes (magenta)",
                                         variable=self.v_color, state="disabled")
        self.chk_color.grid(row=0, column=3, columnspan=2, sticky="w", padx=10)

        # row 1: numeric tuning
        ttk.Label(opt, text="Max move treated as 'same' (px):").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ttk.Spinbox(opt, from_=120, to=2000, increment=40, width=7, textvariable=self.v_search).grid(row=1, column=1, sticky="w")
        ttk.Label(opt, text="Tolerance (px):").grid(row=1, column=2, sticky="e", padx=4)
        ttk.Spinbox(opt, from_=1, to=12, width=5, textvariable=self.v_tol).grid(row=1, column=3, sticky="w")
        ttk.Label(opt, text="Min change size:").grid(row=1, column=4, sticky="e", padx=4)
        ttk.Spinbox(opt, from_=10, to=500, increment=10, width=6, textvariable=self.v_minarea).grid(row=1, column=5, sticky="w")
        ttk.Checkbutton(opt, text="Suppress balloon leader lines", variable=self.v_leaders).grid(row=1, column=6, padx=10)

        # actions
        act = ttk.Frame(self.root)
        act.pack(fill="x", **pad)
        self.btn = ttk.Button(act, text="Compare", command=self._run)
        self.btn.pack(side="left")
        self.btn_open = ttk.Button(act, text="Open full-size result", command=self._open_full, state="disabled")
        self.btn_open.pack(side="left", padx=8)
        self.bar = ttk.Progressbar(act, mode="indeterminate", length=180)
        self.bar.pack(side="left", padx=8)
        ttk.Label(act, textvariable=self.status).pack(side="left", padx=8)

        # preview
        prev = ttk.LabelFrame(self.root, text="Result preview  (green = added in NEW · red = removed from OLD · magenta = color change)")
        prev.pack(fill="both", expand=True, **pad)
        self.canvas = tk.Label(prev, background="#2b2b2b")
        self.canvas.pack(fill="both", expand=True)

    def _sync_mode(self):
        # color-change toggle only applies in general mode
        self.chk_color.config(state="normal" if self.v_mode.get() == "general" else "disabled")

    # ------------------------------------------------------------- pickers --
    def _pick_old(self):
        p = filedialog.askopenfilename(title="Select OLD drawing", filetypes=IMG_TYPES)
        if p:
            self.old_path.set(p)
            self._suggest_out()

    def _pick_new(self):
        p = filedialog.askopenfilename(title="Select NEW drawing", filetypes=IMG_TYPES)
        if p:
            self.new_path.set(p)
            self._suggest_out()

    def _pick_out(self):
        p = filedialog.asksaveasfilename(title="Save diff image as", defaultextension=".png",
                                         filetypes=[("PNG image", "*.png")])
        if p:
            self.out_path.set(p)

    def _suggest_out(self):
        o, n = self.old_path.get(), self.new_path.get()
        if o and n and not self.out_path.get():
            folder = os.path.join(_HERE, "diff results")
            os.makedirs(folder, exist_ok=True)
            stem = f"{os.path.splitext(os.path.basename(o))[0]}__vs__{os.path.splitext(os.path.basename(n))[0]}"
            self.out_path.set(os.path.join(folder, stem + "_DIFF.png"))

    # ----------------------------------------------------------------- run --
    def _run(self):
        o, n, out = self.old_path.get(), self.new_path.get(), self.out_path.get()
        if not (o and n):
            messagebox.showwarning("Missing files", "Please choose both an OLD and a NEW drawing.")
            return
        if not os.path.exists(o) or not os.path.exists(n):
            messagebox.showerror("Not found", "One of the selected files does not exist.")
            return
        if not out:
            self._suggest_out(); out = self.out_path.get()

        params = dict(search=self.v_search.get(), tol=self.v_tol.get(),
                      min_area=self.v_minarea.get(), suppress_leaders=self.v_leaders.get(),
                      mode=self.v_mode.get(), detect_color=self.v_color.get())
        self._job = dict(o=o, n=n, out=out, params=params)

        # Phase 1: fast "are these the same drawing?" pre-check (loads at 1/8 res)
        self._begin_busy("Checking whether these are the same drawing")
        threading.Thread(target=self._sim_worker, args=(o, n), daemon=True).start()
        self.root.after(120, self._poll_sim)

    # busy-state / animated status ------------------------------------------
    def _begin_busy(self, base_msg):
        self._anim_base = base_msg
        self.btn.config(state="disabled")
        self.btn_open.config(state="disabled")
        self.bar.start(12)
        self._anim = 0
        self._animate()

    def _end_busy(self):
        self.bar.stop()
        self.btn.config(state="normal")

    def _animate(self):
        if str(self.btn["state"]) == "disabled":
            self._anim = (self._anim + 1) % 4
            self.status.set(self._anim_base + "." * self._anim)
            self.root.after(400, self._animate)

    # phase 1: similarity ----------------------------------------------------
    def _sim_worker(self, o, n):
        try:
            self.q.put(("sim", bm.similarity(o, n), None))
        except Exception:
            self.q.put(("err", traceback.format_exc(), None))

    def _poll_sim(self):
        try:
            kind, payload, _ = self.q.get_nowait()
        except queue.Empty:
            self.root.after(120, self._poll_sim)
            return
        if kind == "err":
            self._end_busy()
            self.status.set("Error — see dialog.")
            messagebox.showerror("Comparison failed", payload)
            return
        score = payload
        if score < SIMILARITY_WARN:
            self.bar.stop()
            proceed = messagebox.askyesno(
                "Drawings look different",
                f"These two images look largely different (about {score * 100:.0f}% similar).\n\n"
                "They may not be two revisions of the same drawing. Comparing anyway can take "
                "several minutes and may highlight almost everything.\n\nCompare anyway?")
            if not proceed:
                self._end_busy()
                self.status.set(f"Cancelled — only {score * 100:.0f}% similar. "
                                "Pick two revisions of the same drawing.")
                return
            self.bar.start(12)
        self._start_compare(score)

    # phase 2: full compare --------------------------------------------------
    def _start_compare(self, score):
        j = self._job
        self._anim_base = "Comparing — large drawings take ~1–2 min"
        threading.Thread(target=self._worker,
                         args=(j["o"], j["n"], j["out"], j["params"], score), daemon=True).start()
        self.root.after(120, self._poll)

    def _worker(self, o, n, out, params, score):
        try:
            res = bm.compare(o, n, out_path=out, **params)
            res["similarity"] = score
            prev = self._make_preview(out)
            self.q.put(("ok", res, prev))
        except Exception:
            self.q.put(("err", traceback.format_exc(), None))

    def _poll(self):
        try:
            kind, payload, prev = self.q.get_nowait()
        except queue.Empty:
            self.root.after(120, self._poll)
            return
        self._end_busy()
        if kind == "err":
            self.status.set("Error — see dialog.")
            messagebox.showerror("Comparison failed", payload)
            return
        self._result = payload
        self.btn_open.config(state="normal")
        color_txt = (f" · {payload['color_regions']} color regions"
                     if payload.get("color_regions") else "")
        self.status.set(f"Done · {payload['change_regions']} change regions{color_txt} · "
                        f"added {payload['added_px']:,}px · removed {payload['removed_px']:,}px · "
                        f"{payload.get('similarity', 0) * 100:.0f}% similar · "
                        f"saved {os.path.basename(payload['out_path'])}")
        if prev is not None:
            self._show_preview(prev)

    # ------------------------------------------------------------- preview --
    def _make_preview(self, out_path):
        im = cv2.imread(out_path)
        if im is None:
            return None
        h, w = im.shape[:2]
        mw, mh = PREVIEW_MAX
        s = min(mw / w, mh / h, 1.0)
        small = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    def _show_preview(self, rgb):
        img = Image.fromarray(rgb)
        self._preview_img = ImageTk.PhotoImage(img)
        self.canvas.config(image=self._preview_img)

    def _open_full(self):
        if self._result:
            try:
                os.startfile(os.path.abspath(self._result["out_path"]))
            except Exception as e:
                messagebox.showinfo("Open", f"Saved at:\n{self._result['out_path']}\n\n({e})")


def _selftest(result_file: str) -> int:
    """Exercise the bundled stack (cv2/numpy/PIL/tkinter/engine) in a frozen exe.
    Writes OK or a traceback to result_file (windowed exe has no visible console)."""
    import tempfile
    import numpy as np
    try:
        d = tempfile.mkdtemp()
        old = np.full((200, 300), 255, np.uint8)
        cv2.rectangle(old, (30, 30), (150, 150), 0, 2)
        new = old.copy()
        cv2.line(new, (180, 40), (260, 160), 0, 3)   # one added line
        op, npth, outp = (os.path.join(d, x) for x in ("o.png", "n.png", "out.png"))
        cv2.imwrite(op, old); cv2.imwrite(npth, new)
        sim = bm.similarity(op, npth)          # exercise the pre-check
        res = bm.compare(op, npth, out_path=outp)                       # classic
        # exercise the general/color path too (color images)
        oc = cv2.cvtColor(old, cv2.COLOR_GRAY2BGR); nc = cv2.cvtColor(new, cv2.COLOR_GRAY2BGR)
        opc, npc, outg = (os.path.join(d, x) for x in ("oc.png", "nc.png", "outg.png"))
        cv2.imwrite(opc, oc); cv2.imwrite(npc, nc)
        resg = bm.compare(opc, npc, out_path=outg, mode="general", detect_color=True)
        r = tk.Tk(); r.withdraw()
        _ = ImageTk.PhotoImage(Image.open(outp))
        r.destroy()
        ok = (os.path.exists(outp) and os.path.exists(outg)
              and res["added_px"] > 0 and 0.0 <= sim <= 1.0)
        with open(result_file, "w") as f:
            f.write(f"OK classic_add={res['added_px']} general_add={resg['added_px']} sim={sim:.2f}"
                    if ok else "FAIL")
        return 0 if ok else 1
    except Exception:
        with open(result_file, "w") as f:
            f.write(traceback.format_exc())
        return 1


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        sys.exit(_selftest(sys.argv[2]))
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
