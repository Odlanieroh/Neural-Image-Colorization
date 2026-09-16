import os
import queue
import threading
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image

import config
import inference
import train as train_module

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class ColorizerApp:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.busy = False
        self.q = queue.Queue()
        self._imgs = {}

        root.title("Chromatic Neural Restoration")
        root.geometry("920x700")
        root.minsize(840, 620)

        header = ctk.CTkFrame(root, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(22, 4))
        ctk.CTkLabel(header, text="Chromatic Neural Restoration",
                     font=ctk.CTkFont(size=26, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(header, text="Grayscale  ->  Color    |    ResNet50 U-Net    |    CIE Lab",
                     font=ctk.CTkFont(size=13), text_color="gray60").pack(anchor="w")

        controls = ctk.CTkFrame(root, corner_radius=12)
        controls.pack(fill="x", padx=24, pady=14)
        self.train_btn = ctk.CTkButton(controls, text="Train model",
                                        command=self.on_train, width=150, height=40)
        self.color_btn = ctk.CTkButton(controls, text="Open & colorize",
                                       command=self.on_colorize, width=150, height=40)
        self.train_btn.pack(side="left", padx=(16, 8), pady=16)
        self.color_btn.pack(side="left", padx=8, pady=16)
        self.progress = ctk.CTkProgressBar(controls, mode="indeterminate", width=170)
        self.progress.pack(side="right", padx=18)
        self.progress.set(0)

        cards = ctk.CTkFrame(root, fg_color="transparent")
        cards.pack(fill="both", expand=True, padx=24, pady=6)
        self.in_panel = self._make_card(cards, "Input (B&W)")
        self.out_panel = self._make_card(cards, "Colorized")
        self.in_panel["frame"].pack(side="left", expand=True, fill="both", padx=(0, 8))
        self.out_panel["frame"].pack(side="left", expand=True, fill="both", padx=(8, 0))

        ctk.CTkLabel(root, text="STATUS", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="gray60").pack(anchor="w", padx=27, pady=(6, 0))
        self.log = ctk.CTkTextbox(root, height=120, font=ctk.CTkFont(family="Consolas", size=12))
        self.log.pack(fill="x", padx=24, pady=(2, 20))
        self.log.configure(state="disabled")

        self._set_busy(True)
        self._log("Starting up - first run downloads the ResNet50 weights (~90 MB)...")
        threading.Thread(target=self._startup, daemon=True).start()
        self.root.after(100, self._poll)

    def _make_card(self, parent, title):
        frame = ctk.CTkFrame(parent, corner_radius=14)
        ctk.CTkLabel(frame, text=title,
                     font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(14, 8))
        img = ctk.CTkLabel(frame, text="(no image yet)", width=320, height=320,
                           fg_color="gray20", corner_radius=10, text_color="gray55")
        img.pack(padx=16, pady=(0, 16), expand=True)
        return {"frame": frame, "img": img}

    def _startup(self):
        try:
            if os.path.exists(config.WEIGHTS_PATH):
                self.q.put(("model", inference.load_model()))
                self.q.put(("log", "Trained model loaded - ready to colorize."))
            else:
                self.q.put(("log", "No trained model yet. Click 'Train model' to make one."))
        except Exception as e:
            self.q.put(("log", f"Startup note: {e}"))
        finally:
            self.q.put(("done", None))

    def _train_worker(self):
        try:
            self.q.put(("log", "Downloading dataset + training - this takes a while..."))
            model = train_module.train(progress_cb=lambda m: self.q.put(("log", m)))
            self.q.put(("model", model))
            self.q.put(("log", "Training finished. Weights saved to " + config.WEIGHTS_PATH))
        except Exception as e:
            self.q.put(("log", "ERROR during training: " + str(e)))
        finally:
            self.q.put(("done", None))

    def _colorize_worker(self, path):
        try:
            if self.model is None:
                self.model = inference.load_model()
            gray, color = inference.colorize_array(self.model, path)
            self.q.put(("images", gray, color))
            self.q.put(("log", "Colorized " + os.path.basename(path)))
        except Exception as e:
            self.q.put(("log", "ERROR: " + str(e)))
        finally:
            self.q.put(("done", None))

    def on_train(self):
        if self._busy_guard():
            return
        self._set_busy(True)
        threading.Thread(target=self._train_worker, daemon=True).start()

    def on_colorize(self):
        if self._busy_guard():
            return
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self._set_busy(True)
        threading.Thread(target=self._colorize_worker, args=(path,), daemon=True).start()

    def _busy_guard(self):
        if self.busy:
            self._log("Busy - wait for the current task to finish.")
            return True
        return False

    def _set_busy(self, b):
        self.busy = b
        state = "disabled" if b else "normal"
        for btn in (self.train_btn, self.color_btn):
            btn.configure(state=state)
        if b:
            self.progress.configure(mode="indeterminate")
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.set(0)

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _show(self, panel, arr, mode):
        pil = Image.fromarray(arr, mode).resize((320, 320))
        ctkimg = ctk.CTkImage(light_image=pil, dark_image=pil, size=(320, 320))
        panel["img"].configure(image=ctkimg, text="")
        self._imgs[id(panel)] = ctkimg

    def _poll(self):
        try:
            while True:
                kind, *payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload[0])
                elif kind == "model":
                    self.model = payload[0]
                elif kind == "images":
                    self._show(self.in_panel, payload[0], "L")
                    self._show(self.out_panel, payload[1], "RGB")
                elif kind == "done":
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)
