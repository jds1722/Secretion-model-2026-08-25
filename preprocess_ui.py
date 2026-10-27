from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
from skimage import morphology

from src.config import LAYERS
from src.preprocess_v2 import (
    ObjectPreprocessConfig,
    build_model_input_v2,
    identify_round_objects,
)
from src.tiff_io import read_multilayer_tiff, robust_normalize


class PreprocessApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Secretion Model - TIFF Pre-processing")
        self.geometry("900x960")
        self.minsize(760, 720)

        self.project_dir = Path(__file__).resolve().parent
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None

        self.manifest = tk.StringVar()
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.preview_tiff = tk.StringVar()
        self.roi_size = tk.StringVar(value="128")
        self.bead_min = tk.StringVar(value="5")
        self.bead_max = tk.StringVar(value="20")
        self.bead_dilation = tk.StringVar(value="2")
        self.nucleus_min = tk.StringVar(value="10")
        self.nucleus_max = tk.StringVar(value="100")
        self.cell_opening = tk.StringVar(value="4")
        self.overwrite = tk.BooleanVar(value=False)
        self.fail_on_qc = tk.BooleanVar(value=False)

        self._build_ui()
        self.after(100, self._drain_messages)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(12, weight=1)

        ttk.Label(outer, text="Independent TIFF Pre-processing", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 14)
        )
        ttk.Label(
            outer,
            text=(
                "Each run is launched as an independent background job. You may close "
                "this window and launch other folders concurrently."
            ),
            foreground="#3a5f8a",
        ).grid(row=0, column=1, columnspan=2, sticky="e", pady=(0, 14))

        self._path_row(outer, 1, "Input TIFF folder", self.input_dir, self._choose_input)
        self._path_row(outer, 2, "Or source manifest CSV", self.manifest, self._choose_manifest)
        self._path_row(outer, 3, "Output folder", self.output_dir, self._choose_output)

        settings = ttk.LabelFrame(outer, text="Object detection settings", padding=12)
        settings.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(14, 8))
        for column in range(4):
            settings.columnconfigure(column, weight=1 if column in (1, 3) else 0)

        fields = [
            ("ROI size", self.roi_size, "Bead dilation (px)", self.bead_dilation),
            ("Bead min diameter", self.bead_min, "Bead max diameter", self.bead_max),
            ("Nucleus min diameter", self.nucleus_min, "Nucleus max diameter", self.nucleus_max),
            ("Cell opening (px)", self.cell_opening, None, None),
        ]
        for row, (left_label, left_var, right_label, right_var) in enumerate(fields):
            ttk.Label(settings, text=left_label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            ttk.Entry(settings, textvariable=left_var, width=14).grid(row=row, column=1, sticky="ew", padx=(0, 20), pady=4)
            if right_label and right_var:
                ttk.Label(settings, text=right_label).grid(row=row, column=2, sticky="w", padx=(0, 8), pady=4)
                ttk.Entry(settings, textvariable=right_var, width=14).grid(row=row, column=3, sticky="ew", pady=4)

        options = ttk.Frame(outer)
        options.grid(row=5, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Checkbutton(options, text="Overwrite existing tensors", variable=self.overwrite).pack(side="left", padx=(0, 18))
        ttk.Checkbutton(options, text="Exclude samples that fail nucleus QC", variable=self.fail_on_qc).pack(side="left")

        buttons = ttk.Frame(outer)
        buttons.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 8))
        self.run_button = ttk.Button(buttons, text="Launch background job", command=self._start)
        self.run_button.pack(side="left")
        ttk.Button(buttons, text="Open progress terminal", command=self._open_monitor).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(buttons, text="Open output folder", command=self._open_output).pack(side="left", padx=8)
        self.progress = ttk.Progressbar(buttons, mode="indeterminate")
        self.progress.pack(side="right", fill="x", expand=True, padx=(20, 0))

        self._path_row(outer, 7, "Preview TIFF", self.preview_tiff, self._choose_preview)
        ttk.Button(outer, text="Generate before / after preview", command=self._start_preview).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(4, 8)
        )
        preview_frame = ttk.LabelFrame(outer, text="Preview", padding=8)
        preview_frame.grid(row=9, column=0, columnspan=3, sticky="ew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        ttk.Label(preview_frame, text="Before: raw composite + INS(red) / GCG(cyan) outlines").grid(row=0, column=0, pady=(0, 5))
        ttk.Label(preview_frame, text="After: model input tensor").grid(row=0, column=1, pady=(0, 5))
        self.before_preview = ttk.Label(preview_frame, text="Select a representative TIFF", anchor="center")
        self.after_preview = ttk.Label(preview_frame, text="Preview not generated", anchor="center")
        self.before_preview.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.after_preview.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        self.preview_status = ttk.Label(outer, text="")
        self.preview_status.grid(row=10, column=0, columnspan=3, sticky="w", pady=(4, 0))

        ttk.Label(outer, text="Launcher log").grid(row=11, column=0, columnspan=3, sticky="w", pady=(8, 4))
        log_frame = ttk.Frame(outer)
        log_frame.grid(row=12, column=0, columnspan=3, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=14, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _path_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse...", command=command).grid(row=row, column=2, padx=(8, 0), pady=4)

    def _choose_manifest(self) -> None:
        path = filedialog.askopenfilename(title="Select source manifest", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.manifest.set(path)
            self.input_dir.set("")
            if not self.output_dir.get():
                self.output_dir.set(str(Path(path).parent / "preprocessed_v2"))

    def _choose_input(self) -> None:
        path = filedialog.askdirectory(title="Select folder containing multi-layer TIFF files")
        if path:
            self.input_dir.set(path)
            self.manifest.set("")
            if not self.output_dir.get():
                self.output_dir.set(str(Path(path).parent / f"{Path(path).name}_preprocessed_v2"))

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir.set(path)

    def _choose_preview(self) -> None:
        path = filedialog.askopenfilename(
            title="Select representative multi-layer TIFF",
            filetypes=[("TIFF images", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.preview_tiff.set(path)

    def _numeric_values(self) -> dict[str, int | float]:
        numeric = {
            "ROI size": (self.roi_size.get(), int),
            "Bead minimum diameter": (self.bead_min.get(), float),
            "Bead maximum diameter": (self.bead_max.get(), float),
            "Bead dilation": (self.bead_dilation.get(), int),
            "Nucleus minimum diameter": (self.nucleus_min.get(), float),
            "Nucleus maximum diameter": (self.nucleus_max.get(), float),
            "Cell opening": (self.cell_opening.get(), int),
        }
        values: dict[str, int | float] = {}
        for name, (raw, converter) in numeric.items():
            try:
                value = converter(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be a number.") from error
            if value < 0 or (name == "ROI size" and value == 0):
                raise ValueError(f"{name} must be positive.")
            values[name] = value
        if values["Bead minimum diameter"] > values["Bead maximum diameter"]:
            raise ValueError("Bead minimum diameter cannot exceed the maximum.")
        if values["Nucleus minimum diameter"] > values["Nucleus maximum diameter"]:
            raise ValueError("Nucleus minimum diameter cannot exceed the maximum.")
        return values

    def _validated_command(self) -> list[str]:
        output = self.output_dir.get().strip()
        if not output:
            raise ValueError("Select an output folder.")

        manifest_text = self.manifest.get().strip()
        input_text = self.input_dir.get().strip()
        if manifest_text:
            manifest = Path(manifest_text)
            if not manifest.is_file():
                raise ValueError("Select an existing source manifest CSV.")
            source_args = ["--manifest", str(manifest.resolve())]
        elif input_text:
            input_dir = Path(input_text)
            if not input_dir.is_dir():
                raise ValueError("Select an existing input TIFF folder.")
            source_args = ["--input-dir", str(input_dir.resolve())]
        else:
            raise ValueError("Select an input TIFF folder or a source manifest CSV.")

        self._numeric_values()

        python_executable = Path(sys.executable)
        if python_executable.name.lower() == "pythonw.exe":
            console_python = python_executable.with_name("python.exe")
            if console_python.is_file():
                python_executable = console_python
        command = [
            str(python_executable),
            "-u",
            str(self.project_dir / "preprocess_dataset.py"),
            *source_args,
            "--output-dir", str(Path(output).resolve()),
            "--roi-size", self.roi_size.get(),
            "--bead-min-diameter", self.bead_min.get(),
            "--bead-max-diameter", self.bead_max.get(),
            "--bead-dilation-px", self.bead_dilation.get(),
            "--nucleus-min-diameter", self.nucleus_min.get(),
            "--nucleus-max-diameter", self.nucleus_max.get(),
            "--cell-opening-px", self.cell_opening.get(),
        ]
        if self.overwrite.get():
            command.append("--overwrite")
        if self.fail_on_qc.get():
            command.append("--fail-on-qc")
        command.append("--background")
        return command

    def _start_preview(self) -> None:
        source = Path(self.preview_tiff.get().strip())
        if not source.is_file():
            messagebox.showerror("Preview", "Select an existing TIFF image.", parent=self)
            return
        try:
            values = self._numeric_values()
        except ValueError as error:
            messagebox.showerror("Invalid settings", str(error), parent=self)
            return
        self.preview_status.configure(text="Generating preview...")
        threading.Thread(
            target=self._generate_preview,
            args=(source, values),
            daemon=True,
        ).start()

    def _generate_preview(self, source: Path, values: dict[str, int | float]) -> None:
        try:
            stack = read_multilayer_tiff(source)
            needed = max(LAYERS.bf, LAYERS.nucleus, LAYERS.bead_ins, LAYERS.bead_gcg)
            if stack.shape[0] <= needed:
                raise ValueError(f"TIFF has {stack.shape[0]} layers; at least {needed + 1} are required.")
            raw = np.stack(
                [
                    robust_normalize(stack[LAYERS.bf]),
                    robust_normalize(stack[LAYERS.nucleus]),
                    np.maximum(
                        robust_normalize(stack[LAYERS.bead_ins]),
                        robust_normalize(stack[LAYERS.bead_gcg]),
                    ),
                ],
                axis=-1,
            )
            config = ObjectPreprocessConfig(
                bead_min_diameter=float(values["Bead minimum diameter"]),
                bead_max_diameter=float(values["Bead maximum diameter"]),
                bead_dilation_px=int(values["Bead dilation"]),
                nucleus_min_diameter=float(values["Nucleus minimum diameter"]),
                nucleus_max_diameter=float(values["Nucleus maximum diameter"]),
                cell_opening_px=int(values["Cell opening"]),
            )
            ins_mask, _ins_labels, _ins_count = identify_round_objects(
                stack[LAYERS.bead_ins],
                threshold_method="otsu",
                min_diameter=config.bead_min_diameter,
                max_diameter=config.bead_max_diameter,
                smoothing_sigma=0.0,
                min_peak_distance=config.bead_min_peak_distance,
                discard_border=False,
            )
            gcg_mask, _gcg_labels, _gcg_count = identify_round_objects(
                stack[LAYERS.bead_gcg],
                threshold_method="li",
                min_diameter=config.bead_min_diameter,
                max_diameter=config.bead_max_diameter,
                smoothing_sigma=0.0,
                min_peak_distance=config.bead_min_peak_distance,
                discard_border=False,
            )
            ins_outline = morphology.dilation(ins_mask, morphology.disk(1)) ^ morphology.erosion(ins_mask, morphology.disk(1))
            gcg_outline = morphology.dilation(gcg_mask, morphology.disk(1)) ^ morphology.erosion(gcg_mask, morphology.disk(1))
            raw[ins_outline] = (1.0, 0.1, 0.0)
            raw[gcg_outline] = (0.0, 1.0, 1.0)
            model_input, _masks, qc = build_model_input_v2(
                stack,
                object_config=config,
                roi_size=int(values["ROI size"]),
            )
            # Three-color preview of the four model channels: BF, nucleus
            # fluorescence, and the new nucleus-localized BF channel.
            after = np.stack(
                [model_input[0], model_input[1], model_input[3]], axis=-1
            )
            after = np.clip(after, 0.0, 1.0)
            self.messages.put(("preview_done", (raw, after, qc)))
        except Exception as error:
            self.messages.put(("preview_error", str(error)))

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            command = self._validated_command()
        except ValueError as error:
            messagebox.showerror("Invalid settings", str(error), parent=self)
            return
        self._append_log("Launching independent background preprocessing job...\n")
        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self.worker = threading.Thread(target=self._run_process, args=(command,), daemon=True)
        self.worker.start()

    def _run_process(self, command: list[str]) -> None:
        try:
            result = subprocess.run(
                command,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
            self.messages.put(("launch_done", (result.returncode, output)))
        except Exception as error:
            self.messages.put(("error", str(error)))

    def _drain_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "launch_done":
                    return_code, output = payload
                    self._finish_launch(int(return_code), str(output))
                elif kind == "error":
                    self._finish_launch(-1, str(payload))
                elif kind == "preview_done":
                    raw, after, qc = payload
                    self._show_previews(raw, after, qc)
                elif kind == "preview_error":
                    self.preview_status.configure(text="Preview failed")
                    messagebox.showerror("Preview failed", str(payload), parent=self)
        except queue.Empty:
            pass
        self.after(100, self._drain_messages)

    def _show_previews(self, raw: np.ndarray, after: np.ndarray, qc: dict[str, object]) -> None:
        def photo(array: np.ndarray) -> ImageTk.PhotoImage:
            image = Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255))
            image.thumbnail((380, 300), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image)

        self.before_photo = photo(raw)
        self.after_photo = photo(after)
        self.before_preview.configure(image=self.before_photo, text="")
        self.after_preview.configure(image=self.after_photo, text="")
        self.preview_status.configure(
            text=(
                f"ROI {qc.get('effective_roi_size', '?')} px | "
                f"INS objects {qc.get('ins_bead_count', '?')} | "
                f"GCG objects {qc.get('gcg_bead_count', '?')} | "
                f"Nucleus valid: {qc.get('nucleus_valid', '?')}"
            )
        )

    def _finish_launch(self, return_code: int, output: str = "") -> None:
        self.progress.stop()
        self.run_button.configure(state="normal")
        if output:
            self._append_log(output + "\n")
        if return_code == 0:
            self._append_log(
                "Background job launched. It will continue after this UI closes.\n"
            )
            messagebox.showinfo(
                "Background job launched",
                (
                    "Pre-processing is running independently. You can select another "
                    "input/output folder and launch another job. Use Open progress "
                    "terminal to monitor this output folder."
                ),
                parent=self,
            )
        else:
            detail = output or f"Launcher exited with code {return_code}."
            self._append_log(f"ERROR: {detail}\n")
            messagebox.showerror("Launch failed", detail, parent=self)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _open_output(self) -> None:
        path = Path(self.output_dir.get().strip())
        if not path.is_dir():
            messagebox.showerror("Output folder", "Select an existing output folder first.", parent=self)
            return
        subprocess.Popen(["explorer", str(path.resolve())])

    def _open_monitor(self) -> None:
        path = Path(self.output_dir.get().strip())
        if not path.is_dir():
            messagebox.showerror(
                "Progress monitor",
                "Select an existing preprocessing output folder first.",
                parent=self,
            )
            return
        monitor_python = Path(sys.executable)
        if monitor_python.name.lower() == "pythonw.exe":
            console_python = monitor_python.with_name("python.exe")
            if console_python.is_file():
                monitor_python = console_python
        command = [
            str(monitor_python),
            "-u",
            str(self.project_dir / "monitor_preprocessing.py"),
            "--output-dir",
            str(path.resolve()),
            "--watch",
            "--pause-on-exit",
        ]
        try:
            if sys.platform == "win32":
                subprocess.Popen(
                    command,
                    cwd=self.project_dir,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
            else:
                subprocess.Popen(command, cwd=self.project_dir, start_new_session=True)
        except OSError as error:
            messagebox.showerror("Progress monitor", str(error), parent=self)


if __name__ == "__main__":
    PreprocessApp().mainloop()
