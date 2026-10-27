from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from src.model import MODEL_NAMES
from src.training_manifest import DEFAULT_TRAINING_MANIFEST_NAME


class PreprocessedTrainingApp(tk.Tk):
    """Train from frozen NPZ tensors without opening source TIFF files."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Secretion Model - NPZ Training")
        self.geometry("850x720")
        self.minsize(720, 600)

        self.project_dir = Path(__file__).resolve().parent
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None

        default_manifest = (
            self.project_dir / "data" / "tensors_0825" / DEFAULT_TRAINING_MANIFEST_NAME
        )
        self.manifest = tk.StringVar(
            value=str(default_manifest) if default_manifest.is_file() else ""
        )
        self.output_dir = tk.StringVar()
        self.target = tk.StringVar(value="auto")
        self.model = tk.StringVar(value="resnet18")
        self.image_size = tk.StringVar(value="224")
        self.epochs = tk.StringVar(value="25")
        self.batch_size = tk.StringVar(value="32")
        self.learning_rate = tk.StringVar(value="0.0001")
        self.weight_decay = tk.StringVar(value="0.0001")
        self.workers = tk.StringVar(value="0")
        self.patience = tk.StringVar(value="5")
        self.preview_images = tk.StringVar(value="0")
        self.pretrained = tk.BooleanVar(value=False)
        self.resume = tk.BooleanVar(value=False)
        self.manifest_summary = tk.StringVar(value="Select a balanced-test training manifest")

        self._build_ui()
        if self.manifest.get():
            self.after(0, self._refresh_manifest_summary)
        self.after(100, self._drain_messages)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(9, weight=1)

        ttk.Label(outer, text="Independent NPZ Model Training", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 5)
        )
        ttk.Label(
            outer,
            text="This UI calls train_preprocessed.py. Source TIFF files are never opened during training.",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 12))

        self._path_row(outer, 2, "Preprocessed manifest", self.manifest, self._choose_manifest)
        self._path_row(outer, 3, "Training output folder", self.output_dir, self._choose_output)
        ttk.Label(outer, textvariable=self.manifest_summary, foreground="#3a5f8a").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(2, 10)
        )

        settings = ttk.LabelFrame(outer, text="Training settings", padding=12)
        settings.grid(row=5, column=0, columnspan=3, sticky="ew")
        for column in range(4):
            settings.columnconfigure(column, weight=1 if column in (1, 3) else 0)

        self._setting_combo(settings, 0, "Target", self.target, ("auto", "insulin", "glucagon"), 0)
        self._setting_combo(settings, 0, "Model", self.model, MODEL_NAMES, 2)
        self._setting_entry(settings, 1, "Image size", self.image_size, 0)
        self._setting_entry(settings, 1, "Epochs", self.epochs, 2)
        self._setting_entry(settings, 2, "Batch size", self.batch_size, 0)
        self._setting_entry(settings, 2, "Data workers", self.workers, 2)
        self._setting_entry(settings, 3, "Learning rate", self.learning_rate, 0)
        self._setting_entry(settings, 3, "Weight decay", self.weight_decay, 2)
        self._setting_entry(settings, 4, "Early-stop patience", self.patience, 0)
        self._setting_entry(settings, 4, "Grad-CAM previews", self.preview_images, 2)

        checks = ttk.Frame(settings)
        checks.grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Checkbutton(checks, text="Use ImageNet pretrained weights", variable=self.pretrained).pack(side="left", padx=(0, 18))
        ttk.Checkbutton(checks, text="Resume latest checkpoint", variable=self.resume).pack(side="left")

        controls = ttk.Frame(outer)
        controls.grid(row=6, column=0, columnspan=3, sticky="ew", pady=12)
        self.run_button = ttk.Button(controls, text="Start NPZ training", command=self._start)
        self.run_button.pack(side="left")
        ttk.Button(controls, text="Open output folder", command=self._open_output).pack(side="left", padx=8)
        self.progress = ttk.Progressbar(controls, mode="indeterminate")
        self.progress.pack(side="right", fill="x", expand=True, padx=(20, 0))

        ttk.Label(outer, text="Training log").grid(row=8, column=0, columnspan=3, sticky="w", pady=(4, 4))
        log_frame = ttk.Frame(outer)
        log_frame.grid(row=9, column=0, columnspan=3, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

    def _path_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse...", command=command).grid(row=row, column=2, padx=(8, 0), pady=4)

    def _setting_entry(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, column: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=variable, width=16).grid(row=row, column=column + 1, sticky="ew", padx=(0, 18), pady=4)

    def _setting_combo(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, values, column: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=18).grid(
            row=row, column=column + 1, sticky="ew", padx=(0, 18), pady=4
        )

    def _choose_manifest(self) -> None:
        path = filedialog.askopenfilename(
            title="Select preprocessed manifest",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.manifest.set(path)
        if not self.output_dir.get():
            self.output_dir.set(str(Path(path).parent / "training_run"))
        self._refresh_manifest_summary()

    def _refresh_manifest_summary(self) -> None:
        try:
            df = pd.read_csv(self.manifest.get())
            required = {"preprocessed_path", "label"}
            missing = sorted(required - set(df.columns))
            if missing:
                self.manifest_summary.set(f"Not trainable yet: missing columns {', '.join(missing)}")
                return
            missing_npz = sum(not Path(str(item)).is_file() for item in df["preprocessed_path"])
            splits = ", ".join(f"{key}={value}" for key, value in df.get("split", pd.Series(dtype=str)).value_counts().items())
            hormones = ", ".join(sorted(df["hormone"].dropna().astype(str).unique())) if "hormone" in df else "single target"
            self.manifest_summary.set(
                f"Rows: {len(df):,} | NPZ missing: {missing_npz:,} | Splits: {splits or 'missing'} | Targets: {hormones}"
            )
        except Exception as error:
            self.manifest_summary.set(f"Cannot read manifest: {error}")

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="Select training output folder")
        if path:
            self.output_dir.set(path)

    def _validated_command(self) -> list[str]:
        manifest = Path(self.manifest.get().strip())
        if not manifest.is_file():
            raise ValueError("Select an existing preprocessed_manifest.csv file.")
        output_text = self.output_dir.get().strip()
        if not output_text:
            raise ValueError("Select a training output folder.")
        df = pd.read_csv(manifest)
        required = {"preprocessed_path", "label", "split"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Manifest is not training-ready; missing: {', '.join(missing)}")
        expected_splits = {"train", "val", "test"}
        found_splits = set(df["split"].astype(str).str.lower())
        if not expected_splits.issubset(found_splits):
            raise ValueError("Manifest must contain train, val, and test splits.")
        if any(not Path(str(path)).is_file() for path in df["preprocessed_path"]):
            raise ValueError("One or more preprocessed NPZ paths do not exist.")
        if "hormone" in df.columns:
            hormones = set(df["hormone"].dropna().astype(str).str.lower())
            if len(hormones) > 1 and self.target.get() == "auto":
                raise ValueError("This manifest contains multiple hormones; select insulin or glucagon.")

        integers = {
            "Image size": self.image_size.get(),
            "Epochs": self.epochs.get(),
            "Batch size": self.batch_size.get(),
            "Data workers": self.workers.get(),
            "Early-stop patience": self.patience.get(),
            "Grad-CAM previews": self.preview_images.get(),
        }
        for name, raw in integers.items():
            try:
                value = int(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be an integer.") from error
            if value < 0 or (name in {"Image size", "Epochs", "Batch size"} and value == 0):
                raise ValueError(f"{name} has an invalid value.")
        for name, raw in {"Learning rate": self.learning_rate.get(), "Weight decay": self.weight_decay.get()}.items():
            try:
                value = float(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be a number.") from error
            if value < 0 or (name == "Learning rate" and value == 0):
                raise ValueError(f"{name} has an invalid value.")

        command = [
            sys.executable,
            "-u",
            str(self.project_dir / "train_preprocessed.py"),
            "--manifest", str(manifest.resolve()),
            "--output-dir", str(Path(output_text).resolve()),
            "--model", self.model.get(),
            "--image-size", self.image_size.get(),
            "--epochs", self.epochs.get(),
            "--batch-size", self.batch_size.get(),
            "--lr", self.learning_rate.get(),
            "--weight-decay", self.weight_decay.get(),
            "--num-workers", self.workers.get(),
            "--early-stopping-patience", self.patience.get(),
            "--preview-images", self.preview_images.get(),
        ]
        if self.target.get() != "auto":
            command.extend(["--target", self.target.get()])
        if self.pretrained.get():
            command.append("--pretrained")
        if self.resume.get():
            command.append("--resume")
        return command

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            command = self._validated_command()
        except Exception as error:
            messagebox.showerror("Cannot start training", str(error), parent=self)
            return
        self._append_log("Starting independent NPZ training...\n")
        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self.worker = threading.Thread(target=self._run_process, args=(command,), daemon=True)
        self.worker.start()

    def _run_process(self, command: list[str]) -> None:
        try:
            process = subprocess.Popen(
                command,
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                self.messages.put(("log", line))
            self.messages.put(("done", process.wait()))
        except Exception as error:
            self.messages.put(("error", str(error)))

    def _drain_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    self._finish(int(payload))
                elif kind == "error":
                    self._finish(-1, str(payload))
        except queue.Empty:
            pass
        self.after(100, self._drain_messages)

    def _finish(self, return_code: int, error: str | None = None) -> None:
        self.progress.stop()
        self.run_button.configure(state="normal")
        if return_code == 0:
            self._append_log("\nTraining completed successfully.\n")
            messagebox.showinfo("Training complete", "Training completed successfully.", parent=self)
        else:
            detail = error or f"Training exited with code {return_code}. Check the log."
            self._append_log(f"\nERROR: {detail}\n")
            messagebox.showerror("Training failed", detail, parent=self)

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


if __name__ == "__main__":
    PreprocessedTrainingApp().mainloop()
