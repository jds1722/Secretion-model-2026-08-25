from __future__ import annotations

import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from augment_training_images import AugmentationOptions, augment_images, build_augmentations


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_DIR / "input_files"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "augmented_files"


class AugmentTrainingImagesUi(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Morphology-safe TIFF Augmenter")
        self.geometry("1080x760")
        self.minsize(920, 650)

        default_input = DEFAULT_INPUT_DIR if DEFAULT_INPUT_DIR.exists() else PROJECT_DIR
        self.input_path = tk.StringVar(value=str(default_input))
        self.mask_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        self.status = tk.StringVar(value="Ready")
        self.recursive = tk.BooleanVar(value=True)
        self.overwrite = tk.BooleanVar(value=False)
        self.quarter_turns = tk.BooleanVar(value=True)
        self.horizontal_flips = tk.BooleanVar(value=True)
        self.small_rotation = tk.BooleanVar(value=False)
        self.small_angle = tk.StringVar(value="5")
        self.translation = tk.BooleanVar(value=False)
        self.shift_pixels = tk.StringVar(value="10")
        self.fill_mode = tk.StringVar(value="Per-plane border median")
        self.fill_value = tk.StringVar(value="0")
        self.transform_count = tk.StringVar()
        self.log_queue: queue.Queue[object] = queue.Queue()
        self.worker: threading.Thread | None = None

        self._build_style()
        self._build_layout()
        for variable in (
            self.quarter_turns, self.horizontal_flips, self.small_rotation,
            self.small_angle, self.translation, self.shift_pixels, self.fill_mode,
        ):
            variable.trace_add("write", self._update_transform_count)
        self._update_transform_count()
        self.after(100, self._drain_log_queue)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.configure("Panel.TFrame", padding=12)
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Warning.TLabel", foreground="#9a5b00")
        style.configure("Status.TLabel", foreground="#315a85")
        style.configure("Run.TButton", padding=(12, 8))

    def _path_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, optional: bool = False) -> None:
        ttk.Label(parent, text=label, style="Title.TLabel").grid(row=row, column=0, sticky="w")
        holder = ttk.Frame(parent)
        holder.grid(row=row + 1, column=0, sticky="ew", pady=(6, 12))
        holder.columnconfigure(0, weight=1)
        ttk.Entry(holder, textvariable=variable).grid(row=0, column=0, sticky="ew")
        ttk.Button(holder, text="File", command=lambda: self._browse_file(variable)).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(holder, text="Folder", command=lambda: self._browse_folder(variable)).grid(row=0, column=2, padx=(6, 0))
        if optional:
            ttk.Button(holder, text="Clear", command=lambda: variable.set("")).grid(row=0, column=3, padx=(6, 0))

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0, minsize=430)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        left = ttk.Frame(self, style="Panel.TFrame")
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        right = ttk.Frame(self, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._path_row(left, 0, "Image input", self.input_path)
        self._path_row(left, 2, "Mask input (optional, mirrored paths)", self.mask_path, optional=True)

        ttk.Label(left, text="Output folder", style="Title.TLabel").grid(row=4, column=0, sticky="w")
        output_row = ttk.Frame(left)
        output_row.grid(row=5, column=0, sticky="ew", pady=(6, 12))
        output_row.columnconfigure(0, weight=1)
        ttk.Entry(output_row, textvariable=self.output_dir).grid(row=0, column=0, sticky="ew")
        ttk.Button(output_row, text="Browse", command=self._browse_output).grid(row=0, column=1, padx=(8, 0))

        ttk.Label(left, text="Transforms", style="Title.TLabel").grid(row=6, column=0, sticky="w")
        options = ttk.Frame(left)
        options.grid(row=7, column=0, sticky="ew", pady=(6, 8))
        options.columnconfigure(2, weight=1)
        ttk.Checkbutton(options, text="90° / 180° / 270° rotations", variable=self.quarter_turns).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(options, text="Horizontal flips", variable=self.horizontal_flips).grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(options, text="Small rotations ±", variable=self.small_rotation).grid(row=2, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.small_angle, width=7).grid(row=2, column=1, sticky="w")
        ttk.Label(options, text="degrees (nearest neighbour)").grid(row=2, column=2, sticky="w")
        ttk.Checkbutton(options, text="Integer translations", variable=self.translation).grid(row=3, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.shift_pixels, width=7).grid(row=3, column=1, sticky="w")
        ttk.Label(options, text="pixels in four directions").grid(row=3, column=2, sticky="w")
        ttk.Label(options, text="Empty-border fill").grid(row=4, column=0, sticky="w", pady=(4, 0))
        ttk.Combobox(
            options,
            textvariable=self.fill_mode,
            values=("Per-plane border median", "Constant value"),
            state="readonly",
            width=24,
        ).grid(row=4, column=1, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(options, text="Constant value (only for constant mode)").grid(row=5, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(options, textvariable=self.fill_value, width=7).grid(row=5, column=1, sticky="w", pady=(4, 0))

        ttk.Label(
            left,
            text="Per-plane border median is the default and selects an observed edge pixel value for each plane. Small-angle rotation can still duplicate/omit pixels and make stepped boundaries.",
            style="Warning.TLabel",
            wraplength=400,
        ).grid(row=8, column=0, sticky="ew", pady=(0, 8))
        general = ttk.Frame(left)
        general.grid(row=9, column=0, sticky="ew")
        ttk.Checkbutton(general, text="Search subfolders", variable=self.recursive).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(general, text="Overwrite existing files", variable=self.overwrite).grid(row=1, column=0, sticky="w")
        ttk.Label(left, textvariable=self.transform_count).grid(row=10, column=0, sticky="w", pady=(8, 4))
        self.run_button = ttk.Button(left, text="Create augmented images", style="Run.TButton", command=self._start)
        self.run_button.grid(row=11, column=0, sticky="ew", pady=(4, 8))
        ttk.Label(left, textvariable=self.status, style="Status.TLabel", wraplength=400).grid(row=12, column=0, sticky="ew")

        ttk.Label(right, text="Log", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        log_frame = ttk.Frame(right)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

    def _browse_file(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")])
        if path:
            variable.set(path)

    def _browse_folder(self, variable: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            variable.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def _build_augmentation_options(self) -> AugmentationOptions:
        angle = abs(float(self.small_angle.get())) if self.small_rotation.get() else 0
        shift = abs(int(self.shift_pixels.get())) if self.translation.get() else 0
        return AugmentationOptions(
            include_quarter_turns=self.quarter_turns.get(),
            include_horizontal_flips=self.horizontal_flips.get(),
            small_angles=(-angle, angle) if angle else (),
            shift_pixels=shift,
            fill_mode=(
                "border_median"
                if self.fill_mode.get() == "Per-plane border median"
                else "constant"
            ),
            fill_value=float(self.fill_value.get()),
        )

    def _update_transform_count(self, *_args: object) -> None:
        try:
            count = len(build_augmentations(self._build_augmentation_options()))
            self.transform_count.set(f"Outputs per image: {count}")
        except ValueError:
            self.transform_count.set("Check numeric transform settings")

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            options = self._build_augmentation_options()
            input_path = Path(self.input_path.get().strip())
            output_dir = Path(self.output_dir.get().strip())
            mask_text = self.mask_path.get().strip()
            mask_path = Path(mask_text) if mask_text else None
        except ValueError as exc:
            messagebox.showerror("Invalid option", str(exc))
            return
        self._clear_log()
        self.run_button.configure(state="disabled")
        self.status.set("Running augmentation...")
        self.worker = threading.Thread(
            target=self._worker,
            args=(
                input_path, output_dir, mask_path, options,
                self.recursive.get(), self.overwrite.get(),
            ),
            daemon=True,
        )
        self.worker.start()

    def _worker(
        self,
        input_path: Path,
        output_dir: Path,
        mask_path: Path | None,
        options: AugmentationOptions,
        recursive: bool,
        overwrite: bool,
    ) -> None:
        try:
            self.log_queue.put(f"Input: {input_path}\nOutput: {output_dir}\n")
            if mask_path:
                self.log_queue.put(f"Masks: {mask_path}\n")
            self.log_queue.put(f"Transforms per image: {len(build_augmentations(options))}\n\n")
            results, report_path = augment_images(
                input_path, output_dir, recursive, overwrite, options, mask_path,
            )
            for result in results:
                target = result.destination_path or result.source_path
                details = f" — {result.note}" if result.note else ""
                self.log_queue.put(f"[{result.status.upper()}] {target}{details}\n")
            counts = {
                status: sum(result.status == status for result in results)
                for status in ("written", "skipped", "failed")
            }
            summary = (
                f"Written: {counts['written']}, skipped: {counts['skipped']}, "
                f"failed: {counts['failed']}. Report: {report_path}"
            )
            self.log_queue.put(("done", summary))
        except Exception as exc:
            self.log_queue.put(("error", str(exc)))

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _drain_log_queue(self) -> None:
        max_items_per_tick = 250
        processed = 0
        log_lines: list[str] = []
        while processed < max_items_per_tick:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1
            if isinstance(item, tuple):
                if log_lines:
                    self._append_log("".join(log_lines))
                    log_lines.clear()
                kind, message = item
                self.run_button.configure(state="normal")
                self.status.set("Error" if kind == "error" else message)
                if kind == "error":
                    messagebox.showerror("Augmentation failed", message, parent=self)
                else:
                    self._append_log(f"\n{message}\n")
                    messagebox.showinfo("Augmentation complete", message, parent=self)
            else:
                log_lines.append(str(item))
        if log_lines:
            self._append_log("".join(log_lines))
        delay = 10 if processed == max_items_per_tick else 100
        self.after(delay, self._drain_log_queue)


if __name__ == "__main__":
    AugmentTrainingImagesUi().mainloop()
