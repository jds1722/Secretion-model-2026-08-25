from __future__ import annotations

import csv
import queue
import shutil
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from sorter_core import AnalysisResult, analyze_tiff


class SorterUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Multi-layer TIFF Sorter")
        self.root.geometry("1160x760")
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.min_size = tk.IntVar(value=25)
        self.max_overlap = tk.DoubleVar(value=0.0)
        self.margin = tk.IntVar(value=0)
        self.background_radius = tk.IntVar(value=10)
        self.recursive = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="입력 폴더를 선택하세요.")
        self.events: queue.Queue = queue.Queue()
        self.results: list[AnalysisResult | Exception] = []
        self.photo = None
        self._build()

    def _build(self):
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Segoe UI", 19, "bold"))
        body = ttk.Frame(self.root, padding=16)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Multi-layer TIFF Sorter", style="Header.TLabel").pack(anchor="w")
        ttk.Label(body, text="배열 index 1(파랑)과 index 5(빨강)·6(초록)의 오브젝트가 겹치지 않는 TIFF를 선별합니다.").pack(anchor="w", pady=(2, 12))

        options = ttk.LabelFrame(body, text="폴더 및 판정 설정", padding=10)
        options.pack(fill="x")
        self._folder_row(options, 0, "입력 폴더", self.input_path, self.choose_input)
        self._folder_row(options, 1, "출력 폴더", self.output_path, self.choose_output)
        ttk.Label(options, text="최소 오브젝트(px)").grid(row=2, column=0, sticky="w", pady=7)
        ttk.Spinbox(options, from_=1, to=999999, textvariable=self.min_size, width=9).grid(row=2, column=1, sticky="w")
        ttk.Label(options, text="허용 겹침률(%)").grid(row=2, column=2, padx=(18, 6))
        ttk.Spinbox(options, from_=0, to=100, increment=0.1, textvariable=self.max_overlap, width=9).grid(row=2, column=3)
        ttk.Label(options, text="접촉 여유(px)").grid(row=2, column=4, padx=(18, 6))
        ttk.Spinbox(options, from_=0, to=50, textvariable=self.margin, width=8).grid(row=2, column=5)
        ttk.Checkbutton(options, text="하위 폴더 포함", variable=self.recursive).grid(row=2, column=6, padx=16)
        ttk.Label(options, text="index 5·6 배경 제거 반경(px)").grid(row=3, column=0, sticky="w", pady=7)
        ttk.Spinbox(options, from_=0, to=500, textvariable=self.background_radius, width=9).grid(row=3, column=1, sticky="w")
        ttk.Label(options, text="0이면 배경 제거 안 함 · 오브젝트보다 큰 반경 권장", foreground="#555555").grid(
            row=3, column=2, columnspan=5, sticky="w", padx=(18, 0)
        )
        options.columnconfigure(1, weight=1)

        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=10)
        self.sample_button = ttk.Button(actions, text="대표 이미지 미리보기", command=self.preview_sample)
        self.sample_button.pack(side="left", padx=(0, 8))
        self.run_button = ttk.Button(actions, text="분석 및 선별 시작", command=self.start)
        self.run_button.pack(side="left")
        self.progress = ttk.Progressbar(actions, length=300)
        self.progress.pack(side="left", padx=12)
        ttk.Label(actions, textvariable=self.status).pack(side="left")

        panes = ttk.Panedwindow(body, orient="horizontal")
        panes.pack(fill="both", expand=True)
        table_box = ttk.Frame(panes)
        preview_box = ttk.LabelFrame(panes, text="겹침 판정 미리보기", padding=8)
        panes.add(table_box, weight=3)
        panes.add(preview_box, weight=2)

        columns = ("name", "decision", "overlap", "nucleus", "signal")
        self.table = ttk.Treeview(table_box, columns=columns, show="headings")
        definitions = (("name", "파일", 270), ("decision", "판정", 70), ("overlap", "겹침률", 85),
                       ("nucleus", "핵 픽셀", 90), ("signal", "5·6 신호", 90))
        for key, title, width in definitions:
            self.table.heading(key, text=title)
            self.table.column(key, width=width, anchor="w" if key == "name" else "center")
        scrollbar = ttk.Scrollbar(table_box, command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.table.bind("<<TreeviewSelect>>", self.show_preview)
        ttk.Label(
            preview_box,
            text="① 원본 합성   ② 검출 마스크(index 1=파랑, 5=빨강, 6=초록)   ③ 겹침=노랑",
            anchor="center",
        ).pack(fill="x", pady=(0, 5))
        self.preview = ttk.Label(preview_box, text="분석 항목을 선택하세요.", anchor="center")
        self.preview.pack(fill="both", expand=True)
        self.preview_info = tk.StringVar(value="")
        ttk.Label(preview_box, textvariable=self.preview_info, anchor="center").pack(fill="x", pady=(5, 0))

    @staticmethod
    def _folder_row(parent, row, label, variable, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, columnspan=5, sticky="ew", padx=8)
        ttk.Button(parent, text="찾아보기", command=command).grid(row=row, column=6)

    def choose_input(self):
        folder = filedialog.askdirectory(title="TIFF 입력 폴더")
        if folder:
            self.input_path.set(folder)
            if not self.output_path.get():
                self.output_path.set(str(Path(folder) / "selected_no_overlap"))

    def choose_output(self):
        folder = filedialog.askdirectory(title="선별 결과 폴더")
        if folder:
            self.output_path.set(folder)

    def preview_sample(self):
        initial = self.input_path.get().strip() or None
        path = filedialog.askopenfilename(
            title="대표 멀티레이어 TIFF 선택",
            initialdir=initial,
            filetypes=(("TIFF image", "*.tif *.tiff"), ("All files", "*.*")),
        )
        if not path:
            return
        settings = (self.min_size.get(), self.max_overlap.get(), self.margin.get(), self.background_radius.get())
        self.sample_button.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.status.set(f"대표 이미지 분석 중: {Path(path).name}")
        threading.Thread(target=self._sample_worker, args=(Path(path), settings), daemon=True).start()
        self.root.after(75, self._poll)

    def _sample_worker(self, path: Path, settings: tuple):
        try:
            self.events.put(("sample", analyze_tiff(path, *settings)))
        except Exception as error:
            self.events.put(("sample_error", path, error))

    def start(self):
        input_text, output_text = self.input_path.get().strip(), self.output_path.get().strip()
        if not input_text or not output_text or not Path(input_text).is_dir():
            messagebox.showerror("설정 오류", "유효한 입력 폴더와 출력 폴더를 지정하세요.")
            return
        source, output = Path(input_text), Path(output_text)
        iterator = source.rglob("*") if self.recursive.get() else source.iterdir()
        files = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
        if not files:
            messagebox.showinfo("TIFF 없음", "입력 폴더에서 TIFF 파일을 찾지 못했습니다.")
            return
        settings = (self.min_size.get(), self.max_overlap.get(), self.margin.get(), self.background_radius.get())
        self.results.clear()
        self.table.delete(*self.table.get_children())
        self.progress.configure(maximum=len(files), value=0)
        self.run_button.configure(state="disabled")
        threading.Thread(target=self._worker, args=(files, output, settings), daemon=True).start()
        self.root.after(75, self._poll)

    def _worker(self, files: list[Path], output: Path, settings: tuple):
        output.mkdir(parents=True, exist_ok=True)
        report = []
        selected_count = 0
        for index, path in enumerate(files, 1):
            try:
                result = analyze_tiff(path, *settings)
                if result.selected:
                    destination = output / path.name
                    number = 1
                    while destination.exists():
                        destination = output / f"{path.stem}_{number}{path.suffix}"
                        number += 1
                    shutil.copy2(path, destination)
                    selected_count += 1
                report.append((
                    path, result.selected, result.overlap_percent, result.overlap_pixels,
                    result.nucleus_pixels, result.signal_pixels, result.touching_nucleus_count,
                    result.nucleus_object_count, ""
                ))
                self.events.put(("row", index, len(files), result))
            except Exception as error:
                report.append((path, False, "", 0, 0, 0, 0, 0, str(error)))
                self.events.put(("error", index, len(files), path, error))
        with (output / "analysis_report.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow((
                "file", "selected", "overlap_percent", "overlap_pixels", "nucleus_pixels",
                "signal_pixels", "touching_nucleus_objects", "total_nucleus_objects", "error"
            ))
            writer.writerows(report)
        self.events.put(("done", selected_count, len(files), output))

    def _poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "row":
                    _, index, total, result = event
                    self.results.append(result)
                    self.table.insert("", "end", iid=str(len(self.results) - 1), values=(result.path.name,
                        "선택" if result.selected else "제외", f"{result.overlap_percent:.3f}%",
                        result.nucleus_pixels, result.signal_pixels))
                    self.progress["value"] = index
                    self.status.set(f"분석 중 {index}/{total}")
                elif event[0] == "error":
                    _, index, total, path, error = event
                    self.results.append(error)
                    self.table.insert("", "end", iid=str(len(self.results) - 1), values=(path.name, "오류", "-", 0, 0))
                    self.progress["value"] = index
                    self.status.set(f"분석 중 {index}/{total}")
                elif event[0] == "sample":
                    result = event[1]
                    self._display_result(result)
                    decision = "선택" if result.selected else "제외"
                    self.status.set(
                        f"대표 이미지: {result.path.name} | 예상 판정: {decision} | 겹침률: {result.overlap_percent:.3f}%"
                    )
                    self.sample_button.configure(state="normal")
                    self.run_button.configure(state="normal")
                elif event[0] == "sample_error":
                    _, path, error = event
                    self.preview.configure(image="", text=f"대표 이미지 분석 오류\n{error}")
                    self.status.set(f"대표 이미지 오류: {path.name}")
                    self.sample_button.configure(state="normal")
                    self.run_button.configure(state="normal")
                    messagebox.showerror("미리보기 오류", str(error))
                elif event[0] == "done":
                    _, selected, total, output = event
                    self.run_button.configure(state="normal")
                    self.sample_button.configure(state="normal")
                    self.status.set(f"완료: {total}개 중 {selected}개 선택")
                    messagebox.showinfo("완료", f"{selected}개 TIFF를 선별했습니다.\n\n결과: {output}")
        except queue.Empty:
            pass
        if str(self.run_button["state"]) == "disabled" or str(self.sample_button["state"]) == "disabled":
            self.root.after(75, self._poll)

    def show_preview(self, _event=None):
        selection = self.table.selection()
        if not selection:
            return
        result = self.results[int(selection[0])]
        if isinstance(result, Exception):
            self.preview.configure(image="", text=f"분석 오류\n{result}")
            return
        self._display_result(result)

    def _display_result(self, result: AnalysisResult):
        image = Image.fromarray(result.preview_rgb)
        image.thumbnail((560, 430), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo, text="")
        self.preview_info.set(
            f"겹침 {result.overlap_pixels:,} px ({result.overlap_percent:.3f}%)  |  "
            f"겹친 index 1 오브젝트 {result.touching_nucleus_count}/{result.nucleus_object_count}개"
        )


if __name__ == "__main__":
    window = tk.Tk()
    SorterUI(window)
    window.mainloop()
