"""Desktop interface for the KLT/SIFT planar-object tracker."""

import argparse
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from render import draw_tracking
from tracker import HybridTracker, TrackerConfig

BG = '#0d1624'
PANEL = "#333C49"
PANEL_LIGHT = '#1e3045'
TEXT = '#f1f6fb'
MUTED = '#94a8bd'
ACCENT = "#d6903b"


class TrackerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title('Трекинг ключевых точек')
        root.geometry('1280x800')
        root.minsize(970, 650)
        root.configure(bg=BG)
        self.cap = None
        self.path = None
        self.frame_count = 0
        self.video_fps = 25.0
        self.frame_index = 0
        self.first_frame = None
        self.raw_frame = None
        self.shown_frame = None
        self.roi = None
        self.tracker = None
        self.result = None
        self.fps = None
        self.playing = False
        self.marking = False
        self.anchor = None
        self.drag = None
        self.slider_internal = False
        self.seek_after_id = None
        self.play_after_id = None
        self.photo = None
        self.view = (0, 0, 1.0)
        self.export_thread = None
        self.export_cancel = threading.Event()
        self.messages = queue.Queue()
        self.close_after_export = False

        self.label_var = tk.StringVar(value='Объект')
        self.show_points = tk.BooleanVar(value=True)
        self.show_fps = tk.BooleanVar(value=True)
        self.info = tk.StringVar(
            value='Откройте видео, затем выделите объект на первом кадре.'
        )
        self.current_time = tk.StringVar(value='00:00 / 00:00')
        self.status_var = tk.StringVar(value='НЕ ЗАДАН')
        self.performance = tk.StringVar(value='— FPS')
        self.file_name = tk.StringVar(value='Видео не загружено')
        self.setting_vars = {}
        self._build_style()
        self._build_ui()
        root.bind('<Control-o>', lambda _: self.open_video())
        root.bind('<Control-s>', lambda _: self.export_video())
        root.bind('<space>', self._hotkey_space)
        root.bind('<Left>', lambda event: self._seek_relative(-1, event))
        root.bind('<Right>', lambda event: self._seek_relative(1, event))
        root.bind('<Escape>', lambda _: self._cancel_marking())
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self._poll_export)

    def _build_style(self):
        s = ttk.Style(self.root)
        s.theme_use('clam')
        s.configure('App.TFrame', background=BG)
        s.configure('Panel.TFrame', background=PANEL)
        s.configure(
            'Title.TLabel', background=BG, foreground=TEXT, font=('Arial', 19, 'bold')
        )
        s.configure('Sub.TLabel', background=BG, foreground=MUTED, font=('Arial', 10))
        s.configure(
            'Panel.TLabel', background=PANEL, foreground=TEXT, font=('Arial', 10)
        )
        s.configure(
            'Muted.TLabel', background=PANEL, foreground=MUTED, font=('Arial', 9)
        )
        s.configure(
            'Section.TLabel',
            background=PANEL,
            foreground=ACCENT,
            font=('Arial', 10, 'bold'),
        )
        s.configure(
            'Panel.TCheckbutton', background=PANEL, foreground=TEXT, font=('Arial', 10)
        )
        s.map(
            'Panel.TCheckbutton',
            background=[('active', PANEL)],
            foreground=[('disabled', MUTED)],
        )
        s.configure(
            'Dark.TEntry',
            fieldbackground=PANEL_LIGHT,
            background=PANEL_LIGHT,
            foreground=TEXT,
            insertcolor=TEXT,
        )
        s.configure(
            'Dark.TSpinbox',
            fieldbackground=PANEL_LIGHT,
            background=PANEL_LIGHT,
            foreground=TEXT,
            arrowcolor=ACCENT,
        )
        s.configure(
            'Timeline.Horizontal.TScale', background=BG, troughcolor=PANEL_LIGHT
        )
        s.configure(
            'Export.Horizontal.TProgressbar', background=ACCENT, troughcolor=PANEL_LIGHT
        )

    def _button(self, parent, text, command, accent=True, width=None):
        color = ACCENT if accent else PANEL_LIGHT
        fg = BG if accent else TEXT
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg=fg,
            activebackground='#7be7ce' if accent else '#30465f',
            activeforeground=BG if accent else TEXT,
            relief='flat',
            cursor='hand2',
            padx=15,
            pady=9,
            font=('Arial', 10, 'bold'),
            bd=0,
            width=width,
        )
        return button

    def _build_ui(self):
        header = ttk.Frame(self.root, style='App.TFrame', padding=(22, 15, 22, 12))
        header.pack(fill='x')
        brand = ttk.Frame(header, style='App.TFrame')
        brand.pack(side='left')
        actions = ttk.Frame(header, style='App.TFrame')
        actions.pack(side='right', pady=4)
        self._button(actions, 'Открыть видео', self.open_video).pack(
            side='left', padx=4
        )
        self._button(actions, 'Разметить объект', self.begin_marking).pack(
            side='left', padx=4
        )
        self.play_button = self._button(
            actions, 'Воспроизвести видео', self.toggle_play, accent=True
        )
        self.play_button.pack(side='left', padx=4)
        self._button(actions, 'Экспорт видео', self.export_video).pack(
            side='left', padx=4
        )

        body = ttk.Frame(self.root, style='App.TFrame', padding=(22, 0, 22, 0))
        body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body, style='App.TFrame')
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 14))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        title_bar = ttk.Frame(left, style='App.TFrame', padding=(1, 2, 1, 8))
        title_bar.grid(row=0, column=0, sticky='ew')
        ttk.Label(title_bar, textvariable=self.file_name, style='Sub.TLabel').pack(
            side='left'
        )
        tk.Label(
            title_bar,
            textvariable=self.status_var,
            bg=BG,
            fg=ACCENT,
            font=('Arial', 10, 'bold'),
        ).pack(side='right')
        self.canvas = tk.Canvas(
            left,
            bg='#070d16',
            highlightthickness=1,
            highlightbackground='#2a3a4f',
            cursor='crosshair',
        )
        self.canvas.grid(row=1, column=0, sticky='nsew')
        self.canvas.create_text(
            450,
            280,
            text='Загрузите видео для начала работы',
            fill=MUTED,
            font=('Arial', 16),
        )
        self.canvas.bind('<Configure>', lambda _: self._render_canvas())
        self.canvas.bind('<ButtonPress-1>', self._mouse_down)
        self.canvas.bind('<B1-Motion>', self._mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self._mouse_up)

        timeline = ttk.Frame(left, style='App.TFrame', padding=(0, 13, 0, 0))
        timeline.grid(row=2, column=0, sticky='ew')
        self.scale = ttk.Scale(
            timeline,
            from_=0,
            to=1,
            orient='horizontal',
            style='Timeline.Horizontal.TScale',
            command=self._slider_changed,
        )
        self.scale.pack(fill='x')
        foot = ttk.Frame(left, style='App.TFrame', padding=(0, 7, 0, 2))
        foot.grid(row=3, column=0, sticky='ew')
        ttk.Label(foot, textvariable=self.current_time, style='Sub.TLabel').pack(
            side='left'
        )
        ttk.Label(foot, textvariable=self.performance, style='Sub.TLabel').pack(
            side='right'
        )
        ttk.Label(
            left,
            textvariable=self.info,
            style='Sub.TLabel',
            wraplength=650,
            padding=(0, 10, 0, 6),
        ).grid(row=4, column=0, sticky='ew')

        right = ttk.Frame(body, style='Panel.TFrame', width=305)
        right.grid(row=0, column=1, sticky='ns')
        right.grid_propagate(False)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        side_canvas = tk.Canvas(right, bg=PANEL, highlightthickness=0, width=300)
        scroll = ttk.Scrollbar(right, orient='vertical', command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=scroll.set)
        side_canvas.grid(row=0, column=0, sticky='nsew')
        scroll.grid(row=0, column=1, sticky='ns')
        self.side = ttk.Frame(side_canvas, style='Panel.TFrame', padding=16)
        win = side_canvas.create_window((0, 0), window=self.side, anchor='nw')
        self.side.bind(
            '<Configure>',
            lambda _: side_canvas.configure(scrollregion=side_canvas.bbox('all')),
        )
        side_canvas.bind(
            '<Configure>',
            lambda event: side_canvas.itemconfigure(win, width=event.width),
        )
        self._build_settings()
        tk.Label(
            self.root,
            text='KLT: зелёные точки  |  SIFT: оранжевые точки',
            bg=BG,
            fg=MUTED,
            anchor='w',
            font=('Arial', 9),
        ).pack(fill='x', padx=22, pady=(8, 11))

    def _build_settings(self):
        ttk.Label(self.side, text='ОБЪЕКТ', style='Section.TLabel').pack(
            anchor='w', pady=(0, 8)
        )
        ttk.Label(self.side, text='Подпись на рамке', style='Muted.TLabel').pack(
            anchor='w'
        )
        name_entry = ttk.Entry(
            self.side, textvariable=self.label_var, style='Dark.TEntry'
        )
        name_entry.pack(fill='x', pady=(5, 10))
        self.label_var.trace_add('write', lambda *_: self._compose())
        ttk.Checkbutton(
            self.side,
            text='Отображать ключевые точки',
            variable=self.show_points,
            command=self._compose,
            style='Panel.TCheckbutton',
        ).pack(anchor='w', pady=3)
        ttk.Checkbutton(
            self.side,
            text='Отображать FPS',
            variable=self.show_fps,
            command=self._compose,
            style='Panel.TCheckbutton',
        ).pack(anchor='w', pady=3)
        ttk.Separator(self.side).pack(fill='x', pady=14)
        ttk.Label(self.side, text='ПАРАМЕТРЫ АЛГОРИТМОВ', style='Section.TLabel').pack(
            anchor='w', pady=(0, 8)
        )
        fields = [
            ('max_points', '[KLT] Макс. число точек', 250, 40, 1500, 10),
            ('lk_window', '[KLT] Окно (нечётное)', 21, 5, 61, 2),
            ('lk_levels', '[KLT] Уровни пирамиды', 3, 0, 6, 1),
            ('fb_threshold', '[KLT] FB ошибка, px', 1.5, 0.2, 15, 0.1),
            ('quality', '[Shi–Tomasi] Качество', 0.01, 0.001, 0.2, 0.005),
            ('min_inliers', 'Мин. число согласованных точек', 10, 4, 50, 1),
            ('replenish_at', '[SIFT] Использование точках меньше', 40, 5, 300, 5),
            ('sift_features', '[SIFT] Число признаков', 1500, 100, 6000, 100),
            ('sift_ratio', '[SIFT] Lowe ratio', 0.75, 0.5, 0.95, 0.05),
            ('ransac_px', '[RANSAC] Порог, px', 3.0, 0.5, 12, 0.5),
        ]
        for key, label, default, minimum, maximum, step in fields:
            row = ttk.Frame(self.side, style='Panel.TFrame')
            row.pack(fill='x', pady=3)
            ttk.Label(row, text=label, style='Muted.TLabel').pack(side='left')
            var = tk.StringVar(value=str(default))
            ttk.Spinbox(
                row,
                from_=minimum,
                to=maximum,
                increment=step,
                textvariable=var,
                width=8,
                style='Dark.TSpinbox',
            ).pack(side='right')
            self.setting_vars[key] = (var, minimum, maximum)
        self._button(self.side, 'Применить настройки', self.apply_settings).pack(
            fill='x', pady=(14, 8)
        )
        ttk.Label(
            self.side,
            text='Новые параметры применяются после повторной\nинициализации по первому кадру.',
            style='Muted.TLabel',
            justify='left',
        ).pack(anchor='w')
        ttk.Separator(self.side).pack(fill='x', pady=16)
        ttk.Label(self.side, text='СОХРАНЕНИЕ', style='Section.TLabel').pack(
            anchor='w', pady=(0, 8)
        )
        self.progress = ttk.Progressbar(
            self.side, maximum=100, style='Export.Horizontal.TProgressbar'
        )
        self.progress.pack(fill='x', pady=3)
        self.export_info = ttk.Label(
            self.side, text='Экспорт ещё не запускался', style='Muted.TLabel'
        )
        self.export_info.pack(anchor='w', pady=(5, 6))
        self.cancel_button = self._button(
            self.side, 'Отменить экспорт', self.cancel_export
        )
        self.cancel_button.pack(fill='x')
        self.cancel_button.configure(state='disabled')

    def _read_config(self):
        data = {}
        for key, (var, minimum, maximum) in self.setting_vars.items():
            number = float(var.get().replace(',', '.'))
            if not minimum <= number <= maximum:
                raise ValueError(
                    f'Параметр «{key}» должен быть от {minimum} до {maximum}.'
                )
            if key in {
                'max_points',
                'lk_window',
                'lk_levels',
                'min_inliers',
                'replenish_at',
                'sift_features',
            }:
                if not number.is_integer():
                    raise ValueError(f'Параметр «{key}» должен быть целым числом.')
                number = int(number)
            data[key] = number
        if data['lk_window'] % 2 != 1:
            raise ValueError('Размер окна KLT должен быть нечётным.')
        if data['min_inliers'] > data['max_points']:
            raise ValueError('Минимум точек не может превышать максимум KLT.')
        if data['replenish_at'] > data['max_points']:
            raise ValueError(
                'Порог пополнения должен быть не больше максимума точек KLT.'
            )
        return TrackerConfig(**data)

    def open_video(self, path=None):
        if self._is_exporting():
            messagebox.showinfo('Экспорт', 'Сначала завершите или отмените экспорт.')
            return
        path = path or filedialog.askopenfilename(
            title='Выберите видео',
            filetypes=[
                ('Видео', '*.mp4 *.avi *.mov *.mkv *.m4v *.webm'),
                ('Все файлы', '*.*'),
            ],
        )
        if not path:
            return
        new_cap = cv2.VideoCapture(str(path))
        ok, frame = new_cap.read()
        frames = int(new_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not new_cap.isOpened() or not ok or frames <= 0:
            new_cap.release()
            messagebox.showerror(
                'Не удалось открыть видео',
                'Видео повреждено либо кодек/индекс кадров не поддерживается OpenCV.',
            )
            return
        self.pause()
        if self.cap is not None:
            self.cap.release()
        self.cap = new_cap
        self.path = Path(path)
        self.frame_count = frames
        self.video_fps = float(new_cap.get(cv2.CAP_PROP_FPS))
        if not 1 <= self.video_fps <= 240:
            self.video_fps = 25.0
        self.first_frame = frame.copy()
        self.raw_frame = frame
        self.frame_index = 0
        self.roi = None
        self.tracker = None
        self.result = None
        self.fps = None
        self.performance.set('— FPS')
        self.marking = False
        self.file_name.set(self.path.name)
        self.status_var.set('НУЖНА РАЗМЕТКА')
        self.info.set(
            'Нажмите «Разметить объект» и обведите его мышью на первом кадре.'
        )
        self.scale.configure(to=max(1, frames - 1))
        self._update_timeline()
        self._compose()

    def begin_marking(self):
        if self.cap is None:
            messagebox.showinfo('Видео не загружено', 'Сначала откройте видео.')
            return
        self.pause()
        self._read_at(0)
        self.marking = True
        self.anchor = None
        self.drag = None
        self.info.set('Зажмите левую кнопку мыши и выделите прямоугольник объекта.')
        self._compose()

    def _cancel_marking(self):
        if self.marking:
            self.marking = False
            self.anchor = None
            self.drag = None
            self.info.set('Разметка отменена.')
            self._compose()

    def _mouse_to_source(self, x, y):
        if self.raw_frame is None:
            return None
        x0, y0, scale = self.view
        height, width = self.raw_frame.shape[:2]
        return (
            int(np.clip(round((x - x0) / scale), 0, width - 1)),
            int(np.clip(round((y - y0) / scale), 0, height - 1)),
        )

    def _mouse_down(self, event):
        if self.marking:
            self.anchor = self._mouse_to_source(event.x, event.y)
            self.drag = self.anchor

    def _mouse_move(self, event):
        if self.marking and self.anchor is not None:
            self.drag = self._mouse_to_source(event.x, event.y)
            self._render_canvas()

    def _mouse_up(self, event):
        if not self.marking or self.anchor is None:
            return
        bx, by = self._mouse_to_source(event.x, event.y)
        ax, ay = self.anchor
        x1, x2 = sorted((ax, bx))
        y1, y2 = sorted((ay, by))
        if x2 - x1 < 25 or y2 - y1 < 25:
            self.info.set('Выделите объект размером не менее 25×25 пикселей.')
            return
        roi = (x1, y1, x2 - x1, y2 - y1)
        try:
            config = self._read_config()
            tracker = HybridTracker(self.first_frame, roi, config)
        except (ValueError, cv2.error) as exc:
            messagebox.showerror('Не удалось создать трекер', str(exc))
            return
        self.roi, self.tracker = roi, tracker
        self.result = tracker.reset_first()
        self.fps = None
        self.performance.set('— FPS')
        self.marking = False
        self.anchor = None
        self.drag = None
        self.info.set(
            'Объект задан. Воспроизведите видео, перемотайте его или сохраните результат.'
        )
        self._compose()

    def _slider_changed(self, value):
        if self.slider_internal or self.cap is None:
            return
        self.pause()
        if self.seek_after_id is not None:
            self.root.after_cancel(self.seek_after_id)
        index = round(float(value))
        self.seek_after_id = self.root.after(90, lambda: self._read_at(index))

    def _read_at(self, index):
        if self.cap is None:
            return
        self.seek_after_id = None
        index = max(0, min(self.frame_count - 1, index))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            self.info.set(f'Не удалось прочитать кадр {index}.')
            return
        self.raw_frame = frame
        self.frame_index = index
        if self.tracker is not None:
            start = time.perf_counter()
            try:
                self.result = self.tracker.jump(frame, first=index == 0)
                self._update_fps(time.perf_counter() - start)
            except (cv2.error, ValueError) as exc:
                self.tracker._lose()
                self.result = self.tracker._result('LOST', 'SEARCH')
                self.info.set(f'Не удалось обработать кадр: {exc}')
        self._compose()
        self._update_timeline()

    def _seek_relative(self, delta, event=None):
        if event is not None and isinstance(
            event.widget, (tk.Entry, ttk.Entry, ttk.Spinbox, ttk.Scale)
        ):
            return
        if self.cap is None:
            return
        self.pause()
        self._read_at(self.frame_index + delta)

    def _hotkey_space(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Spinbox)):
            return
        self.toggle_play()

    def toggle_play(self):
        if self.playing:
            self.pause()
        elif self.cap is not None and self.tracker is not None and not self.marking:
            if self.seek_after_id is not None:
                self.root.after_cancel(self.seek_after_id)
                self.seek_after_id = None
                self._read_at(round(self.scale.get()))
            if self.frame_index >= self.frame_count - 1:
                self._read_at(0)
            self.playing = True
            self.play_button.configure(text='❚❚ Пауза')
            self._play_next()
        else:
            self.info.set('Для воспроизведения с трекингом сначала выделите объект.')

    def pause(self):
        self.playing = False
        if self.seek_after_id is not None:
            self.root.after_cancel(self.seek_after_id)
            self.seek_after_id = None
        if self.play_after_id is not None:
            self.root.after_cancel(self.play_after_id)
            self.play_after_id = None
        self.play_button.configure(text='▶ Воспроизвести')

    def _play_next(self):
        if not self.playing:
            return
        start = time.perf_counter()
        ok, frame = self.cap.read()
        if not ok or self.frame_index >= self.frame_count - 1:
            self.pause()
            self.info.set('Воспроизведение завершено.')
            return
        self.raw_frame = frame
        self.frame_index += 1
        try:
            begin_tracking = time.perf_counter()
            self.result = self.tracker.step(frame)
            self._update_fps(time.perf_counter() - begin_tracking)
            self._compose()
            self._update_timeline()
        except cv2.error as exc:
            self.pause()
            messagebox.showerror('Ошибка обработки кадра', str(exc))
            return
        period_ms = 1000 / self.video_fps
        delay = max(1, round(period_ms - 1000 * (time.perf_counter() - start)))
        self.play_after_id = self.root.after(delay, self._play_next)

    def _update_fps(self, seconds):
        instantaneous = 1 / max(seconds, 0.0001)
        self.fps = (
            instantaneous
            if self.fps is None
            else 0.75 * self.fps + 0.25 * instantaneous
        )
        self.performance.set(f'{self.fps:.1f} FPS · обработка')

    def _compose(self):
        if self.raw_frame is None:
            return
        self.shown_frame = (
            self.raw_frame
            if self.marking
            else draw_tracking(
                self.raw_frame,
                self.result,
                self.label_var.get(),
                self.fps,
                self.show_points.get(),
                self.show_fps.get(),
            )
        )
        if self.marking:
            self.status_var.set('РАЗМЕТКА')
        else:
            self.status_var.set(self.result.status if self.result else 'НУЖНА РАЗМЕТКА')
        self._render_canvas()

    def _render_canvas(self):
        self.canvas.delete('all')
        if self.shown_frame is None:
            self.canvas.create_text(
                max(90, self.canvas.winfo_width() // 2),
                max(80, self.canvas.winfo_height() // 2),
                text='Откройте видео, чтобы начать',
                fill=MUTED,
                font=('Arial', 14),
            )
            return
        height, width = self.shown_frame.shape[:2]
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        factor = min(cw / width, ch / height)
        display_width = max(1, int(width * factor))
        display_height = max(1, int(height * factor))
        left, top = (cw - display_width) // 2, (ch - display_height) // 2
        self.view = (left, top, factor)
        rgb = cv2.cvtColor(self.shown_frame, cv2.COLOR_BGR2RGB)
        preview = Image.fromarray(rgb)
        if display_width != width or display_height != height:
            preview = preview.resize(
                (display_width, display_height), Image.Resampling.BILINEAR
            )
        self.photo = ImageTk.PhotoImage(preview)
        self.canvas.create_image(left, top, image=self.photo, anchor='nw')
        if self.marking and self.anchor is not None and self.drag is not None:
            ax, ay = self.anchor
            bx, by = self.drag
            self.canvas.create_rectangle(
                left + ax * factor,
                top + ay * factor,
                left + bx * factor,
                top + by * factor,
                outline=ACCENT,
                width=2,
                dash=(6, 3),
            )

    def _update_timeline(self):
        self.slider_internal = True
        self.scale.set(self.frame_index)
        self.slider_internal = False
        duration = self.frame_count / self.video_fps
        position = self.frame_index / self.video_fps

        def clock(value):
            total = int(value)
            return f'{total // 3600:02}:{(total // 60) % 60:02}:{total % 60:02}'

        self.current_time.set(
            f'{clock(position)} / {clock(duration)}    •    кадр {self.frame_index + 1}/{self.frame_count}'
        )

    def apply_settings(self):
        try:
            config = self._read_config()
            if self.tracker is not None:
                self.pause()
                current_index = self.frame_index
                self.tracker = HybridTracker(self.first_frame, self.roi, config)
                self.result = self.tracker.jump(
                    self.raw_frame, first=current_index == 0
                )
                self.fps = None
                self.performance.set('— FPS')
                self._compose()
            self.info.set('Параметры применены.')
        except (ValueError, cv2.error) as exc:
            messagebox.showerror('Некорректные настройки', str(exc))

    def _is_exporting(self):
        return self.export_thread is not None and self.export_thread.is_alive()

    def export_video(self):
        if self.tracker is None or self.path is None:
            messagebox.showinfo(
                'Нет объекта', 'Сначала откройте видео и выделите объект.'
            )
            return
        if self._is_exporting():
            messagebox.showinfo('Экспорт', 'Сохранение видео уже выполняется.')
            return
        try:
            config = self._read_config()
        except ValueError as exc:
            messagebox.showerror('Некорректные настройки', str(exc))
            return
        destination = filedialog.asksaveasfilename(
            title='Сохранить видео с трекингом',
            defaultextension='.mp4',
            filetypes=[('MP4 video', '*.mp4')],
            initialfile=f'{self.path.stem}_tracked.mp4',
        )
        if not destination:
            return
        output = Path(destination)
        if output.resolve() == self.path.resolve():
            messagebox.showerror(
                'Неверное имя файла',
                'Выходной файл не должен совпадать с исходным видео.',
            )
            return
        self.pause()
        self.progress['value'] = 0
        self.export_info.configure(text='Обработка: 0%')
        self.export_cancel.clear()
        self.cancel_button.configure(state='normal')
        self.export_thread = threading.Thread(
            target=self._export_worker,
            args=(
                self.path,
                output,
                self.roi,
                config,
                self.label_var.get(),
                self.show_points.get(),
                self.show_fps.get(),
                self.video_fps,
                self.frame_count,
            ),
            daemon=True,
        )
        self.export_thread.start()
        self.info.set(
            f'Экспорт в {output.name} запущен. Можно просматривать исходное видео.'
        )

    def _export_worker(
        self, source, output, roi, config, label, points, show_fps, fps, total
    ):
        cap = cv2.VideoCapture(str(source))
        writer = None
        completed = False
        index = 0
        error = None
        try:
            ok, first = cap.read()
            if not ok:
                raise OSError('Не удалось прочитать исходное видео.')
            height, width = first.shape[:2]
            writer = cv2.VideoWriter(
                str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
            )
            if not writer.isOpened():
                raise OSError(
                    'Не удалось создать MP4. Проверьте путь и поддержку кодека mp4v.'
                )
            tracker = HybridTracker(first, roi, config)
            result = tracker.reset_first()
            frame = first
            rolling_fps = None
            while not self.export_cancel.is_set():
                if index:
                    started = time.perf_counter()
                    result = tracker.step(frame)
                    speed = 1 / max(time.perf_counter() - started, 0.0001)
                    rolling_fps = (
                        speed
                        if rolling_fps is None
                        else 0.75 * rolling_fps + 0.25 * speed
                    )
                writer.write(
                    draw_tracking(frame, result, label, rolling_fps, points, show_fps)
                )
                index += 1
                if index % 5 == 0 or index == total:
                    self.messages.put(('progress', index, total))
                ok, frame = cap.read()
                if not ok:
                    completed = True
                    break
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if not completed:
                try:
                    output.unlink(missing_ok=True)
                except OSError:
                    pass
            if error:
                self.messages.put(('error', error))
            else:
                self.messages.put(
                    ('done' if completed else 'cancel', str(output), index)
                )

    def cancel_export(self):
        if self._is_exporting():
            self.export_cancel.set()
            self.export_info.configure(text='Отмена экспорта…')

    def _poll_export(self):
        try:
            while True:
                msg = self.messages.get_nowait()
                if msg[0] == 'progress':
                    _, count, total = msg
                    percent = 100 * count / max(total, 1)
                    self.progress['value'] = min(percent, 100)
                    self.export_info.configure(
                        text=f'Обработка: {count}/{total} кадров ({percent:.0f}%)'
                    )
                elif msg[0] == 'done':
                    self.progress['value'] = 100
                    self.export_info.configure(text=f'Готово: {msg[2]} кадров')
                    self.info.set(f'Видео сохранено: {msg[1]}')
                    self.cancel_button.configure(state='disabled')
                elif msg[0] == 'cancel':
                    self.export_info.configure(
                        text='Экспорт отменён; неполный файл удалён'
                    )
                    self.cancel_button.configure(state='disabled')
                elif msg[0] == 'error':
                    self.export_info.configure(text='Ошибка экспорта')
                    self.cancel_button.configure(state='disabled')
                    messagebox.showerror('Ошибка экспорта', msg[1])
        except queue.Empty:
            pass
        if self.close_after_export and not self._is_exporting():
            self._shutdown()
            return
        self.root.after(100, self._poll_export)

    def close(self):
        self.pause()
        if self._is_exporting():
            if not messagebox.askyesno(
                'Экспорт', 'Прервать экспорт и закрыть приложение?'
            ):
                return
            self.close_after_export = True
            self.export_cancel.set()
            return
        self._shutdown()

    def _shutdown(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(
        description='PlaneTrack: KLT + SIFT desktop tracker'
    )
    parser.add_argument('video', nargs='?', help='Необязательный путь к видео')
    args = parser.parse_args()
    root = tk.Tk()
    app = TrackerApp(root)
    if args.video:
        root.after(150, lambda: app.open_video(args.video))
    root.mainloop()


if __name__ == '__main__':
    main()
