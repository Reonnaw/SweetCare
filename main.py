import tkinter as tk
from tkinter import ttk, scrolledtext
from collections import defaultdict

DATASET = [
    ("Muda", "Tinggi", "Tinggi", "Tinggi", "Ya"),
    ("Tua", "Tinggi", "Tinggi", "Tinggi", "Ya"),
    ("Muda", "Normal", "Normal", "Normal", "Tidak"),
    ("Paruh Baya", "Tinggi", "Tinggi", "Sedang", "Ya"),
    ("Tua", "Normal", "Normal", "Normal", "Tidak"),
    ("Paruh Baya", "Normal", "Tinggi", "Tinggi", "Ya"),
    ("Muda", "Tinggi", "Normal", "Sedang", "Tidak"),
    ("Tua", "Tinggi", "Tinggi", "Tinggi", "Ya"),
    ("Paruh Baya", "Normal", "Normal", "Normal", "Tidak"),
    ("Muda", "Tinggi", "Tinggi", "Sedang", "Ya"),
    ("Tua", "Normal", "Normal", "Normal", "Tidak"),
    ("Paruh Baya", "Tinggi", "Tinggi", "Tinggi", "Ya"),
    ("Muda", "Normal", "Normal", "Sedang", "Tidak"),
    ("Tua", "Tinggi", "Tinggi", "Tinggi", "Ya"),
    ("Paruh Baya", "Normal", "Normal", "Normal", "Tidak"),
    ("Muda", "Tinggi", "Tinggi", "Tinggi", "Ya"),
    ("Tua", "Normal", "Normal", "Sedang", "Tidak"),
    ("Paruh Baya", "Tinggi", "Tinggi", "Tinggi", "Ya"),
    ("Muda", "Normal", "Normal", "Normal", "Tidak"),
    ("Tua", "Tinggi", "Tinggi", "Tinggi", "Ya"),
]

FEATURES = ["Umur", "BMI", "Gula Darah", "Tekanan Darah"]
CLASSES = ["Ya", "Tidak"]

OPTIONS = {
    "Umur": ["Muda", "Paruh Baya", "Tua"],
    "BMI": ["Normal", "Tinggi"],
    "Gula Darah": ["Normal", "Tinggi"],
    "Tekanan Darah": ["Normal", "Sedang", "Tinggi"],
}

C = {
    "bg":        "#fdf0f3",
    "surface":   "#ffffff",
    "surface2":  "#fce8ed",
    "surface3":  "#fad4de",
    "border":    "#f0bfca",
    "border2":   "#e8a0b0",
    "accent":    "#d4607a",
    "accent_h":  "#b84d65",
    "text":      "#3d1f28",
    "text2":     "#7a4455",
    "text3":     "#a06070",
    "yes_bg":    "#fff0f2",
    "yes_txt":   "#c0374f",
    "yes_alt":   "#ffe8ec",
    "no_bg":     "#f0faf5",
    "no_txt":    "#2a7a56",
    "no_alt":    "#e4f5ed",
    "warn_bg":   "#fef9ec",
    "warn_txt":  "#9a7010",
}


class NaiveBayesClassifier:
    def __init__(self, laplace=1.0):
        self.laplace = laplace
        self.n_total = len(DATASET)
        self._train()

    def _train(self):
        self.class_counts = defaultdict(int)
        self.feat_val_counts = {}

        for feat in FEATURES:
            self.feat_val_counts[feat] = defaultdict(lambda: defaultdict(int))

        for row in DATASET:
            cls = row[4]
            self.class_counts[cls] += 1
            for i, feat in enumerate(FEATURES):
                self.feat_val_counts[feat][cls][row[i]] += 1

        self.prior = {cls: self.class_counts[cls] / self.n_total for cls in CLASSES}

        self.likelihood = {}
        for feat in FEATURES:
            self.likelihood[feat] = {}
            n_vals = len(OPTIONS[feat])
            for cls in CLASSES:
                self.likelihood[feat][cls] = {}
                cls_total = self.class_counts[cls]
                for val in OPTIONS[feat]:
                    count = self.feat_val_counts[feat][cls].get(val, 0)
                    self.likelihood[feat][cls][val] = (
                        (count + self.laplace) / (cls_total + self.laplace * n_vals)
                    )

    def predict_with_steps(self, test):
        lines = []
        lines.append("=" * 56)
        lines.append("       RINCIAN PERHITUNGAN NAIVE BAYES")
        lines.append("=" * 56)
        lines.append("")
        lines.append(f"  Total data latih   : {self.n_total} sampel")
        lines.append(f"  Laplace smoothing  : alpha = {int(self.laplace)}")
        lines.append(f"  Data uji           : {', '.join(f'{k}={v}' for k, v in test.items())}")
        lines.append("")

        scores = {}

        for cls in CLASSES:
            n_cls = self.class_counts[cls]
            prior = self.prior[cls]
            lines.append("-" * 56)
            lines.append(f"  KELAS : {cls}")
            lines.append("-" * 56)
            lines.append(f"  Jumlah data '{cls}'  = {n_cls} dari {self.n_total}")
            lines.append(f"  P({cls})             = {n_cls}/{self.n_total} = {prior:.4f}")
            lines.append("")

            product = prior
            for feat in FEATURES:
                val = test[feat]
                n_v = len(OPTIONS[feat])
                raw = self.feat_val_counts[feat][cls].get(val, 0)
                prob = self.likelihood[feat][cls][val]
                numer = raw + int(self.laplace)
                denom = n_cls + int(self.laplace) * n_v
                lines.append(
                    f"  P({feat}={val} | {cls})"
                )
                lines.append(
                    f"    = ({raw} + {int(self.laplace)}) / ({n_cls} + {int(self.laplace)}x{n_v})"
                    f"  =  {numer}/{denom}  =  {prob:.4f}"
                )
                product *= prob

            lines.append("")
            lines.append(f"  Skor '{cls}'  =  {prior:.4f}")
            for feat in FEATURES:
                val = test[feat]
                prob = self.likelihood[feat][cls][val]
                lines.append(f"           x  {prob:.4f}  (P({feat}={val}|{cls}))")
            lines.append(f"           =  {product:.8f}")
            lines.append("")
            scores[cls] = product

        prediction = max(scores, key=scores.__getitem__)

        lines.append("=" * 56)
        lines.append("  PERBANDINGAN SKOR AKHIR")
        lines.append("=" * 56)
        lines.append("")
        for cls in CLASSES:
            marker = "  <<< TERPILIH" if cls == prediction else ""
            lines.append(f"  Skor '{cls:9}' = {scores[cls]:.8f}{marker}")
        lines.append("")
        lines.append(f"  Kesimpulan : Diabetes = {prediction}")
        lines.append("")
        lines.append("=" * 56)

        return prediction, "\n".join(lines)


class SweetCareApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SweetCare — Kalkulator Risiko Diabetes")
        self.geometry("1200x720")
        self.minsize(960, 620)
        self.configure(bg=C["bg"])

        self.nb = NaiveBayesClassifier()
        self._apply_style()
        self._build_ui()
        self._fill_table()

        self.combos["Umur"].set("Paruh Baya")
        self.combos["BMI"].set("Tinggi")
        self.combos["Gula Darah"].set("Tinggi")
        self.combos["Tekanan Darah"].set("Tinggi")

    def _apply_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        s.configure(".",
            background=C["bg"], foreground=C["text"],
            fieldbackground=C["surface"], font=("Georgia", 11))

        s.configure("Treeview",
            background=C["surface"], foreground=C["text"],
            fieldbackground=C["surface"], rowheight=30,
            borderwidth=0, font=("Segoe UI", 10))
        s.configure("Treeview.Heading",
            background=C["surface2"], foreground=C["text"],
            font=("Georgia", 10, "bold"), relief="flat", padding=(0, 6))
        s.map("Treeview",
            background=[("selected", C["surface3"])],
            foreground=[("selected", C["text"])])

        s.configure("TCombobox",
            background=C["surface"], foreground=C["text"],
            selectbackground=C["border"], fieldbackground=C["surface"],
            arrowcolor=C["accent"], font=("Segoe UI", 11))
        s.map("TCombobox",
            fieldbackground=[("readonly", C["surface"])],
            arrowcolor=[("active", C["accent_h"])])

        s.configure("TLabelframe",
            background=C["bg"], foreground=C["text"],
            bordercolor=C["border"], relief="solid", borderwidth=1)
        s.configure("TLabelframe.Label",
            background=C["bg"], foreground=C["text2"],
            font=("Georgia", 11, "italic"))

        s.configure("Predict.TButton",
            background=C["accent"], foreground=C["surface"],
            font=("Georgia", 12, "bold"),
            borderwidth=0, relief="flat", padding=(24, 10))
        s.map("Predict.TButton",
            background=[("active", C["accent_h"]), ("pressed", C["border2"])])

        s.configure("TSeparator", background=C["border"])

    def _build_ui(self):
        header = tk.Frame(self, bg=C["surface2"], pady=0)
        header.pack(fill="x")

        inner_hdr = tk.Frame(header, bg=C["surface2"])
        inner_hdr.pack(fill="x", padx=24, pady=14)

        title_frame = tk.Frame(inner_hdr, bg=C["surface2"])
        title_frame.pack(side="left")

        tk.Label(
            title_frame,
            text="SweetCare",
            bg=C["surface2"], fg=C["accent"],
            font=("Georgia", 22, "bold italic")
        ).pack(anchor="w")

        tk.Label(
            title_frame,
            text="Kalkulator Risiko Diabetes  |  Naive Bayes  |  Laplace alpha = 1",
            bg=C["surface2"], fg=C["text2"],
            font=("Segoe UI", 10)
        ).pack(anchor="w")

        badge_frame = tk.Frame(inner_hdr, bg=C["surface3"],
                               padx=14, pady=6, relief="flat")
        badge_frame.pack(side="right", padx=(0, 4))
        tk.Label(
            badge_frame,
            text="Studi Kasus 4",
            bg=C["surface3"], fg=C["text2"],
            font=("Georgia", 10, "italic")
        ).pack()
        tk.Label(
            badge_frame,
            text="20 Data Latih",
            bg=C["surface3"], fg=C["accent"],
            font=("Georgia", 11, "bold")
        ).pack()

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=20, pady=16)

        body.grid_columnconfigure(0, weight=4, minsize=320)
        body.grid_columnconfigure(1, weight=1, minsize=8)
        body.grid_columnconfigure(2, weight=6, minsize=480)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=C["bg"])
        left.grid(row=0, column=0, sticky="nsew")

        left_card = tk.Frame(left, bg=C["surface"], relief="flat",
                             highlightbackground=C["border"], highlightthickness=1)
        left_card.pack(fill="both", expand=True)

        lbl_header = tk.Frame(left_card, bg=C["surface2"], padx=12, pady=8)
        lbl_header.pack(fill="x")
        tk.Label(lbl_header, text="Data Latih", bg=C["surface2"], fg=C["text2"],
                 font=("Georgia", 11, "italic bold")).pack(anchor="w")
        tk.Label(lbl_header, text="20 sampel training set", bg=C["surface2"],
                 fg=C["text3"], font=("Segoe UI", 9)).pack(anchor="w")

        tree_frame = tk.Frame(left_card, bg=C["surface"])
        tree_frame.pack(fill="both", expand=True, padx=1, pady=1)

        cols = ("No", "Umur", "BMI", "Gula Darah", "Tekanan Darah", "Diabetes")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                 selectmode="browse")
        widths = [34, 84, 64, 84, 104, 66]
        for col, w in zip(cols, widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center", stretch=True)

        self.tree.tag_configure("ya",      background=C["yes_bg"],  foreground=C["yes_txt"])
        self.tree.tag_configure("ya_alt",  background=C["yes_alt"], foreground=C["yes_txt"])
        self.tree.tag_configure("tidak",   background=C["no_bg"],   foreground=C["no_txt"])
        self.tree.tag_configure("tidak_alt",background=C["no_alt"], foreground=C["no_txt"])

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        tk.Frame(body, bg=C["border"], width=1).grid(row=0, column=1, sticky="ns", padx=8)

        right = tk.Frame(body, bg=C["bg"])
        right.grid(row=0, column=2, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        input_card = tk.Frame(right, bg=C["surface"], relief="flat",
                              highlightbackground=C["border"], highlightthickness=1)
        input_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        inp_hdr = tk.Frame(input_card, bg=C["surface2"], padx=14, pady=8)
        inp_hdr.pack(fill="x")
        tk.Label(inp_hdr, text="Data Uji", bg=C["surface2"], fg=C["text2"],
                 font=("Georgia", 11, "italic bold")).pack(anchor="w")
        tk.Label(inp_hdr, text="Pilih nilai atribut pasien yang akan diuji",
                 bg=C["surface2"], fg=C["text3"], font=("Segoe UI", 9)).pack(anchor="w")

        fields_frame = tk.Frame(input_card, bg=C["surface"], padx=16, pady=12)
        fields_frame.pack(fill="x")

        self.combos = {}
        for idx, feat in enumerate(FEATURES):
            col_off = (idx % 2) * 2
            row_pos = idx // 2

            lbl = tk.Label(fields_frame, text=feat, bg=C["surface"],
                           fg=C["text2"], font=("Segoe UI", 10, "bold"),
                           anchor="w")
            lbl.grid(row=row_pos*2, column=col_off, sticky="w",
                     padx=(0 if col_off == 0 else 20, 0), pady=(0, 2))

            cb = ttk.Combobox(fields_frame, values=OPTIONS[feat],
                              state="readonly", width=15,
                              font=("Segoe UI", 11))
            cb.grid(row=row_pos*2+1, column=col_off, sticky="ew",
                    padx=(0 if col_off == 0 else 20, 0), pady=(0, 8))
            self.combos[feat] = cb

        fields_frame.grid_columnconfigure(0, weight=1)
        fields_frame.grid_columnconfigure(2, weight=1)

        btn_bar = tk.Frame(input_card, bg=C["surface"], pady=4)
        btn_bar.pack(fill="x", padx=16, pady=(0, 14))

        predict_btn = ttk.Button(
            btn_bar, text="Mulai Prediksi",
            style="Predict.TButton",
            command=self._run_prediction
        )
        predict_btn.pack(side="left")

        reset_lbl = tk.Label(btn_bar, text="Reset",
                             bg=C["surface"], fg=C["text3"],
                             font=("Segoe UI", 10, "underline"),
                             cursor="hand2")
        reset_lbl.pack(side="left", padx=14)
        reset_lbl.bind("<Button-1>", self._reset)
        reset_lbl.bind("<Enter>", lambda e: reset_lbl.configure(fg=C["accent"]))
        reset_lbl.bind("<Leave>", lambda e: reset_lbl.configure(fg=C["text3"]))

        self.result_var = tk.StringVar(value="Belum diprediksi")

        self.result_card = tk.Frame(right, bg=C["surface2"], relief="flat",
                                    highlightbackground=C["border"], highlightthickness=1,
                                    pady=0)
        self.result_card.grid(row=1, column=0, sticky="nsew")
        self.result_card.grid_rowconfigure(1, weight=1)
        self.result_card.grid_columnconfigure(0, weight=1)

        res_hdr = tk.Frame(self.result_card, bg=C["surface2"], padx=14, pady=8)
        res_hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(res_hdr, text="Langkah Perhitungan", bg=C["surface2"], fg=C["text2"],
                 font=("Georgia", 11, "italic bold")).pack(side="left")

        self.result_badge = tk.Label(
            res_hdr,
            textvariable=self.result_var,
            bg=C["surface3"], fg=C["text3"],
            font=("Georgia", 10, "bold"),
            padx=10, pady=3,
            relief="flat"
        )
        self.result_badge.pack(side="right")

        text_frame = tk.Frame(self.result_card, bg=C["surface"])
        text_frame.grid(row=1, column=0, sticky="nsew", padx=1, pady=1)

        self.steps_text = scrolledtext.ScrolledText(
            text_frame,
            bg=C["surface"], fg=C["text"],
            insertbackground=C["accent"],
            font=("Consolas", 10),
            relief="flat", bd=0,
            state="disabled",
            wrap="none",
            padx=14, pady=10,
            selectbackground=C["border"],
        )
        self.steps_text.pack(fill="both", expand=True)

        self._show_placeholder()

    def _show_placeholder(self):
        self.steps_text.configure(state="normal")
        self.steps_text.delete("1.0", "end")
        placeholder = (
            "\n"
            "  Selamat datang di SweetCare.\n\n"
            "  Pilih nilai atribut pasien pada panel Data Uji,\n"
            "  lalu klik 'Mulai Prediksi' untuk melihat\n"
            "  perhitungan Naive Bayes secara lengkap.\n\n"
            "  Dataset     : 20 data training\n"
            "  Metode      : Naive Bayes + Laplace Smoothing\n"
            "  Studi Kasus : 4 - Risiko Diabetes\n"
        )
        self.steps_text.insert("end", placeholder)
        self.steps_text.configure(state="disabled", fg=C["text3"])

    def _fill_table(self):
        for i, row in enumerate(DATASET, 1):
            is_ya = row[4] == "Ya"
            alt = i % 2 == 0
            if is_ya:
                tag = "ya_alt" if alt else "ya"
            else:
                tag = "tidak_alt" if alt else "tidak"
            self.tree.insert("", "end", values=(i,) + row, tags=(tag,))

    def _reset(self, event=None):
        for cb in self.combos.values():
            cb.set("")
        self.result_var.set("Belum diprediksi")
        self.result_badge.configure(bg=C["surface3"], fg=C["text3"])
        self._show_placeholder()

    def _run_prediction(self):
        test = {feat: self.combos[feat].get() for feat in FEATURES}

        if not all(test.values()):
            self.result_var.set("Lengkapi semua atribut")
            self.result_badge.configure(bg=C["warn_bg"], fg=C["warn_txt"])
            return

        prediction, steps = self.nb.predict_with_steps(test)

        self.steps_text.configure(state="normal", fg=C["text"])
        self.steps_text.delete("1.0", "end")
        self.steps_text.insert("end", steps)
        self.steps_text.configure(state="disabled")
        self.steps_text.see("1.0")

        if prediction == "Ya":
            self.result_var.set("Diabetes : Ya  (Risiko Tinggi)")
            self.result_badge.configure(bg=C["yes_bg"], fg=C["yes_txt"])
        else:
            self.result_var.set("Diabetes : Tidak  (Risiko Rendah)")
            self.result_badge.configure(bg=C["no_bg"], fg=C["no_txt"])


if __name__ == "__main__":
    app = SweetCareApp()
    app.mainloop()