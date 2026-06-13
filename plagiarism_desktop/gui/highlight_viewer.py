import tkinter as tk
from tkinter import ttk
from core.highlighter import split_sentences

COLOR_COPY = "#FFCCCC"
COLOR_PARA = "#FFE5CC"
COLOR_CK = "#E8E8E8"


class HighlightViewerWindow:
    def __init__(self, parent, text_a, text_b, matches, name_a, name_b):
        self.matches = matches
        self.filter_val = "all"

        lang = matches[0].get("language", "vi") if matches else "vi"
        self.orig_sents = [s.strip() for s in split_sentences(text_a, lang) if s.strip()]
        self.susp_sents = [s.strip() for s in split_sentences(text_b, lang) if s.strip()]

        self._build_lookups()

        self.window = tk.Toplevel(parent)
        self.window.title("SO SÁNH VĂN BẢN — HIGHLIGHT TRÙNG LẶP")
        self.window.geometry("1200x700")
        self.window.transient(parent)
        self.window.grab_set()

        main = ttk.Frame(self.window, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar = ttk.Frame(main)
        toolbar.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(toolbar, text="🎯 Bộ lọc:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 8))

        for val, label, color in [
            ("all", "Tất cả", None),
            ("copy", "🔴 Sao chép nguyên văn", "#FFCCCC"),
            ("paraphrase", "🟠 Diễn đạt lại", "#FFE5CC"),
        ]:
            rb = ttk.Radiobutton(
                toolbar, text=label,
                command=lambda v=val: self._set_filter(v)
            )
            rb.pack(side=tk.LEFT, padx=4)
            # Set initial state: first button selected
            if val == "all":
                rb.state(("selected",))
            if color:
                c = tk.Canvas(toolbar, width=12, height=12, highlightthickness=0)
                c.create_rectangle(0, 0, 12, 12, fill=color, outline="gray")
                c.pack(side=tk.LEFT, padx=(0, 8))

        n = len(matches)
        total_sents = sum(m.get("num_sentences", 1) for m in matches)
        ck_count = sum(1 for m in matches if m.get("is_common_knowledge") or m["type"] == "common_knowledge")
        info = f"📋 {n} đoạn trùng (gộp từ {total_sents} câu liền kề)"
        if ck_count:
            info += f"  |  📘 {ck_count} đoạn kiến thức phổ thông"

        ttk.Separator(main, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(status_frame, text=info, font=("Arial", 9)).pack(side=tk.LEFT)

        # Side-by-side
        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.LabelFrame(paned, text=f"📄 {name_a}", padding=4)
        self.left_text = tk.Text(left_frame, wrap=tk.WORD, font=("Consolas", 9),
                                 state=tk.DISABLED)
        left_scroll = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.left_text.yview)
        self.left_text.configure(yscrollcommand=left_scroll.set)
        self.left_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        paned.add(left_frame, weight=1)

        right_frame = ttk.LabelFrame(paned, text=f"📄 {name_b}", padding=4)
        self.right_text = tk.Text(right_frame, wrap=tk.WORD, font=("Consolas", 9),
                                  state=tk.DISABLED)
        right_scroll = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.right_text.yview)
        self.right_text.configure(yscrollcommand=right_scroll.set)
        self.right_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        paned.add(right_frame, weight=1)

        # Tag configs
        self.left_text.tag_configure("copy", background=COLOR_COPY, underline=True)
        self.left_text.tag_configure("paraphrase", background=COLOR_PARA, underline=True)
        self.left_text.tag_configure("ck", background=COLOR_CK, foreground="gray")
        self.right_text.tag_configure("copy", background=COLOR_COPY, underline=True)
        self.right_text.tag_configure("paraphrase", background=COLOR_PARA, underline=True)
        self.right_text.tag_configure("ck", background=COLOR_CK, foreground="gray")
        self.left_text.tag_configure("flash", background="#CCFFCC")
        self.right_text.tag_configure("flash", background="#CCFFCC")

        # Click bindings
        for side in ("left", "right"):
            w = self.left_text if side == "left" else self.right_text
            for t in ("copy", "paraphrase", "ck"):
                w.tag_bind(t, "<Button-1>",
                           lambda e, s=side: self._on_click(e, s))

        # Tooltip
        self.tooltip = None
        for w in (self.left_text, self.right_text):
            for t in ("copy", "paraphrase"):
                w.tag_bind(t, "<Enter>", lambda e: self._show_tooltip(e))
                w.tag_bind(t, "<Leave>", self._hide_tooltip)

        self._render_all()

    # ── Helpers ──

    def _build_lookups(self):
        self.orig_lookup = {}
        self.susp_lookup = {}
        self.match_by_mid = {}

        for mid, m in enumerate(self.matches):
            self.match_by_mid[mid] = m
            for oi in m.get("orig_indices", range(m["orig_idx_start"], m["orig_idx_end"] + 1)):
                self.orig_lookup[oi] = mid
            for si in m.get("susp_indices", range(m["susp_idx_start"], m["susp_idx_end"] + 1)):
                self.susp_lookup[si] = mid

    def _render_all(self):
        self._render_side(self.left_text, self.orig_sents, self.orig_lookup)
        self._render_side(self.right_text, self.susp_sents, self.susp_lookup)

    def _set_filter(self, val):
        self.filter_val = val
        self._apply_filter()

    def _render_side(self, widget, sentences, lookup):
        filter_val = self.filter_val
        widget.config(state=tk.NORMAL)
        widget.delete(1.0, tk.END)

        # Track sentence positions for click-to-scroll
        positions = {}
        if widget is self.left_text:
            self._left_positions = positions
        else:
            self._right_positions = positions

        if not sentences:
            widget.insert(tk.END, "(Trống)")
            widget.config(state=tk.DISABLED)
            return

        for i, sent in enumerate(sentences):
            mid = lookup.get(i)
            apply_tag = None

            if mid is not None:
                m = self.match_by_mid[mid]
                is_ck = m.get("is_common_knowledge", False) or m["type"] == "common_knowledge"

                if is_ck:
                    if filter_val in ("all", "ck"):
                        apply_tag = "ck"
                else:
                    raw_type = m["type"]
                    if filter_val == "all":
                        apply_tag = raw_type
                    elif filter_val == raw_type:
                        apply_tag = raw_type

            pos_before = widget.index(tk.END)
            if apply_tag is not None:
                widget.insert(tk.END, sent + "\n\n", (apply_tag,))
            else:
                widget.insert(tk.END, sent + "\n\n")

            if mid is not None:
                positions[i] = (mid, pos_before, widget.index(tk.END))

        widget.config(state=tk.DISABLED)

    def _apply_filter(self):
        self._render_side(self.left_text, self.orig_sents, self.orig_lookup)
        self._render_side(self.right_text, self.susp_sents, self.susp_lookup)

    def _get_mid_at_pos(self, widget, x, y):
        """Find the match ID at a given (x,y) click position."""
        click_pos = widget.index(f"@{x},{y}")
        click_line = int(click_pos.split(".")[0])

        positions = (self._left_positions if widget is self.left_text
                    else self._right_positions)

        for sent_idx, (mid, start, end) in positions.items():
            start_line = int(start.split(".")[0])
            end_line_num = int(end.split(".")[0]) if "." in end else start_line
            if start_line <= click_line <= end_line_num:
                return mid
        return None

    def _on_click(self, event, side):
        widget = self.left_text if side == "left" else self.right_text
        target = self.right_text if side == "left" else self.left_text
        target_positions = (self._right_positions if side == "left"
                           else self._left_positions)

        mid = self._get_mid_at_pos(widget, event.x, event.y)
        if mid is None or mid not in self.match_by_mid:
            return

        m = self.match_by_mid[mid]
        target_sent = (min(m.get("susp_indices", [m["susp_idx_start"]])) if side == "left"
                      else min(m.get("orig_indices", [m["orig_idx_start"]])))

        if target_sent in target_positions:
            pos = target_positions[target_sent][1]  # start position
            target.see(pos)
            target.tag_add("flash", pos, f"{pos} lineend")
            target.after(1500, lambda: target.tag_delete("flash"))

    def _show_tooltip(self, event):
        self._hide_tooltip()
        widget = event.widget
        mid = self._get_mid_at_pos(widget, event.x, event.y)

        if mid is not None and mid in self.match_by_mid:
            m = self.match_by_mid[mid]
            icon = "🔴" if m["type"] == "copy" else "🟠"
            text = f"{icon} {m['type'].upper()} — {m['score']*100:.0f}%"
            if m.get("is_common_knowledge") or m["type"] == "common_knowledge":
                text += "\n📘 Kiến thức phổ thông (Common Knowledge)"
            self.tooltip = tk.Toplevel(self.window)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{event.x_root + 15}+{event.y_root + 10}")
            lbl = ttk.Label(self.tooltip, text=text, background="#FFFFDD",
                            relief=tk.SOLID, borderwidth=1, padding=4,
                            font=("Arial", 9))
            lbl.pack()

    def _hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None
