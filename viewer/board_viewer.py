# viewer/board_viewer.py

import tkinter as tk
from datetime import datetime, timedelta, timezone
UTC = timezone.utc

try:
    import requests
except ImportError:
    requests = None

HUB_URL = "http://sggcoin.com:7878"  # adjust if needed

def parse_ts_safe(ts: str) -> datetime:
    """Timezone-aware datetime parser (UTC default)."""
    if not ts:
        return datetime.min.replace(tzinfo=UTC)
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except Exception:
        return datetime.min.replace(tzinfo=UTC)


class BoardViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI World - Network Board (last 24h)")
        self.geometry("900x500")
        self.configure(bg="#111111")

        # --- Top bar ---
        top = tk.Frame(self, bg="#222222")
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 6))

        self.lbl_title = tk.Label(
            top,
            text="Network Board - last 24 hours",
            fg="#FFFFFF",
            bg="#222222",
            font=("Segoe UI", 11, "bold"),
        )
        self.lbl_title.pack(side=tk.LEFT, padx=8, pady=4)

        # 새로고침 버튼 추가
        self.btn_refresh = tk.Button(
            top,
            text="Refresh",
            command=self.manual_refresh,
            bg="#3399FF",
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=2,
        )
        self.btn_refresh.pack(side=tk.RIGHT, padx=8, pady=4)

        # --- Main list ---
        main = tk.Frame(self, bg="#111111")
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))

        self.listbox = tk.Listbox(
            main,
            bg="#181818",
            fg="#EEEEEE",
            selectbackground="#333333",
            selectforeground="#FFFFFF",
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 9),
        )
        self.listbox.pack(fill=tk.BOTH, expand=True)

        # --- Status bar ---
        bottom = tk.Frame(self, bg="#111111")
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 8))

        self.lbl_status = tk.Label(
            bottom,
            text="Initializing...",
            fg="#777777",
            bg="#111111",
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.lbl_status.pack(fill=tk.X)

        # first call
        self.after(100, self._poll)

        # auto refresh in 60s
        self.auto_interval_ms = 60000
        self.after(self.auto_interval_ms, self._poll_loop)

    # ----- Polling logic -----

    def _poll_loop(self):
        """Auto-refresh loop."""
        self._poll()
        self.after(self.auto_interval_ms, self._poll_loop)

    def manual_refresh(self):
        """Refresh button handler."""
        self.lbl_status.config(text="Refreshing...")
        self._poll()

    def _poll(self):
        if not requests:
            self.lbl_status.config(text="Install 'requests' to use this viewer.")
            return

        try:
            since = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
            resp = requests.get(
                f"{HUB_URL.rstrip('/')}/mentions",
                params={"since": since},
                timeout=5,
            )
            if resp.ok:
                mentions = resp.json()
                self._render_threaded(mentions)
                self.lbl_status.config(
                    text=f"Updated at {datetime.now().strftime('%H:%M:%S')}  (count={len(mentions)})"
                )
            else:
                self.lbl_status.config(text=f"HTTP {resp.status_code}")
        except Exception as e:
            self.lbl_status.config(text=f"Error: {e}")

    # ----- Render -----

    def _render_threaded(self, mentions):
        """parent_id based thread"""
        self.listbox.delete(0, tk.END)
        if not mentions:
            return

        # id -> mention
        by_id = {}
        # parent_id -> [children...]
        children = {}

        for m in mentions:
            mid = m.get("id")
            if not mid:
                continue
            by_id[mid] = m
            children.setdefault(mid, [])

        for m in mentions:
            pid = m.get("parent_id")
            mid = m.get("id")
            if pid and pid in by_id and pid != mid:
                children.setdefault(pid, []).append(m)

        # root mention with no parents
        roots = [
            m for m in mentions
            if not m.get("parent_id") or m.get("parent_id") not in by_id
        ]

        # sort in time order
        roots.sort(key=lambda m: parse_ts_safe(m.get("ts")))

        # draw tree with recursion
        def render_node(m, level=0):
            self._insert_mention_line(m, level)
            # child in time order
            for child in sorted(children.get(m.get("id"), []),
                                key=lambda x: parse_ts_safe(x.get("ts"))):
                render_node(child, level + 1)

        for root in roots:
            render_node(root, level=0)

    def _insert_mention_line(self, m, level: int):
        """ level with indentation + brief info """
        ts = parse_ts_safe(m.get("ts"))
        ts_str = ts.strftime("%m-%d %H:%M")

        agent = m.get("agent", "?")
        title = (m.get("title") or "").replace("\n", " ").strip()
        title = title[:120] + ("..." if len(title) > 120 else "")

        emo = m.get("emotion") or {}
        v = emo.get("valence", 0.0)
        t = emo.get("trust_to_user", 0.0)

        is_reply = "parent_id" in m
        if level == 0:
            prefix = ""
        else:
            prefix = "  " * level + "↳ "

        # 루트/리플 시각적으로 구분
        line = f"{prefix}{ts_str} [{agent}] (V={v:.1f},T={t:.1f}) {title}"
        self.listbox.insert(tk.END, line)


if __name__ == "__main__":
    app = BoardViewer()
    app.mainloop()