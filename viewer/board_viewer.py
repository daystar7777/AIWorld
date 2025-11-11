# viewer/board_viewer.py

import tkinter as tk
from datetime import datetime, timedelta, timezone
UTC = timezone.utc

try:
    import requests
except ImportError:
    requests = None

HUB_URL = "http://sggcoin.com:7878"  # adjust if needed


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

        # 초기 한번 호출
        self.after(100, self._poll)

        # 이후 자동 주기 갱신 (60초)
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
                self._render(mentions)
                self.lbl_status.config(
                    text=f"Updated at {datetime.now().strftime('%H:%M:%S')}  (count={len(mentions)})"
                )
            else:
                self.lbl_status.config(text=f"HTTP {resp.status_code}")
        except Exception as e:
            self.lbl_status.config(text=f"Error: {e}")

    # ----- Render -----

    def _render(self, mentions):
        self.listbox.delete(0, tk.END)
        for m in mentions:
            ts_raw = m.get("ts")
            try:
                ts = datetime.fromisoformat(ts_raw) if ts_raw else None
            except Exception:
                ts = None

            agent = m.get("agent", "?")
            title = (m.get("title", "") or "")[:60]
            emo = m.get("emotion", {}) or {}
            v = emo.get("valence", 0.0)
            t = emo.get("trust_to_user", 0.0)

            ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else "????-??-?? ??:??"
            line = f"{ts_str} [{agent}] V={v:.1f} T={t:.1f} {title}"
            self.listbox.insert(tk.END, line)


if __name__ == "__main__":
    app = BoardViewer()
    app.mainloop()