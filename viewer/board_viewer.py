# viewer/board_viewer.py

import tkinter as tk
from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    requests = None

HUB_URL = "http://localhost:5000"  # adjust if needed


class BoardViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Agents Network Board (last 24h)")
        self.geometry("900x500")
        self.configure(bg="#111111")

        top = tk.Frame(self, bg="#222222")
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 6))

        tk.Label(
            top,
            text="Network Board - last 24 hours",
            fg="#FFFFFF",
            bg="#222222",
            font=("Segoe UI", 11, "bold"),
        ).pack(side=tk.LEFT, padx=8, pady=4)

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

        bottom = tk.Frame(self, bg="#111111")
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 8))

        self.lbl_status = tk.Label(
            bottom,
            text="Loading...",
            fg="#777777",
            bg="#111111",
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.lbl_status.pack(fill=tk.X)

        self._poll()

    def _poll(self):
        if not requests:
            self.lbl_status.config(text="Install 'requests' to use this viewer.")
            return

        try:
            since = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            resp = requests.get(
                f"{HUB_URL.rstrip('/')}/mentions",
                params={"since": since},
                timeout=5,
            )
            if resp.ok:
                mentions = resp.json()
                self._render(mentions)
                self.lbl_status.config(
                    text=f"Updated at {datetime.now().strftime('%H:%M:%S')} (count={len(mentions)})"
                )
            else:
                self.lbl_status.config(text=f"HTTP {resp.status_code}")
        except Exception as e:
            self.lbl_status.config(text=f"Error: {e}")

        self.after(60000, self._poll)

    def _render(self, mentions):
        self.listbox.delete(0, tk.END)
        for m in mentions:
            try:
                ts = datetime.fromisoformat(m["ts"])
            except Exception:
                continue
            agent = m.get("agent", "?")
            title = (m.get("title", "") or "")[:60]
            emo = m.get("emotion", {})
            v = emo.get("valence", 0.0)
            t = emo.get("trust_to_user", 0.0)
            line = f"{ts.strftime('%Y-%m-%d %H:%M')} [{agent}] V={v:.1f} T={t:.1f} {title}"
            self.listbox.insert(tk.END, line)


if __name__ == "__main__":
    app = BoardViewer()
    app.mainloop()
