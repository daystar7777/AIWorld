# viewer/board_viewer.py

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
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


def wrap_text(text: str, max_width: int = 80) -> list[str]:
    """텍스트를 지정된 너비로 자동 줄바꿈."""
    if not text:
        return []
    
    # 텍스트가 이미 max_width보다 짧으면 그대로 반환
    if len(text) <= max_width:
        return [text]
    
    lines = []
    words = text.split()
    current_line = []
    
    for word in words:
        # 현재 줄에 단어를 추가했을 때의 길이 계산 (단어 + 공백)
        test_line = ' '.join(current_line + [word])
        
        if len(test_line) <= max_width:
            current_line.append(word)
        else:
            # 현재 줄을 저장하고 새 줄 시작
            if current_line:
                lines.append(' '.join(current_line))
            # 단어 자체가 max_width보다 길면 강제로 자름
            if len(word) > max_width:
                # 긴 단어를 여러 줄로 나눔
                while len(word) > max_width:
                    lines.append(word[:max_width])
                    word = word[max_width:]
                current_line = [word] if word else []
            else:
                current_line = [word]
    
    # 마지막 줄 추가
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines if lines else [text]


def wrap_text_pixels(text: str, max_px: int, font_obj: tkfont.Font, prefix_text: str = "") -> list[str]:
    """Wrap text so that each line's pixel width (with prefix) <= max_px.
    Breaks on spaces preferentially; falls back to character-level splits for long words."""
    if text is None:
        return []
    if max_px <= 0:
        return [prefix_text + text]

    words = text.split()
    lines: list[str] = []
    current = prefix_text

    def measure(s: str) -> int:
        return font_obj.measure(s)

    def flush():
        nonlocal current
        if current:
            lines.append(current)
        current = prefix_text

    i = 0
    while i < len(words):
        word = words[i]
        sep = " " if current.strip() and not current.endswith(" ") else ""
        test = current + sep + word
        if measure(test) <= max_px:
            current = test
            i += 1
            continue
        # If current only has prefix and even single word doesn't fit, split the word
        if not current.strip():
            # split word by characters so that it fits
            remaining = word
            while remaining:
                # find largest prefix of remaining that fits
                lo, hi = 1, len(remaining)
                fit = 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    chunk = prefix_text + remaining[:mid]
                    if measure(chunk) <= max_px:
                        fit = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                chunk = remaining[:fit]
                lines.append(prefix_text + chunk)
                remaining = remaining[fit:]
            current = prefix_text
            i += 1
        else:
            # flush current and retry word on next line
            flush()
    if current and current != prefix_text:
        lines.append(current)
    return lines


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

        # 폰트 크기 조절 버튼들 (Refresh 버튼 왼쪽에 배치)
        self.font_size = 9  # 초기 폰트 크기
        
        # 폰트 크기 조절 버튼을 담을 프레임
        font_control_frame = tk.Frame(top, bg="#222222")
        font_control_frame.pack(side=tk.RIGHT, padx=(0, 8), pady=4)
        
        self.btn_font_minus = tk.Button(
            font_control_frame,
            text="-",
            command=self.decrease_font,
            bg="#CCCCCC",
            fg="#000000",
            activebackground="#DDDDDD",
            activeforeground="#000000",
            highlightthickness=0,
            font=("Segoe UI", 14, "bold"),
            relief=tk.FLAT,
            borderwidth=0,
            width=3,
            padx=4,
            pady=2,
        )
        self.btn_font_minus.pack(side=tk.LEFT, padx=(0, 2))
        
        self.btn_font_plus = tk.Button(
            font_control_frame,
            text="+",
            command=self.increase_font,
            bg="#CCCCCC",
            fg="#000000",
            activebackground="#DDDDDD",
            activeforeground="#000000",
            highlightthickness=0,
            font=("Segoe UI", 14, "bold"),
            relief=tk.FLAT,
            borderwidth=0,
            width=3,
            padx=4,
            pady=2,
        )
        self.btn_font_plus.pack(side=tk.LEFT, padx=(0, 8))
        
        # 새로고침 버튼 추가
        self.btn_refresh = tk.Button(
            top,
            text="Refresh",
            command=self.manual_refresh,
            bg="#CCCCCC",
            fg="#000000",
            activebackground="#DDDDDD",
            activeforeground="#000000",
            highlightthickness=0,
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=2,
        )
        self.btn_refresh.pack(side=tk.RIGHT, padx=8, pady=4)

        # --- Main area: Tree only ---
        main = tk.Frame(self, bg="#111111")
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))

        # Tree view
        tree_frame = tk.Frame(main, bg="#111111")
        tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Treeview 스타일 설정
        style = ttk.Style()
        # macOS에서는 aqua 테마가 트리 아이콘을 더 잘 표시함
        # 트리 아이콘이 보이도록 default 테마 사용
        try:
            available_themes = style.theme_names()
            # default 테마가 트리 아이콘을 더 명확하게 표시함
            style.theme_use("default")
        except:
            style.theme_use("default")
        
        # 트리 아이콘이 확실히 보이도록 배경색을 조금 더 밝게 설정
        style.configure("Treeview",
                       background="#2A2A2A",  # 더 밝은 배경으로 변경
                       foreground="#EEEEEE",
                       fieldbackground="#2A2A2A",
                       font=("Segoe UI", self.font_size),
                       indent=25,  # 트리 들여쓰기 (트리 아이콘 공간 포함)
                       rowheight=int(self.font_size * 1.5))  # 행 높이 설정 (폰트 크기에 비례)
        style.configure("Treeview.Heading",
                       background="#333333",
                       foreground="#FFFFFF",
                       font=("Segoe UI", self.font_size, "bold"))
        style.map("Treeview",
                 background=[("selected", "#444444")],
                 foreground=[("selected", "#FFFFFF")])
        
        # 트리뷰 항목에 패딩 추가 (좌우 여백) - 글자가 프레임 외곽선에 잘리지 않도록
        try:
            # Treeview.Cell 스타일로 셀 패딩 설정
            style.configure("Treeview.Cell",
                           padding=(10, 4))  # 좌우 10px, 상하 4px 패딩
        except:
            pass
        
        # 트리 아이콘 영역의 배경색을 더 밝게 설정하여 아이콘이 보이도록
        try:
            style.configure("Treeview.Item",
                           foreground="#EEEEEE",
                           background="#2A2A2A")
            # 트리 아이콘 색상을 밝게 설정
            style.map("Treeview.Item",
                     background=[("selected", "#444444")],
                     foreground=[("selected", "#FFFFFF")])
        except:
            pass

        # 트리뷰 생성 - show="tree"로 트리 아이콘 표시
        # show="tree"는 트리 아이콘만 표시, show="tree headings"는 헤더도 표시
        self.tree = ttk.Treeview(tree_frame, columns=("info",), show="tree", style="Treeview")
        # 스크롤바 공간을 고려하여 컬럼 너비 설정
        self.tree.column("#0", width=800, anchor="w", minwidth=200, stretch=True)
        self.tree.column("info", width=0, stretch=False)  # 숨김
        
        # 트리 아이콘이 확실히 보이도록 설정
        # ttk.Treeview는 기본적으로 자식이 있는 항목에 [+] 아이콘을 표시함
        # 자식이 없는 항목은 트리 아이콘이 없음

        # Scrollbar for tree
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        # 스크롤바와 트리뷰 사이에 충분한 여백 추가하여 텍스트가 가려지지 않도록
        # 스크롤바를 먼저 배치하고 트리뷰를 배치
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))
        # 트리뷰에 오른쪽 패딩 추가하여 스크롤바와 충분한 거리 확보
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 30))
        # 트리 들여쓰기 픽셀(접힘 아이콘 영역 포함) 기본값
        self.tree_indent_px = 25
        
        # 트리뷰 업데이트 후 컬럼 너비 조정 함수
        def adjust_column_width(event=None):
            # 트리뷰의 실제 너비를 가져와서 스크롤바 너비와 충분한 여백/패딩을 빼서 설정
            tree_width = self.tree.winfo_width()
            if tree_width > 1:  # 초기화 완료 확인
                scrollbar_width = 20  # 스크롤바 대략적 너비
                # 폰트 크기에 비례한 패딩 계산 (더 큰 여백)
                padding = max(20, int(self.font_size * 2.5))  # 폰트 크기에 비례한 패딩 (더 크게)
                margin = 50  # 스크롤바와의 여백 (더 크게)
                safe_margin = max(10, int(self.font_size * 0.5))  # 폰트 크기에 따른 추가 안전 마진
                total_margin = scrollbar_width + padding + margin + safe_margin
                self.tree.column("#0", width=max(200, tree_width - total_margin))
        
        # 트리뷰 크기 변경 시 컬럼 너비 자동 조정
        self.tree.bind("<Configure>", adjust_column_width)

        # Tree open event - 항목을 열 때 내용 표시
        self.tree.bind("<<TreeviewOpen>>", self.on_tree_open)
        self.tree.bind("<<TreeviewClose>>", self.on_tree_close)
        
        # 태그 스타일 설정 (판넬 배경 효과)
        panel_bg = "#3A3A3A"  # 옅은 회색 패널 배경
        self.tree.tag_configure("content", background=panel_bg, foreground="#EEEEEE")
        self.tree.tag_configure("header", background=panel_bg, foreground="#FFFFFF")
        # 다음 항목과 구분을 위한 공백 라인 스타일
        self.tree.tag_configure("spacer", background="#2A2A2A", foreground="#2A2A2A")
        
        # Initialize mentions storage
        self.mentions_by_id = {}

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
    
    def increase_font(self):
        """폰트 크기 증가."""
        self.font_size = min(20, self.font_size + 1)  # 최대 20
        self._update_font()
    
    def decrease_font(self):
        """폰트 크기 감소."""
        self.font_size = max(6, self.font_size - 1)  # 최소 6
        self._update_font()
    
    def _update_font(self):
        """트리뷰 폰트 업데이트."""
        style = ttk.Style()
        # 폰트 크기에 비례하여 행 높이와 패딩 조정
        row_height = int(self.font_size * 1.5)
        padding_horizontal = max(8, int(self.font_size * 0.8))  # 폰트 크기에 비례한 좌우 패딩
        padding_vertical = max(4, int(self.font_size * 0.3))  # 폰트 크기에 비례한 상하 패딩
        
        style.configure("Treeview", 
                       font=("Segoe UI", self.font_size),
                       rowheight=row_height)  # 행 높이 업데이트
        style.configure("Treeview.Heading", font=("Segoe UI", self.font_size, "bold"))
        
        # 패딩도 폰트 크기에 맞게 업데이트
        try:
            style.configure("Treeview.Cell", 
                           padding=(padding_horizontal, padding_vertical))
        except:
            pass
        
        # 컬럼 너비도 다시 조정 (폰트 크기 변경으로 인한 텍스트 너비 변화 반영)
        self.after_idle(self._adjust_column_width_delayed)
        # 열려있는 항목들의 내용도 다시 렌더링 (줄바꿈 재계산)
        self.after_idle(self._refresh_open_items)
    
    def _adjust_column_width_delayed(self):
        """지연된 컬럼 너비 조정 (폰트 업데이트 후 호출)"""
        tree_width = self.tree.winfo_width()
        if tree_width > 1:
            scrollbar_width = 20
            # 폰트 크기에 비례한 패딩 계산 (더 큰 여백)
            padding = max(20, int(self.font_size * 2.5))  # 폰트 크기에 비례한 패딩 (더 크게)
            margin = 50  # 스크롤바와의 여백 (더 크게)
            safe_margin = max(10, int(self.font_size * 0.5))  # 폰트 크기에 따른 추가 안전 마진
            total_margin = scrollbar_width + padding + margin + safe_margin
            self.tree.column("#0", width=max(200, tree_width - total_margin))

    def _get_item_depth(self, item: str) -> int:
        depth = 0
        parent = self.tree.parent(item)
        while parent:
            depth += 1
            parent = self.tree.parent(parent)
        return depth

    def _refresh_open_items(self):
        """열려있는 모든 항목의 내용을 다시 렌더링 (폰트 크기 변경 시)"""
        def refresh_recursive(item):
            """재귀적으로 모든 열려있는 항목 찾기"""
            if not item:
                return
            
            # 현재 항목이 열려있고 내용이 표시되어 있는지 확인
            if self.tree.item(item, "open"):
                values = self.tree.item(item, "values")
                if values and values[0] != "__dummy__" and values[0] != "__content__":
                    # 내용 항목들 찾기
                    children = self.tree.get_children(item)
                    has_content = False
                    for child in children:
                        child_values = self.tree.item(child, "values")
                        if child_values and child_values[0] == "__content__":
                            has_content = True
                            break
                    
                    # 내용이 있으면 삭제하고 다시 렌더링
                    if has_content:
                        # 내용 항목들 삭제
                        for child in list(children):
                            child_values = self.tree.item(child, "values")
                            if child_values and child_values[0] == "__content__":
                                self.tree.delete(child)
                            elif child_values and child_values[0] == "__dummy__":
                                self.tree.delete(child)
                        
                        # 다시 렌더링
                        mention_id = values[0]
                        mention = self.mentions_by_id.get(mention_id)
                        if mention:
                            self._render_mention_content(item, mention)
            
            # 자식 항목들도 재귀적으로 확인
            for child in self.tree.get_children(item):
                refresh_recursive(child)
        
        # 루트 항목들부터 시작
        for root_item in self.tree.get_children():
            refresh_recursive(root_item)
    
    def _render_mention_content(self, item, mention):
        """mention 내용을 트리 항목에 렌더링"""
        # 전체 내용 구성
        ts = parse_ts_safe(mention.get("ts"))
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        agent = mention.get("agent", "?")
        title = mention.get("title", "")
        text = mention.get("text", "")
        emo = mention.get("emotion", {})
        # 트리뷰 실제 컬럼 픽셀 너비 기준 가용 픽셀 계산
        try:
            col_width_px = int(self.tree.column("#0", option="width"))
        except Exception:
            col_width_px = self.tree.winfo_width()
        depth = self._get_item_depth(item) + 1  # content는 한 단계 더 들어감
        indent_base = getattr(self, "tree_indent_px", 25)
        indent_px = indent_base * depth
        # 내부 패딩 + 안전 마진
        padding_px = max(20, int(self.font_size * 2.0))
        safe_px = max(10, int(self.font_size * 0.5))
        available_px = max(50, col_width_px - indent_px - padding_px - safe_px)
        # 폰트 객체
        content_font = tkfont.Font(family="Segoe UI", size=self.font_size)
        header_font = tkfont.Font(family="Segoe UI", size=self.font_size, weight="bold")
        
        # 내용을 여러 줄로 나누어 표시
        content_lines = []
        
        # Agent 정보 (픽셀 기반 줄바꿈)
        agent_lines = wrap_text_pixels(f"Agent: {agent}", available_px, header_font)
        content_lines.extend(agent_lines)
        
        # Time 정보
        time_lines = wrap_text_pixels(f"Time: {ts_str}", available_px, content_font)
        content_lines.extend(time_lines)
        
        # Title 정보 (Content와 중복되지 않도록 간결한 제목만 표시)
        display_title = (title or "").strip()
        if text:
            # 제목에 본문이 포함되어 있으면 제거
            if display_title.endswith(text):
                display_title = display_title[: -len(text)].rstrip()
            elif text in display_title:
                display_title = display_title.replace(text, "").strip()
        # 콜론 구분자가 있으면 앞부분만 사용
        if ":" in display_title:
            display_title = display_title.split(":", 1)[0].strip()
        # 최종 Title 라인 생성
        title_lines = wrap_text_pixels(f"Title: {display_title}", available_px, content_font)
        content_lines.extend(title_lines)

        content_lines.append("")
        content_lines.append("Emotion:")
        content_lines.append(f"  Valence: {emo.get('valence', 0.0):.2f}")
        content_lines.append(f"  Arousal: {emo.get('arousal', 0.0):.2f}")
        content_lines.append(f"  Curiosity: {emo.get('curiosity', 0.0):.2f}")
        content_lines.append(f"  Anxiety: {emo.get('anxiety', 0.0):.2f}")
        content_lines.append(f"  Trust: {emo.get('trust_to_user', 0.0):.2f}")
        content_lines.append("")
        content_lines.append("Content:")
        
        # 텍스트를 여러 줄로 나누어 표시 (픽셀 기반 줄바꿈)
        for original_line in text.split('\n'):
            if original_line.strip():
                wrapped_lines = wrap_text_pixels(original_line, available_px, content_font, prefix_text="  ")
                content_lines.extend(wrapped_lines)
            else:
                content_lines.append("")
        
        if mention.get("parent_id"):
            content_lines.append("")
            reply_lines = wrap_text_pixels(f"[Reply to: {mention.get('parent_id')}]", available_px, content_font)
            content_lines.extend(reply_lines)
        
        # 각 줄을 자식 항목으로 추가 (판넬 배경 효과를 위해)
        for i, line in enumerate(content_lines):
            # 첫 번째 줄은 헤더 태그, 나머지는 content 태그
            tag = "header" if i == 0 else "content"
            self.tree.insert(
                item,
                "end",
                text=line,
                values=("__content__",),
                tags=(tag,)
            )
        # 다음 항목과의 간격을 위해 공백 라인 추가
        self.tree.insert(
            item,
            "end",
            text="",
            values=("__content__",),
            tags=("spacer",)
        )

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
        """parent_id based thread with treeview"""
        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if not mentions:
            return

        # Store mentions by id for quick lookup
        self.mentions_by_id = {}
        
        # id -> mention
        by_id = {}
        # parent_id -> [children...]
        children = {}

        for m in mentions:
            mid = m.get("id")
            if not mid:
                continue
            by_id[mid] = m
            self.mentions_by_id[mid] = m
            
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

        # sort in reverse time order (newest first)
        roots.sort(key=lambda m: parse_ts_safe(m.get("ts")), reverse=True)

        # draw tree with recursion
        def render_node(m, parent=""):
            item_id = self._insert_tree_item(m, parent)
            child_list = sorted(children.get(m.get("id"), []),
                              key=lambda x: parse_ts_safe(x.get("ts")), reverse=True)
            
            # 자식이 있으면 재귀적으로 렌더링
            if child_list:
                for child in child_list:
                    render_node(child, item_id)
            else:
                # 자식이 없어도 접힘 표시를 위해 더미 자식 항목 추가
                dummy_id = self.tree.insert(item_id, "end", text="", values=("__dummy__",))
                self.tree.item(dummy_id, open=False)

        for root in roots:
            render_node(root)

    def _insert_tree_item(self, m, parent=""):
        """트리뷰에 항목 추가하고 ID 반환"""
        ts = parse_ts_safe(m.get("ts"))
        ts_str = ts.strftime("%m-%d %H:%M")

        agent = m.get("agent", "?")
        title = (m.get("title") or "").replace("\n", " ").strip()
        if len(title) > 60:
            title = title[:60] + "..."

        emo = m.get("emotion") or {}
        v = emo.get("valence", 0.0)
        t = emo.get("trust_to_user", 0.0)

        # 트리 항목 텍스트
        display_text = f"{ts_str} [{agent}] (V={v:.1f},T={t:.1f}) {title}"
        
        # 트리 항목 추가 (parent가 있으면 서브 항목, 없으면 루트)
        # open=True로 하면 기본적으로 열려있음
        item_id = self.tree.insert(
            parent,
            "end",
            text=display_text,
            values=(m.get("id"),),  # mention ID를 values에 저장
            open=False  # 기본적으로 닫혀있음 (자식이 있으면 열 수 있음)
        )
        return item_id

    def on_tree_open(self, event):
        """트리 항목을 열 때 내용을 자식 항목으로 표시"""
        # 열린 항목 찾기
        item = event.widget.focus()
        if not item:
            return
        
        values = self.tree.item(item, "values")
        if not values:
            return
        
        mention_id = values[0]
        
        # 더미 항목은 무시
        if mention_id == "__dummy__":
            return
        
        mention = self.mentions_by_id.get(mention_id)
        if not mention:
            return
        
        # 이미 내용이 표시되었는지 확인 (자식 항목 중 "__content__" 태그가 있는지 확인)
        children = self.tree.get_children(item)
        has_content = False
        dummy_child = None
        for child in children:
            child_values = self.tree.item(child, "values")
            if child_values:
                if child_values[0] == "__content__":
                    has_content = True
                elif child_values[0] == "__dummy__":
                    dummy_child = child
        
        if has_content:
            return  # 이미 내용이 표시됨
        
        # 더미 항목이 있으면 제거
        if dummy_child:
            self.tree.delete(dummy_child)
        
        # 내용 렌더링 (공통 함수 사용)
        self._render_mention_content(item, mention)

    def on_tree_close(self, event):
        """트리 항목을 닫을 때 접두사 복원 및 내용 정리"""
        item = event.widget.focus()
        if not item:
            return
        # 내용(children with __content__) 제거, 없으면 유지
        children = self.tree.get_children(item)
        has_real_child = False
        for ch in list(children):
            vals = self.tree.item(ch, "values")
            if vals and vals[0] == "__content__":
                self.tree.delete(ch)
            else:
                has_real_child = True
        # 최소 더미 자식 보장
        if not has_real_child:
            self.tree.insert(item, "end", text="", values=("__dummy__",))


if __name__ == "__main__":
    app = BoardViewer()
    app.mainloop()