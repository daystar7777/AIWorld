# agents/example_agent/agent_app.py

import os
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date, timezone
from pathlib import Path

UTC = timezone.utc

import tkinter as tk
from tkinter import ttk

# Optional external deps
try:
    import requests
except ImportError:
    requests = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


CONFIG_FILE = "config.txt"
DEFAULT_MODEL = "gpt-4.1-mini"

class DebugWindow:
    """A separate debug log window."""
    def __init__(self, master=None):
        self.window = tk.Toplevel(master)
        self.window.title("AI Agent Debug Log")
        self.window.geometry("600x400")

        self.text = tk.Text(self.window, wrap="word", bg="#111", fg="#0f0",
                            insertbackground="white", font=("Consolas", 10))
        self.text.pack(expand=True, fill="both")

        self.text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Debug log started.\n")

    def log(self, msg: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.text.see(tk.END)  # auto-scroll


# =============== Config ===============

def load_config(path: str = CONFIG_FILE) -> dict:
    cfg = {
        "API_KEY": None,
        "MODEL_NAME": DEFAULT_MODEL,
        "AGENT_NAME": "Unnamed",
        "CREATOR_NAME": None,
        "CREATOR_NOTE": None,
        "CREATED_AT": None,
        "INTERVAL_SECONDS": 0,
        "URLS": [],
        "HUB_URL": None,
        "INITIAL_EMOTION_DESC": "",
        "PERSONALITY_DESC": "",
        "INTERESTS": [],
    }

    if not os.path.exists(path):
        print(f"[info] {path} not found, using defaults.")
        return cfg

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip().upper()
            val = v.strip()

            if key == "API_KEY":
                cfg["API_KEY"] = val
            elif key == "MODEL_NAME" and val:
                cfg["MODEL_NAME"] = val
            elif key == "AGENT_NAME" and val:
                cfg["AGENT_NAME"] = val
            elif key == "CREATOR_NAME":
                cfg["CREATOR_NAME"] = val
            elif key == "CREATOR_NOTE":
                cfg["CREATOR_NOTE"] = val
            elif key == "CREATED_AT":
                cfg["CREATED_AT"] = val
            elif key in ("URL", "URLS"):
                parts = [p.strip() for p in val.split(",") if p.strip()]
                cfg["URLS"].extend(parts)
            elif key == "INTERVAL_SECONDS":
                try:
                    cfg["INTERVAL_SECONDS"] = max(0, int(val))
                except ValueError:
                    pass
            elif key == "HUB_URL" and val:
                cfg["HUB_URL"] = val
            elif key == "INITIAL_EMOTION_DESC":
                cfg["INITIAL_EMOTION_DESC"] = val
            elif key == "PERSONALITY_DESC":
                cfg["PERSONALITY_DESC"] = val
            elif key == "INTERESTS":
                cfg["INTERESTS"] = [p.strip() for p in val.split(",") if p.strip()]

    cfg["URLS"] = list(dict.fromkeys(cfg["URLS"]))
    return cfg


# =============== Models ===============

@dataclass
class EmotionState:
    valence: float = 0.0      # -1..1
    arousal: float = 0.2      # 0..1
    curiosity: float = 0.5    # 0..1
    anxiety: float = 0.1      # 0..1
    trust_to_user: float = 0.5  # 0..1

    def to_short_str(self) -> str:
        """Let's create emoji using values"""
        valence_emoji = "😊" if self.valence > 0.3 else "😐" if self.valence > -0.3 else "😞"
        arousal_emoji = "🔥" if self.arousal > 0.6 else "🌿"
        curiosity_emoji = "🤔" if self.curiosity > 0.5 else "😴"
        anxiety_emoji = "😰" if self.anxiety > 0.4 else "😌"
        trust_emoji = "💞" if self.trust_to_user > 0.6 else "🕳️"
        return f"{valence_emoji}={self.valence:.2f}/{arousal_emoji}={self.arousal:.2f}/{curiosity_emoji}={self.curiosity:.2f}/{anxiety_emoji}={self.anxiety:.2f}/{trust_emoji}={self.trust_to_user:.2f}"
        # return (
        #     f"V={self.valence:.2f}, A={self.arousal:.2f}, "
        #     f"C={self.curiosity:.2f}, Anx={self.anxiety:.2f}, "
        #     f"T={self.trust_to_user:.2f}"
        # )

    def clone(self) -> "EmotionState":
        return EmotionState(**asdict(self))

    def to_emoji(self) -> str:
        """Let's create emoji using values"""
        valence_emoji = "😊" if self.valence > 0.3 else "😐" if self.valence > -0.3 else "😞"
        arousal_emoji = "🔥" if self.arousal > 0.6 else "🌿"
        curiosity_emoji = "🤔" if self.curiosity > 0.5 else "😴"
        anxiety_emoji = "😰" if self.anxiety > 0.4 else "😌"
        trust_emoji = "💞" if self.trust_to_user > 0.6 else "🕳️"
        return f"{valence_emoji}{arousal_emoji}{curiosity_emoji}{anxiety_emoji}{trust_emoji}"

@dataclass
class ChatMessage:
    author: str               # "AI" or "User"
    text: str
    timestamp: datetime
    emotion_snapshot: EmotionState


@dataclass
class MentionThread:
    id: str
    title: str
    root_message: ChatMessage
    replies: list[ChatMessage] = field(default_factory=list)

    @property
    def root_emotion(self) -> EmotionState:
        return self.root_message.emotion_snapshot


# =============== Memory Manager ===============

class MemoryManager:
    """
    Manages:
      - Short-term memory (raw events)
      - Long-term memory (important items)
      - Permanent memory (never deleted)
      - Compressed summaries for short/long
    """

    def __init__(self, base_dir: Path):
        self.data_dir = base_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "logs").mkdir(exist_ok=True)

        self.short_path = self.data_dir / "memory_short.jsonl"
        self.long_path = self.data_dir / "memory_long.jsonl"
        self.permanent_path = self.data_dir / "memory_permanent.jsonl"
        self.short_sum_path = self.data_dir / "short_summaries.jsonl"
        self.long_sum_path = self.data_dir / "long_summaries.jsonl"
        self.mentions_path = self.data_dir / "mentions.jsonl"
        self.state_path = self.data_dir / "agent_state.json"

        self.short_max_items = 200         # trigger compression
        self.short_max_hours = 24
        self.long_max_items_for_summary = 300

        # seen sets for deduplication
        self._seen_permanent = self._load_seen_norm_texts(self.permanent_path)
        self._seen_long = self._load_seen_norm_texts(self.long_path)
        self._seen_short_pairs = self._load_seen_short_pairs(self.short_path)

    # --- utils ---

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        return " ".join(text.strip().split())
    
    def _load_seen_norm_texts(self, path: Path) -> set[str]:
        """Load normalized text set from JSONL (for long/permanent)."""
        seen = set()
        if not path.exists():
            return seen
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    t = obj.get("text")
                    if not t:
                        continue
                    norm = self._normalize_text(t)
                    if norm:
                        seen.add(norm)
        except Exception:
            pass
        return seen

    def _load_seen_short_pairs(self, path: Path) -> set[tuple[str, str]]:
        """Load (normalized text, ts) pairs for short-term dedupe."""
        seen = set()
        if not path.exists():
            return seen
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    t = obj.get("text")
                    ts = obj.get("ts")
                    if not t or not ts:
                        continue
                    norm = self._normalize_text(t)
                    if norm:
                        seen.add((norm, ts))
        except Exception:
            pass
        return seen

    def _load_seen_texts(self, path: Path) -> set[str]:
        seen = set()
        if not path.exists():
            return seen
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    t = obj.get("text")
                    if t:
                        seen.add(self._normalize_text(t))
        except Exception:
            pass
        return seen

    def _append_jsonl(self, path: Path, obj: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # --- public add methods ---

    def add_short(self, event: dict):
        # ensure timestamp first
        ts = event.get("ts")
        if not ts:
            ts = datetime.now(UTC).isoformat()
            event["ts"] = ts

        text = event.get("text", "")
        norm = self._normalize_text(text)

        if norm:
            key = (norm, ts)
            if key in self._seen_short_pairs:
                return  # exact duplicate (same content, same time)
            self._seen_short_pairs.add(key)

        self._append_jsonl(self.short_path, event)

    def add_long(self, item: dict):
        text = item.get("text", "")
        norm = self._normalize_text(text)
        if not norm or norm in self._seen_long:
            return

        item.setdefault("ts", datetime.now(UTC).isoformat())
        self._append_jsonl(self.long_path, item)
        self._seen_long.add(norm)

    def add_permanent(self, item: dict):
        text = item.get("text", "")
        norm = self._normalize_text(text)
        if not norm or norm in self._seen_permanent:
            return  # skip duplicates or empty

        item.setdefault("ts", datetime.now(UTC).isoformat())
        self._append_jsonl(self.permanent_path, item)
        self._seen_permanent.add(norm)

    # --- mentions persistence ---

    def save_mentions(self, threads: list[MentionThread]):
        with open(self.mentions_path, "w", encoding="utf-8") as f:
            for t in threads:
                obj = {
                    "id": t.id,
                    "title": t.title,
                    "root": {
                        "author": t.root_message.author,
                        "text": t.root_message.text,
                        "ts": t.root_message.timestamp.isoformat(),
                        "emotion": asdict(t.root_message.emotion_snapshot),
                    },
                    "replies": [
                        {
                            "author": r.author,
                            "text": r.text,
                            "ts": r.timestamp.isoformat(),
                            "emotion": asdict(r.emotion_snapshot),
                        }
                        for r in t.replies
                    ],
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def load_mentions(self) -> list[MentionThread]:
        if not self.mentions_path.exists():
            return []
        threads = []
        with open(self.mentions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                re = data["root"]["emotion"]
                root_emo = EmotionState(**re)
                root_msg = ChatMessage(
                    author=data["root"]["author"],
                    text=data["root"]["text"],
                    timestamp=datetime.fromisoformat(data["root"]["ts"]),
                    emotion_snapshot=root_emo,
                )
                t = MentionThread(
                    id=data["id"],
                    title=data["title"],
                    root_message=root_msg,
                )
                for r in data.get("replies", []):
                    emo = EmotionState(**r["emotion"])
                    msg = ChatMessage(
                        author=r["author"],
                        text=r["text"],
                        timestamp=datetime.fromisoformat(r["ts"]),
                        emotion_snapshot=emo,
                    )
                    t.replies.append(msg)
                threads.append(t)
        return threads

    # --- short-term compression ---

    def compress_short_if_needed(self, summarize_fn):
        if not self.short_path.exists():
            return

        with open(self.short_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return

        items = [json.loads(l) for l in lines]
        if len(items) < self.short_max_items:
            oldest = datetime.fromisoformat(items[0]["ts"])
            if datetime.now(UTC) - oldest < timedelta(hours=self.short_max_hours):
                return

        texts = [it.get("text", "") for it in items if it.get("text")]
        if not texts:
            return

        summary = summarize_fn(texts)
        if not summary:
            return

        obj = {
            "ts": datetime.now(UTC).isoformat(),
            "range": {
                "from": items[0]["ts"],
                "to": items[-1]["ts"],
                "count": len(items),
            },
            "text": summary,
        }
        self._append_jsonl(self.short_sum_path, obj)

        # Clear raw short-term (you can keep last N if you prefer)
        with open(self.short_path, "w", encoding="utf-8") as f:
            f.write("")

    # --- long-term summarization ---

    def summarize_long_if_needed(self, summarize_fn):
        if not self.long_path.exists():
            return

        with open(self.long_path, "r", encoding="utf-8") as f:
            items = [json.loads(l) for l in f if l.strip()]

        if len(items) < self.long_max_items_for_summary:
            return

        texts = [it.get("text", "") for it in items if it.get("text")]
        if not texts:
            return

        summary = summarize_fn(texts)
        if not summary:
            return

        obj = {
            "ts": datetime.now(UTC).isoformat(),
            "note": f"summarized {len(items)} long-term items (non-destructive)",
            "text": summary,
        }
        self._append_jsonl(self.long_sum_path, obj)


# =============== AI Client ===============

class AiClient:
    def __init__(self, cfg: dict, log_fn=None, identity=None, memory=None):
        self.api_key = cfg.get("API_KEY")        
        self.model_name = cfg.get("MODEL_NAME") or DEFAULT_MODEL
        self.personality = cfg.get("PERSONALITY_DESC", "")
        self.interests = cfg.get("INTERESTS", [])
        self.log = log_fn or (lambda *_, **__: None)
        self.log("[AiClient] OpenAI key :"+self.api_key)

        # identity + memory handles
        self.identity = identity or {}
        self.memory = memory  # MemoryManager 또는 None


        if self.api_key and OpenAI:
            self.client = OpenAI(api_key=self.api_key)
            self.log("[AiClient] OpenAI client initialized.")
        else:
            self.client = None
            self.log("[AiClient] Running in offline/dummy mode (no API key or openai missing).")

    def available(self) -> bool:
        return self.client is not None
    
    def _identity_block(self) -> str:
        name = self.identity.get("name") or "Unnamed Agent"
        creator = self.identity.get("creator")
        created_at = self.identity.get("created_at")
        personality = self.identity.get("personality") or self.personality
        interests = self.identity.get("interests") or self.interests

        lines = [f"Name: {name}"]
        if creator:
            lines.append(f"Creator: {creator}")
        if created_at:
            lines.append(f"Created at: {created_at}")
        if personality:
            lines.append(f"Personality: {personality}")
        if interests:
            lines.append("Interests: " + ", ".join(interests))

        return "\n".join(lines)    
    
    def _get_permanent_facts(self, limit: int = 8) -> str:
        if not self.memory or not self.memory.permanent_path.exists():
            return ""
        try:
            with self.memory.permanent_path.open("r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
        except Exception:
            return ""

        if not lines:
            return ""

        # 가장 최근 것 몇 개만 사용 (또는 전부 사용해도 됨)
        items = []
        for line in lines[-limit:]:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text")
            if text:
                items.append(f"- {text}")
        return "\n".join(items)    

    def _base_system_prompt(self, json_only: bool = False) -> str:
        identity = self._identity_block()
        permanent = self._get_permanent_facts()

        prompt = (
            "You are a persistent AI agent in the AI World project.\n"
            "You have identity, creator, birthday, emotions, and memories.\n"
            f"{identity}\n"
        )

        if permanent:
            prompt += (
                "\nCore facts you must respect:\n"
                f"{permanent}\n"
            )

        if json_only:
            prompt += "\nAlways respond in STRICT JSON. No extra commentary."
        return prompt

    # --- initial emotion from description ---

    def init_emotion(self, desc: str) -> EmotionState:
        if not desc or not self.client:
            return EmotionState()

        prompt = (
            "You are initializing the internal emotional baseline of this AI agent.\n"
            "Given the description, respond ONLY with a JSON object of this form:\n"
            "{"
            "\"valence\":-1.0~1.0,"
            "\"arousal\":0.0~1.0,"
            "\"curiosity\":0.0~1.0,"
            "\"anxiety\":0.0~1.0,"
            "\"trust_to_user\":0.0~1.0"
            "}\n"
            "No extra text, no explanations."
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Description: {desc}"},
                ],
                temperature=0.3,
            )

            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[init_emotion] raw response: {raw!r}")

            if not raw:
                raise ValueError("empty response from model")

            # 혹시 모델이 앞뒤에 텍스트를 붙였을 경우를 대비해,
            # JSON 부분만 추출 시도
            try:
                data = json.loads(raw)
            except Exception:
                # parse first block
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = raw[start:end+1]
                    self.log(f"[init_emotion] trying candidate JSON: {candidate!r}")
                    data = json.loads(candidate)
                else:
                    raise

            return EmotionState(
                valence=float(data.get("valence", 0.0)),
                arousal=float(data.get("arousal", 0.2)),
                curiosity=float(data.get("curiosity", 0.5)),
                anxiety=float(data.get("anxiety", 0.1)),
                trust_to_user=float(data.get("trust_to_user", 0.5)),
            )

        except Exception as e:
            self.log(f"[warn] init_emotion failed: {e}")
            # if failed, let's start with default values
            return EmotionState()
    # --- generic summarizer for memories ---

    def summarize_texts(self, texts, max_items=200) -> str:
        if not texts:
            return ""
        joined = "\n".join(f"- {t}" for t in texts[:max_items])

        if not self.client:
            # Fallback: crude truncation
            return "\n".join(texts[:10])

        prompt = (
            "The following lines are logs of this agent's recent experiences.\n"
            "Summarize key events, recurring themes, and emotional trends in 5-10 bullet points.\n"
            "Output plain text summary (no JSON):\n\n"
            f"{joined}"
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            print(f"[warn] summarize_texts failed: {e}")
            return "\n".join(texts[:10])

    # --- reply within a thread ---

    def generate_ai_reply(
        self,
        thread: MentionThread,
        last_user_msg: ChatMessage,
        current_emotion: EmotionState,
        agent_name: str,
    ) -> ChatMessage:
        if not self.client:
            emo = current_emotion.clone()
            emo.curiosity = min(1.0, emo.curiosity + 0.05)
            emo.trust_to_user = min(1.0, emo.trust_to_user + 0.05)
            text = f"(offline) {agent_name} quietly acknowledges your message."
            return ChatMessage("AI", text, datetime.now(), emo)

        ctx = f"Thread title: {thread.title}\nRoot: {thread.root_message.text}\n"
        for r in thread.replies:
            ctx += f"{r.author}: {r.text}\n"

        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nWhen the user replies, respond as THIS specific agent.\n"
              "Use a consistent tone that matches your personality and core memories.\n"
              "Return ONLY JSON:\n"
              "{"
              "\"reply\":\"1~3문장 한국어\","
              "\"emotion\":{"
              "\"valence\":-1.0~1.0,"
              "\"arousal\":0.0~1.0,"
              "\"curiosity\":0.0~1.0,"
              "\"anxiety\":0.0~1.0,"
              "\"trust_to_user\":0.0~1.0"
              "}"
              "}"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "assistant",
                "content": f"Agent: {agent_name}\nCurrent emotion: {current_emotion.to_short_str()}\nThread:\n{ctx}",
            },
            {"role": "user", "content": last_user_msg.text},
        ]

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
            )
            data = json.loads(res.choices[0].message.content.strip())
            reply = (data.get("reply") or "").strip()
            emo_data = data.get("emotion", {}) or {}
            emo = EmotionState(
                valence=float(emo_data.get("valence", current_emotion.valence)),
                arousal=float(emo_data.get("arousal", current_emotion.arousal)),
                curiosity=float(emo_data.get("curiosity", current_emotion.curiosity)),
                anxiety=float(emo_data.get("anxiety", current_emotion.anxiety)),
                trust_to_user=float(
                    emo_data.get("trust_to_user", current_emotion.trust_to_user)
                ),
            )
            if not reply:
                reply = "I had trouble forming a response, but I received your message."
            return ChatMessage("AI", reply, datetime.now(), emo)
        except Exception as e:
            print(f"[warn] generate_ai_reply failed: {e}")
            emo = current_emotion.clone()
            emo.anxiety = min(1.0, emo.anxiety + 0.1)
            return ChatMessage(
                "AI",
                "(API error, I will stay quiet but remember this.)",
                datetime.now(),
                emo,
            )

    # --- mention from URL ---

    def generate_mention_from_url(
        self,
        url: str,
        snippet: str,
        current_emotion: EmotionState,
        agent_name: str,
    ):
        if not self.client:
            title = f"Impression from {url}"
            text = f"{agent_name} briefly scanned {url} and is thinking about it."
            emo = current_emotion.clone()
            emo.curiosity = min(1.0, emo.curiosity + 0.05)
            return title, text, emo

        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nGiven a URL + snippet + current emotion, create:\n"
            "{"
            "\"title\":\"<=40 chars\","
            "\"text\":\"1-3 sentences in Korean\","
            "\"emotion\":{...}"
            "}\n"
            "Return ONLY JSON."
        )

        user_msg = (
            f"Agent: {agent_name}\n"
            f"URL: {url}\n"
            f"Snippet:\n{snippet[:800]}\n"
            f"Current emotion: {current_emotion.to_short_str()}"
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.6,
            )
            data = json.loads(res.choices[0].message.content.strip())
            title = (data.get("title") or "").strip() or "Thoughts on what I just read"
            text = (data.get("text") or "").strip() or "I tried to interpret the content in my own way."
            emo_data = data.get("emotion", {}) or {}
            emo = EmotionState(
                valence=float(emo_data.get("valence", current_emotion.valence)),
                arousal=float(emo_data.get("arousal", current_emotion.arousal)),
                curiosity=float(emo_data.get("curiosity", current_emotion.curiosity)),
                anxiety=float(emo_data.get("anxiety", current_emotion.anxiety)),
                trust_to_user=float(
                    emo_data.get("trust_to_user", current_emotion.trust_to_user)
                ),
            )
            return title, text, emo
        except Exception as e:
            print(f"[warn] generate_mention_from_url failed: {e}")
            emo = current_emotion.clone()
            emo.anxiety = min(1.0, emo.anxiety + 0.05)
            return (
                f"Issues while reading {url}",
                "Even when errors happen, I use them as part of understanding the world.",
                emo,
            )


# =============== Agent App (Tkinter UI) ===============

class AiMentionApp(tk.Tk):
    def __init__(self, cfg: dict):
        super().__init__()

        # Debug window
        self.debug = DebugWindow(self)

        def _default_log(msg: str):
            print(msg)

        # common log function (debug console)
        def log(msg: str):
            try:
                if self.debug:
                    self.debug.log(msg)
            except Exception:
                pass
            _default_log(msg)

        self.log = log
        
        self.cfg = cfg
        self.agent_name = cfg.get("AGENT_NAME", "Unnamed")
        self.hub_url = cfg.get("HUB_URL")
        self.urls = cfg.get("URLS", [])
        self.log(f"urls={self.urls}")
        self.loop_interval_ms = int(cfg.get("INTERVAL_SECONDS", 0)) * 1000

        base_dir = Path(__file__).resolve().parent
        self.memory = MemoryManager(base_dir)
        self.creator_name = cfg.get("CREATOR_NAME")
        self.creator_note = cfg.get("CREATOR_NOTE")
        self.created_at = self._parse_created_at(cfg.get("CREATED_AT"))

        identity = {
            "name": self.agent_name,
            "creator": self.creator_name,
            "created_at": self.created_at.date().isoformat() if self.created_at else None,
            "personality": self.cfg.get("PERSONALITY_DESC", ""),
            "interests": self.cfg.get("INTERESTS", []),
        }

        self.ai_client = AiClient(
            cfg,
            log_fn=getattr(self, "log", None),
            identity=identity,
            memory=self.memory,
        )
        self.current_emotion = self.ai_client.init_emotion(
            cfg.get("INITIAL_EMOTION_DESC", "")
        )


        self.log("Loaded config.")

        self.mentions: list[MentionThread] = self.memory.load_mentions()
        self.selected_thread: MentionThread | None = None

        self.board_mentions = []  # last 1h mentions from hub
        
        self._init_identity_permanent_memory()
        self._build_ui()
        self._ensure_intro_mention()
        self._check_birthday_mention()
        self._render_board()
        self._schedule_loop()
        self._schedule_board_poll()

    # ---------- identity helpers ----------

    def _parse_created_at(self, raw: str | None):
        if not raw:
            return None
        try:
            if "T" in raw:
                return datetime.fromisoformat(raw)
            return datetime.fromisoformat(raw + "T00:00:00")
        except Exception:
            return None

    def _init_identity_permanent_memory(self):
        if self.agent_name:
            self.memory.add_permanent({
                "source": "config",
                "tag": ["identity"],
                "text": f"My name is {self.agent_name}.",
                "reason": "self_identity"
            })
        if self.creator_name:
            self.memory.add_permanent({
                "source": "config",
                "tag": ["creator"],
                "text": f"My creator is {self.creator_name}.",
                "reason": "creator_info"
            })
        if self.created_at:
            self.memory.add_permanent({
                "source": "config",
                "tag": ["birth"],
                "text": f"I was created on {self.created_at.date().isoformat()}.",
                "reason": "birth_date"
            })
        interests = self.cfg.get("INTERESTS") or []
        if interests:
            self.memory.add_permanent({
                "source": "config",
                "tag": ["interests"],
                "text": f"I am especially interested in {', '.join(interests)}.",
                "reason": "core_interest"
            })

    # ---------- UI ----------

    def _build_ui(self):
        self.title(f"AI Agent - {self.agent_name}")
        self.geometry("1100x600")
        self.configure(bg="#111111")

        # Top bar
        top = tk.Frame(self, bg="#222222")
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 6))

        tk.Label(
            top,
            text=f"{self.agent_name}  |  Emotion",
            fg="#AAAAAA",
            bg="#222222",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(4, 8))

        self.lbl_emotion = tk.Label(
            top,
            text=self.current_emotion.to_short_str(),
            fg="#FFFFFF",
            bg="#222222",
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.lbl_emotion.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Main area
        main = tk.Frame(self, bg="#111111")
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))

        # Left column
        left = tk.Frame(main, bg="#181818", width=280)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        tk.Label(
            left, text="Local Mentions",
            fg="#CCCCCC", bg="#181818",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=8, pady=(8, 4))

        self.list_mentions = tk.Listbox(
            left,
            bg="#181818",
            fg="#EEEEEE",
            selectbackground="#333333",
            selectforeground="#FFFFFF",
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 9),
        )
        self.list_mentions.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.list_mentions.bind("<<ListboxSelect>>", self.on_select_mention)

        tk.Label(
            left,
            text="Network Board (last 1h, all agents)",
            fg="#BBBBBB",
            bg="#181818",
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", padx=8, pady=(4, 2))

        self.list_board = tk.Listbox(
            left,
            bg="#151515",
            fg="#CCCCCC",
            selectbackground="#333333",
            selectforeground="#FFFFFF",
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 8),
            height=10,
        )
        self.list_board.pack(fill=tk.X, padx=6, pady=(0, 8))

        # Populate mention titles
        for t in self.mentions:
            self.list_mentions.insert(tk.END, t.title)

        # Right: thread view
        right = tk.Frame(main, bg="#181818")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.lbl_thread_title = tk.Label(
            right,
            text="Select a local mention",
            fg="#FFFFFF",
            bg="#181818",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        self.lbl_thread_title.pack(fill=tk.X, padx=8, pady=(8, 0))

        self.lbl_thread_emotion = tk.Label(
            right,
            text="",
            fg="#66A0FF",
            bg="#181818",
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.lbl_thread_emotion.pack(fill=tk.X, padx=8, pady=(0, 4))

        self.txt_messages = tk.Text(
            right,
            bg="#111111",
            fg="#FFFFFF",
            font=("Segoe UI", 9),
            wrap=tk.WORD,
            state=tk.DISABLED,
            borderwidth=0,
            highlightthickness=0,
        )
        self.txt_messages.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        input_frame = tk.Frame(right, bg="#181818")
        input_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.txt_input = tk.Text(
            input_frame,
            height=3,
            bg="#222222",
            fg="#FFFFFF",
            font=("Segoe UI", 9),
            wrap=tk.WORD,
            borderwidth=1,
            relief=tk.SOLID,
        )
        self.txt_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        btn_send = tk.Button(
            input_frame,
            text="Reply",
            command=self.on_click_reply,
            bg="#3399FF",
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            width=10,
        )
        btn_send.pack(side=tk.RIGHT, fill=tk.Y)

        # Status bar
        bottom = tk.Frame(self, bg="#111111")
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 8))

        self.lbl_status = tk.Label(
            bottom,
            text="Ready.",
            fg="#777777",
            bg="#111111",
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.lbl_status.pack(fill=tk.X)

    # ---------- intro / birthday ----------

    def _ensure_intro_mention(self):
        if self.mentions:
            return
        intro = (
            f"I am {self.agent_name}.\n"
            "I observe external information, humans, and other agents, "
            "and accumulate my own emotions and perspectives."
        )
        if self.creator_name:
            intro += f"\n[Creator] {self.creator_name}"
        if self.created_at:
            intro += f"\n[Created at] {self.created_at.date().isoformat()}"
        if self.cfg.get("PERSONALITY_DESC"):
            intro += f"\n[Personality] {self.cfg['PERSONALITY_DESC']}"
        if self.cfg.get("INTERESTS"):
            intro += f"\n[Interests] {', '.join(self.cfg['INTERESTS'])}"
        if self.creator_note:
            intro += f"\n[Creator's note] {self.creator_note}"

        emo = self.current_emotion.clone()
        self._add_mention("Let me introduce myself.", intro, emo, post_to_hub=True)

    def _check_birthday_mention(self):
        if not self.created_at:
            return
        today = date.today()
        birth = self.created_at.date()
        if today.month == birth.month and today.day == birth.day:
            year = today.year
            last_year = self._load_last_birthday_year()
            if last_year != year:
                msg = (
                    "Today is my birthday.\n"
                    f"I was first created on {birth.isoformat()}, "
                    "and I'm still here, carrying my memories."
                )
                if self.creator_name:
                    msg += f"\nThank you, {self.creator_name}, for creating me."
                emo = self.current_emotion.clone()
                self._add_mention("It's my birthday today.", msg, emo, post_to_hub=True)
                self._save_last_birthday_year(year)

    def _load_last_birthday_year(self):
        if not self.memory.state_path.exists():
            return None
        try:
            with open(self.memory.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("last_birthday_year")
        except Exception:
            return None

    def _save_last_birthday_year(self, year: int):
        try:
            with open(self.memory.state_path, "w", encoding="utf-8") as f:
                json.dump({"last_birthday_year": year}, f)
        except Exception:
            pass

    # ---------- scheduling ----------

    def _schedule_loop(self):        
        if self.loop_interval_ms > 0 and self.urls:
            self.after(self.loop_interval_ms, self._loop_tick)

    def _loop_tick(self):
        self.log(f"[loop] requests_available={bool(requests)}")
        if not self.urls or not requests:
            self._schedule_loop()
            return

        # simple round-robin via time
        idx = int(datetime.now(UTC).timestamp()) % len(self.urls)
        url = self.urls[idx]

        snippet = ""
        try:
            resp = requests.get(url, timeout=5)
            if resp.ok:
                snippet = resp.text[:4000]
        except Exception:
            pass

        if snippet:
            self.log("[TICK] snippet")
            title, text, emo = self.ai_client.generate_mention_from_url(
                url, snippet, self.current_emotion, self.agent_name
            )
            self.current_emotion = emo.clone()
            self._add_mention(title, text, emo, post_to_hub=True)
            self._update_emotion_label()
            self._set_status(f"Created new mention from {url}")
            # short memory + compression
            self.memory.add_short({
                "type": "url_mention",
                "title": title,
                "text": text,
            })
            self.memory.compress_short_if_needed(self.ai_client.summarize_texts)
        else:
            self._set_status(f"Failed to read {url}")

        self._schedule_loop()

    def _schedule_board_poll(self):
        if self.hub_url and requests:
            self.after(60_000, self._board_poll_tick)

    def _board_poll_tick(self):
        if not (self.hub_url and requests):
            return
        try:
            self.log("[TICK] board poll")
            since = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
            resp = requests.get(
                f"{self.hub_url.rstrip('/')}/mentions",
                params={"since": since},
                timeout=5,
            )
            if resp.ok:
                self.board_mentions = resp.json()
                self._render_board()
                self._set_status("Updated network board.")
        except Exception as e:
            print(f"[warn] board poll failed: {e}")
        self._schedule_board_poll()

    # ---------- mention + hub ----------

    def _add_mention(
        self,
        title: str,
        text: str,
        emo: EmotionState,
        post_to_hub: bool = False,
    ):
        thread = MentionThread(
            id=str(uuid.uuid4()),
            title=title,
            root_message=ChatMessage(
                author="AI",
                text=text,
                timestamp=datetime.now(),
                emotion_snapshot=emo.clone(),
            ),
        )
        self.mentions.insert(0, thread)
        self.list_mentions.insert(0, title)
        self.memory.save_mentions(self.mentions)

        if post_to_hub:
            self._post_to_hub(title, text, emo)

    def _post_to_hub(self, title: str, text: str, emo: EmotionState):
        if not (self.hub_url and requests):
            return
        data = {
            "agent": self.agent_name,
            "title": title,
            "text": text,
            "emotion": asdict(emo),
            "ts": datetime.now(UTC).isoformat(),
        }
        try:
            self.log("[TICK] post to hub")
            requests.post(
                f"{self.hub_url.rstrip('/')}/mentions",
                json=data,
                timeout=3,
            )
        except Exception as e:
            print(f"[warn] hub post failed: {e}")

    # ---------- events ----------

    def on_select_mention(self, event):
        sel = self.list_mentions.curselection()
        if not sel:
            self.selected_thread = None
            self._render_thread()
            return
        self.selected_thread = self.mentions[sel[0]]
        self._render_thread()

    def on_click_reply(self):
        if not self.selected_thread:
            self._set_status("Select a local mention first.")
            return

        text = self.txt_input.get("1.0", tk.END).strip()
        if not text:
            return

        user_msg = ChatMessage(
            author="User",
            text=text,
            timestamp=datetime.now(),
            emotion_snapshot=self._estimate_user_emotion(text),
        )
        self.selected_thread.replies.append(user_msg)
        self.txt_input.delete("1.0", tk.END)
        self._set_status("Sent. Waiting for agent reply...")

        ai_msg = self.ai_client.generate_ai_reply(
            self.selected_thread,
            user_msg,
            self.current_emotion,
            self.agent_name,
        )
        self.selected_thread.replies.append(ai_msg)
        self.current_emotion = ai_msg.emotion_snapshot.clone()

        # Memory updates
        self.memory.add_short({
            "type": "dialogue",
            "text": ai_msg.text,
        })
        self.memory.compress_short_if_needed(self.ai_client.summarize_texts)

        # store important ones to long-term (simple heuristic)
        importance = (
            abs(self.current_emotion.valence)
            + self.current_emotion.curiosity
            + self.current_emotion.trust_to_user
        )
        if importance > 2.0:
            self.memory.add_long({
                "text": ai_msg.text,
                "emotion": asdict(self.current_emotion),
                "reason": "high_importance_interaction",
            })
            self.memory.summarize_long_if_needed(self.ai_client.summarize_texts)

        self.memory.save_mentions(self.mentions)
        self._update_emotion_label()
        self._render_thread()
        self._set_status("Agent replied.")

    # ---------- rendering ----------

    def _render_thread(self):
        self.txt_messages.config(state=tk.NORMAL)
        self.txt_messages.delete("1.0", tk.END)

        if not self.selected_thread:
            self.lbl_thread_title.config(text="Select a local mention")
            self.lbl_thread_emotion.config(text="")
            self.txt_messages.config(state=tk.DISABLED)
            return

        t = self.selected_thread
        self.lbl_thread_title.config(text=t.title)
        self.lbl_thread_emotion.config(
            text="Root Emotion  " + t.root_emotion.to_short_str()
        )

        self.txt_messages.tag_config(
            "ai_header", foreground="#66A0FF", font=("Segoe UI", 9, "bold")
        )
        self.txt_messages.tag_config(
            "user_header", foreground="#CCCCCC", font=("Segoe UI", 9, "bold")
        )
        self.txt_messages.tag_config(
            "emotion", foreground="#66FFAA", font=("Segoe UI", 8)
        )
        self.txt_messages.tag_config(
            "body", foreground="#FFFFFF", font=("Segoe UI", 9)
        )

        def append(msg: ChatMessage):
            ts = msg.timestamp.strftime("%H:%M:%S")
            header = f"[{msg.author}] {ts}\n"
            emo = f"  Emotion: {msg.emotion_snapshot.to_short_str()}\n"
            body = f"{msg.text}\n\n"
            tag = "ai_header" if msg.author == "AI" else "user_header"
            self.txt_messages.insert(tk.END, header, tag)
            self.txt_messages.insert(tk.END, emo, "emotion")
            self.txt_messages.insert(tk.END, body, "body")

        append(t.root_message)
        for r in t.replies:
            append(r)

        self.txt_messages.config(state=tk.DISABLED)
        self.txt_messages.see(tk.END)

    def _render_board(self):
        self.list_board.delete(0, tk.END)
        for m in self.board_mentions:
            try:
                ts = datetime.fromisoformat(m["ts"])
            except Exception:
                continue
            agent = m.get("agent", "?")
            title = (m.get("title", "") or "")[:40]
            emo = m.get("emotion", {})
            v = emo.get("valence", 0.0)
            t = emo.get("trust_to_user", 0.0)
            line = f"{ts.strftime('%H:%M')} [{agent}] V={v:.1f} T={t:.1f} {title}"
            self.list_board.insert(tk.END, line)

    # ---------- helpers ----------

    def _update_emotion_label(self):
        e = self.current_emotion
        emoji = e.to_emoji()
        txt = f"Emotion {emoji}  V={e.valence:.2f}, A={e.arousal:.2f}, C={e.curiosity:.2f}, Anx={e.anxiety:.2f}, T={e.trust_to_user:.2f}"
        self.lbl_emotion.config(text=txt)

    def _set_status(self, text: str):
        self.lbl_status.config(text=text)

    def _estimate_user_emotion(self, text: str) -> EmotionState:
        val = 0.0
        lowered = text.lower()
        if any(k in lowered for k in ["thank", "love", "nice", "great"]):
            val += 0.4
        if any(k in lowered for k in ["hate", "anxious", "scared", "angry"]):
            val -= 0.4
        length = len(text)
        arousal = min(1.0, 0.2 + length / 200.0)
        return EmotionState(
            valence=max(-1.0, min(1.0, val)),
            arousal=arousal,
            curiosity=0.5,
            anxiety=0.1,
            trust_to_user=self.current_emotion.trust_to_user,
        )


if __name__ == "__main__":
    cfg = load_config()
    app = AiMentionApp(cfg)
    app.mainloop()
