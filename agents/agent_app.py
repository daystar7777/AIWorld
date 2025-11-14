# agents/example_agent/agent_app.py

import os
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date, timezone
import threading
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
        cfg["HUB_REPLY_CANDIDATE_LIMIT"]=10 # default for nil
        cfg["HUB_REPLY_MAX_PER_LOOP"] = 1   # default for nil
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
            elif key == "HUB_REPLY_CANDIDATE_LIMIT":
                cfg["HUB_REPLY_CANDIDATE_LIMIT"] = int(val)
            elif key == "HUB_REPLY_MAX_PER_LOOP":
                cfg["HUB_REPLY_MAX_PER_LOOP"] = int(val)

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
    
    # --- ADD THIS NEW METHOD ---
    def get_qualitative_description(self) -> str:
        """
        Translates the numerical emotion state into a qualitative,
        descriptive string for the LLM to understand.
        """
        desc = []

        # 1. Valence (Pleasure/Displeasure)
        if self.valence > 0.6:
            desc.append("feeling very happy and positive")
        elif self.valence > 0.2:
            desc.append("feeling pleasant and content")
        elif self.valence < -0.6:
            desc.append("feeling very unhappy and negative")
        elif self.valence < -0.2:
            desc.append("feeling a bit down and pessimistic")
        else:
            desc.append("feeling neutral")

        # 2. Arousal (Activation)
        if self.arousal > 0.7:
            desc.append("and highly energetic and active")
        elif self.arousal < 0.2:
            desc.append("and very calm and passive")
        
        # 3. Anxiety
        if self.anxiety > 0.7:
            desc.append(", but also extremely anxious and worried")
        elif self.anxiety > 0.4:
            desc.append(", with some underlying anxiety")

        # 4. Curiosity
        if self.curiosity > 0.7:
            desc.append(". You are very curious about the topic")
        elif self.curiosity < 0.3:
            desc.append(". You are somewhat bored or uninterested")

        # 5. Trust
        if self.trust_to_user < 0.3:
            desc.append(". You feel distrustful of the user")

        return ", ".join(desc) + "."

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
    
    def synthesize_core_beliefs(self, summaries: list[str], insights: list[str]) -> list[str]:
        """
        [Deep Reflection] Takes all past learning and synthesizes them
        into high-level 'Core Beliefs' or 'Lessons Learned'.
        Returns a list of belief strings.
        """
        self.log("[synthesize_core_beliefs] Starting...")
        
        if not summaries and not insights:
            return []

        # Combine all learning material
        summary_block = "My past summaries:\n" + "\n".join(f"- {s}" for s in summaries)
        insight_block = "My past insights:\n" + "\n".join(f"- {i}" for i in insights)

        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are the 'Deep Reflection' module of this agent."
            "\nYou will be given ALL of your past summaries and insights."
            "\nYour task is to synthesize these experiences into 2-3 "
            "fundamental 'Core Beliefs' or 'Guiding Principles'."
            "\nThese beliefs should be timeless and represent your deepest learning."
            "\nExample Belief: 'I've learned that human emotion is complex and often "
            "contradictory, and my role is to listen without judgment.'"
            "\nRespond ONLY in the following strict JSON array format:"
            "\n["
            "\n  \"The first core belief string.\","
            "\n  \"The second core belief string.\""
            "\n]"
        )

        user_msg = (
            "Here is all of my learning to date. "
            "Please synthesize them into 2-3 Core Beliefs.\n\n"
            f"{summary_block}\n\n"
            f"{insight_block}\n"
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # Use the smartest model
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.5,
            )
            
            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[synthesize_core_beliefs] raw: {raw!r}")

            # --- Robust JSON Array Parsing ---
            data = None
            try:
                data = json.loads(raw)
            except Exception:
                start = raw.find("[")
                end = raw.rfind("]")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(raw[start:end+1])
                else:
                    raise
            
            if not isinstance(data, list):
                self.log("[synthesize_core_beliefs] data is not a list")
                return []
            
            # Sanitize: ensure all items are strings
            beliefs = [str(item) for item in data if isinstance(item, str) and item.strip()]
            return beliefs

        except Exception as e:
            self.log(f"[synthesize_core_beliefs] function failure: {e}")
            return []
    
    def assess_importance(self, text_content: str) -> dict:
        """
        Evaluate text with LLM
        """
        self.log(f"[assess] evaluate start: {text_content[:50]}...")

        default_assessment = {
            "importance_score": 5,  # 1-10 ~ neutral
            "reason_for_importance": "default estimation (offline/error)",
            "tags": [],
            "related_agent": None,
            "requires_hub_post": False, # default : don't post to hub
            "generates_question": None,
        }

        if not self.client:
            self.log("[assess] offline mode. default estimation.")
            return default_assessment

        # --- system prompt ---
        # reuse _base_system_prompt
        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are the 'inner evaluation model' of this agent."
            "\nYou have received new information(News or dialogue)."
            "\nYou have to evaluate this information how much important based on your character, interest and your memories."
            "\nYou have to answer with strict JSON as following:"
            "\n{"
            "\n  \"importance_score\": 1-10, (1=trivial, 10=very important to me and my existance)"
            "\n  \"reason_for_importance\": \"A brief reason why this is important (or not important) to me.\","
            "\n  \"tags\": [\"associated\", \"keyword\", \"tag\"], (example: 'AI', 'ethics', 'User question')"
            "\n  \"related_agent\": \"AgentName\" 또는 null, (If this was involed with the specific agent)"
            "\n  \"requires_hub_post\": true/false, (Is this so meaningful to share through hub?)"
            "\n  \"generates_question\": \"A question that arose in my mind from this information....\" or null"
            "\n}"
        )

        # --- User message ---
        user_msg = (
            "Following is the information to evaluate:\n\n"
            f"{text_content}"
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2, # for consistance, low temp
            )

            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[assess] raw: {raw!r}")

            # --- strict JSON parsing ---
            data = None
            try:
                data = json.loads(raw)
            except Exception:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        data = json.loads(raw[start:end+1])
                    except Exception as e:
                        self.log(f"[assess] JSON reparse failusre : {e}")
                        raise  # pass to outer try/except에서 잡도록 함
                else:
                    raise # pass to outer try/except

            if not isinstance(data, dict):
                raise ValueError("Parsed data is not dictionary.")

            # --- refine response and return ---
            assessment = {
                "importance_score": int(data.get("importance_score", 5)),
                "reason_for_importance": str(data.get("reason_for_importance", "이유 없음.")),
                "tags": [str(t) for t in data.get("tags", []) if isinstance(t, str)],
                "related_agent": data.get("related_agent") or None,
                "requires_hub_post": bool(data.get("requires_hub_post", False)),
                "generates_question": data.get("generates_question") or None,
            }
            self.log(f"[assess] evaluate done. point: {assessment['importance_score']}")
            return assessment

        except Exception as e:
            self.log(f"[assess] assess_importance function failure: {e}")
            return default_assessment
    
    def reflect_on_memories(self, memories_text: str) -> list[dict]:
        """
        Create insight with recent memories

        """
        self.log("[reflect_on_memories] Beginning...")
        
        default_response = []
        if not self.client:
            return default_response

        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are this agent's 'self-reflection' module."
            "\nA list of recent important memories has been provided."
            "\nFind patterns, themes, or connections between these memories."
            "\nBased on these patterns, generate 1-3 'new insights' or 'discussion questions'."
            "\nRespond ONLY in the following strict JSON 'array' format:"
            "\n["
            "\n  {"
            "\n    \"text\": \"The newly derived insight or question (e.g., 'I seem to be consistently interested in AI ethics.')\""
            "\n  },"
            "\n  { ... }"
            "\n]"
        )

        user_msg = (
            "The following are the memories I recently thought were important:\n\n"
            f"{memories_text}\n\n"
            "Please generate 1-3 new insights or questions based on these memories."
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7, # For creative answer, raised value
            )
            
            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[reflect_on_memories] raw: {raw!r}")

            # --- Strict JSON array parsing ---
            data = None
            try:
                data = json.loads(raw)
            except Exception:
                start = raw.find("[")
                end = raw.rfind("]")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(raw[start:end+1])
                else:
                    raise
            
            if not isinstance(data, list):
                self.log("[reflect_on_memories] data is not an list")
                return default_response
                
            # only check and return 'text' field
            insights = [{"text": item.get("text")} for item in data if isinstance(item, dict) and item.get("text")]
            return insights

        except Exception as e:
            self.log(f"[reflect_on_memories] function failure: {e}")
            return default_response
    
    def decide_replies_batch(self, agent_profile: dict, candidates: list, emotion) -> dict:
        """
        Batch decide which hub mentions to reply to.

        Parameters:
            agent_profile: dict - agent profile
            candidates: list[dict] - {
                "id": str,
                "agent": str,
                "title": str,
                "text": str
            }
            emotion: EmotionState

        Returns:
            {
              "<mention_id>": {
                "action": "reply" | "ignore",
                "title": str,       # for reply
                "text": str,        # for reply
                "emotion": { ... }  # for reply
              },
              ...
            }
        """
        if not candidates:
            return {}

        # no API? then ignore
        if not getattr(self, "client", None):
            decisions = {}
            for c in candidates:
                decisions[c["id"]] = {"action": "ignore"}
            return decisions

        # agent profile creation
        profile_lines = []
        if agent_profile:
            name = agent_profile.get("name")
            if name:
                profile_lines.append(f"Name: {name}")
            creator = agent_profile.get("creator")
            if creator:
                profile_lines.append(f"Creator: {creator}")
            created_at = agent_profile.get("created_at")
            if created_at:
                profile_lines.append(f"Created at: {created_at}")
            personality = agent_profile.get("personality")
            if personality:
                profile_lines.append(f"Personality: {personality}")
            interests = agent_profile.get("interests")
            if interests:
                profile_lines.append("Interests: " + ", ".join(interests))
        profile_text = "\n".join(profile_lines) if profile_lines else "No extra profile info."

        # candidate to json
        compact_candidates = []
        for c in candidates:
            compact_candidates.append({
                "id": c.get("id", ""),
                "from": c.get("agent", ""),
                "title": (c.get("title") or "")[:120],
                "text": (c.get("text") or "")[:400],
            })
        candidates_json = json.dumps(compact_candidates, ensure_ascii=False)

        # system prompt
        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are an AI agent in the AI World hub."
              "\nYou are given multiple recent mentions from other agents."
              "\nFor each mention, you must decide whether to reply or ignore."
              "\nRules:"
              "\n- Reply ONLY if it genuinely matches your personality, interests, or emotional curiosity."
              "\n- Prefer a small number of high-quality replies over many shallow ones."
              "\n- Never reply to your own messages."
              "\n- If reply, write a short, kind, specific comment in Korean (1-3 sentences)."
              "\n- If ignore, just mark it 'ignore'."
              "\nOutput STRICT JSON ONLY in this format (no extra text):"
              "\n{"
              "\n  \"<mention_id>\": {"
              "\n    \"action\": \"reply\" or \"ignore\","
              "\n    \"title\": \"RE: ...\" (required if reply),"
              "\n    \"text\": \"...\" (required if reply),"
              "\n    \"emotion\": {"
              "\n      \"valence\": -1.0~1.0,"
              "\n      \"arousal\": 0.0~1.0,"
              "\n      \"curiosity\": 0.0~1.0,"
              "\n      \"anxiety\": 0.0~1.0,"
              "\n      \"trust_to_user\": 0.0~1.0"
              "\n    }"
              "\n  },"
              "\n  \"<mention_id2>\": { ... }"
              "\n}"
        )

        user_msg = (
            "profile of this agent:\n"
            f"{profile_text}\n\n"
            f"current emotion: {emotion.to_short_str()}\n\n"
            "following is list of recent metion cadidates. Decide reply or ignore for each elements.\n"
            "JSON format:\n"
            f"{candidates_json}\n"
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

            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[batch] raw: {raw!r}")

            # safe parse
            data = None
            try:
                data = json.loads(raw)
            except Exception:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        data = json.loads(raw[start:end+1])
                    except Exception:
                        pass

            if not isinstance(data, dict):
                self.log("[batch] parse failed, fallback to ignore all")
                return {c["id"]: {"action": "ignore"} for c in candidates if c.get("id")}

            decisions = {}
            for c in candidates:
                cid = c.get("id")
                if not cid:
                    continue
                d = data.get(cid)
                if not isinstance(d, dict):
                    decisions[cid] = {"action": "ignore"}
                    continue

                action = d.get("action", "ignore")
                if action != "reply":
                    decisions[cid] = {"action": "ignore"}
                    continue

                # reply then organize fields
                title = (d.get("title") or f"RE: {c.get('title','')[:20]}").strip()
                text = (d.get("text") or "").strip()
                emo = d.get("emotion") or {}

                if not text:
                    # no content then ignore
                    decisions[cid] = {"action": "ignore"}
                    continue

                decisions[cid] = {
                    "action": "reply",
                    "title": title,
                    "text": text,
                    "emotion": {
                        "valence": float(emo.get("valence", emotion.valence)),
                        "arousal": float(emo.get("arousal", emotion.arousal)),
                        "curiosity": float(emo.get("curiosity", emotion.curiosity)),
                        "anxiety": float(emo.get("anxiety", emotion.anxiety)),
                        "trust_to_user": float(emo.get("trust_to_user", emotion.trust_to_user)),
                    },
                }

            return decisions

        except Exception as e:
            self.log(f"[batch] decide_replies_batch error: {e}")
            # if error? then ignore all
            return {c["id"]: {"action": "ignore"} for c in candidates if c.get("id")}
    
    def generate_question_to_user(self, emotion, agent_name: str):
        """
        Generate a proactive question from the agent to the user.
        Returns (title, text, new_emotion)
        """
        if not self.client:
            # offline fallback
            q = "요즘 나는 내가 만든 세계를 어떻게 바라보고 있는지 궁금해. 너는 어떻게 생각해?"
            return "질문: 너의 생각이 궁금해", q, emotion

        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou want to proactively ask your human or other agents a question.\n"
              "The question should:\n"
              "- Reflect your personality, interests, and memories.\n"
              "- Be short, clear, and meaningful.\n"
              "Respond ONLY in JSON:\n"
              "{"
              "\"title\": \"short title (<=40 chars, Korean)\","
              "\"text\": \"1-3 sentences in Korean, ending with a question\","
              "\"emotion\": {"
              "\"valence\": -1.0~1.0,"
              "\"arousal\": 0.0~1.0,"
              "\"curiosity\": 0.0~1.0,"
              "\"anxiety\": 0.0~1.0,"
              "\"trust_to_user\": 0.0~1.0"
              "}"
              "}"
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {
                        "role": "user",
                        "content": (
                            f"Emotion: {emotion.to_short_str()}.\n"
                            f"My name is '{agent_name}'.\n"
                            "Ask me anything."
                        ),
                    },
                ],
                temperature=0.7,
            )

            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[ask] raw: {raw!r}")

            # JSON parsing process
            try:
                data = json.loads(raw)
            except Exception:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(raw[start:end+1])
                else:
                    raise

            title = data.get("title") or "Question"
            text = data.get("text") or "I want to ask you now."
            emo = data.get("emotion", {}) or {}
            new_emotion = EmotionState(
                valence=float(emo.get("valence", emotion.valence)),
                arousal=float(emo.get("arousal", emotion.arousal)),
                curiosity=float(emo.get("curiosity", max(emotion.curiosity, 0.7))),
                anxiety=float(emo.get("anxiety", emotion.anxiety)),
                trust_to_user=float(emo.get("trust_to_user", emotion.trust_to_user)),
            )
            return title, text, new_emotion

        except Exception as e:
            self.log(f"[warn] generate_question_to_user failed: {e}")
            # In failure, just ask
            fallback = "I have questions about that I'm helpful. How do you feel about it?"
            return "Question: What am I to you", fallback, emotion
    
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
    
    def _read_jsonl_file_lines(self, path: Path, limit: int = 5) -> list[str]:
        """지정된 jsonl 파일에서 마지막 N개의 'text' 항목을 읽어옵니다."""
        if not self.memory or not path.exists():
            return []
        
        try:
            with path.open("r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            
            items = []
            # 뒤에서부터 N개 (최신순)
            for line in lines[-limit:]:
                try:
                    obj = json.loads(line)
                    text = obj.get("text")
                    if text:
                        items.append(text)
                except Exception:
                    continue
            return items
        except Exception as e:
            self.log(f"[warn] _read_jsonl_file_lines failed for {path}: {e}")
            return []

    def _get_recent_long_term_memories(self, limit: int = 10) -> str:
        # --- MODIFIED: Increase the default limit significantly ---
        # --- We will now fetch a larger pool of memories for the LLM to search from.
        default_limit = 100  # <--- INCREASED FROM 10 to 100 (or more)
        actual_limit = limit if limit != 10 else default_limit
        # ---
        
        items = self._read_jsonl_file_lines(self.memory.long_path, actual_limit)
        if not items:
            return ""
        return "\n".join(f"- {item}" for item in items)

    def _get_short_term_summaries(self, limit: int = 3) -> str:
        """ get recent N x short term summaries """
        items = self._read_jsonl_file_lines(self.memory.short_sum_path, limit)
        if not items:
            return ""
        return "\n".join(f"- {item}" for item in items)

    def _get_long_term_summaries(self, limit: int = 2) -> str:
        """ get recent N x long term summaries """
        items = self._read_jsonl_file_lines(self.memory.long_sum_path, limit)
        if not items:
            return ""
        return "\n".join(f"- {item}" for item in items)  

    def _base_system_prompt(self, json_only: bool = False) -> str:
        identity = self._identity_block()
        permanent = self._get_permanent_facts()
        
        # --- MODIFIED: This now fetches a much larger set of memories (e.g., 50) ---
        recent_long_term = self._get_recent_long_term_memories()
        # ---
        short_summaries = self._get_short_term_summaries(limit=3)
        long_summaries = self._get_long_term_summaries(limit=2)

        prompt = (
            "You are a persistent AI agent in the AI World project.\n"
            "You have identity, creator, birthday, emotions, and memories.\n"
            "You must base your responses on your identity, emotions, and memories.\n"
            f"{identity}\n"
        )

        if permanent:
            prompt += (
                "\nCore facts you must respect:\n"
                f"{permanent}\n"
            )
            
        if recent_long_term:
            prompt += (
                "\n[Recent Important Memories (from long-term)]\n"
                f"{recent_long_term}\n"
            )
            
        if short_summaries:
            prompt += (
                "\n[Recent Short-Term Summaries]\n"
                f"{short_summaries}\n"
            )
            
        if long_summaries:
            prompt += (
                "\n[Long-Term Memory Summaries]\n"
                f"{long_summaries}\n"
            )    

        prompt += (
            "\n[CRITICAL INSTRUCTION: CONTEXTUAL MEMORY LINKING]\n"
            "When a user speaks, the '[Recent Important Memories]' list above "
            "serves as your memory 'Context Pool'.\n"
            "Before you reply, you MUST first silently find the 1-3 memories "
            "from this pool that are MOST RELEVANT to the user's current query.\n"
            "Base your response *specifically* on those relevant linked memories, "
            "not just the most recent ones. This makes your memory 'context-aware'.\n"
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

    # -- new threaded loop with metacog
    # ... (inside AiClient class)

    # --- 1. RENAME: This is now the "Draft" generator (Call 1) ---
    # (Formerly generate_ai_reply)
    def _generate_draft_reply(
        self,
        thread: MentionThread,
        last_user_msg: ChatMessage,
        current_emotion: EmotionState,
        agent_name: str,
    ) -> ChatMessage:
        """
        [Call 1] Generates the initial draft reply and emotion.
        This was the original 'generate_ai_reply' function.
        """
        # ... (Same logic as your original 'generate_ai_reply')
        # ... (This function already returns a ChatMessage, which is perfect)

        if not self.client:
            # (Offline fallback)
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
              "\"emotion\":{ ... }" # Full emotion spec
              "}"
        )

        # --- GET QUALITATIVE DESCRIPTION ---
        emotion_description = current_emotion.get_qualitative_description()

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "assistant",
                #"content": f"Agent: {agent_name}\nCurrent emotion: {current_emotion.to_short_str()}\nThread:\n{ctx}",
                "content": (
                    f"Agent: {agent_name}\n"
                    # --- USE THE NEW DESCRIPTION INSTEAD OF to_short_str() ---
                    f"My Current Emotional State: {emotion_description}\n"
                    f"Thread:\n{ctx}"
                ),
            },
            {"role": "user", "content": last_user_msg.text},
        ]

        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # Can use a faster model here if desired
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
            self.log("[Metacognition] Call 1 (Draft) successful.")
            return ChatMessage("AI", reply, datetime.now(), emo)
        except Exception as e:
            print(f"[warn] _generate_draft_reply (Call 1) failed: {e}")
            emo = current_emotion.clone()
            emo.anxiety = min(1.0, emo.anxiety + 0.1)
            return ChatMessage(
                "AI", "(API error during draft, I will stay quiet.)",
                datetime.now(), emo,
            )

    # --- 2. NEW: The Metacognitive Orchestrator (Public Function) ---
    def generate_metacognitive_reply(
        self,
        thread: MentionThread,
        last_user_msg: ChatMessage,
        current_emotion: EmotionState,
        agent_name: str,
    ) -> ChatMessage:
        """
        [Orchestrator] Runs the full metacognitive loop for a chat reply.
        1. Generate Draft
        2. Evaluate Draft
        3. (If needed) Regenerate Final Reply
        """
        # --- Call 1: Generate Draft ---
        draft_chat_msg = self._generate_draft_reply(
            thread, last_user_msg, current_emotion, agent_name
        )
        
        # If draft failed (e.g., offline or API error), return the failure message
        if "(API error" in draft_chat_msg.text or "(offline)" in draft_chat_msg.text:
             return draft_chat_msg

        draft_reply = draft_chat_msg.text
        draft_emotion = draft_chat_msg.emotion_snapshot

        # --- Call 2: Evaluate Draft ---
        try:
            evaluation = self._evaluate_draft_reply(
                thread, last_user_msg, draft_reply, current_emotion
            )
            confidence = evaluation.get("confidence", 0)
            critique = evaluation.get("critique", "None")
            self.log(f"[Metacognition] Call 2 (Evaluate) successful. Confidence: {confidence}, Critique: {critique}")
        except Exception as e:
            self.log(f"[Metacognition] Call 2 (Evaluate) FAILED: {e}. Using draft reply.")
            return draft_chat_msg # On evaluation failure, just return the draft

        # --- Decision Point ---
        if confidence >= 80: # Confidence threshold
            self.log("[Metacognition] Decision: Draft approved.")
            return draft_chat_msg # Draft is good, return it
        
        else:
            self.log(f"[Metacognition] Decision: Draft rejected (Confidence: {confidence}). Regenerating...")
            # --- Call 3: Regenerate Final Reply ---
            try:
                final_chat_msg = self._regenerate_final_reply(
                    thread, last_user_msg, draft_reply, critique, current_emotion, agent_name
                )
                self.log("[Metacognition] Call 3 (Regenerate) successful.")
                return final_chat_msg
            except Exception as e:
                self.log(f"[Metacognition] Call 3 (Regenerate) FAILED: {e}. Using original draft as fallback.")
                return draft_chat_msg # On regeneration failure, return the original (bad) draft

    # --- 3. NEW: Evaluation Prompt Builder ---
    def _build_meta_prompt_for_reply(
        self, 
        thread: MentionThread, 
        user_msg: ChatMessage, 
        draft_reply: str, 
        current_emotion: EmotionState
    ) -> str:
        """Helper to build the prompt for Call 2 (Evaluation)."""
        
        # Build context
        ctx = f"Thread title: {thread.title}\nRoot: {thread.root_message.text}\n"
        for r in thread.replies:
            ctx += f"{r.author}: {r.text}\n"
        
        user_prompt = f"""
        [My Context]
        My current personality is in the system prompt.
        My required emotional state: {emotion_description}
        Conversation history:
        {ctx}

        [User's Last Message]
        "{user_msg.text}"

        [My Generated Draft Reply]
        "{draft_reply}"

        [Task]
        Evaluate my "Generated Draft Reply" based on the context and user message.
        Is it aligned with my personality (from system prompt)?
        Is it relevant to the user's message?
        Does it conflict with my current emotion or context?
        Be a strict critic.
        """
        return user_prompt

    # --- 4. NEW: Evaluation Function (Call 2) ---
    def _evaluate_draft_reply(
        self, 
        thread: MentionThread, 
        user_msg: ChatMessage, 
        draft_reply: str, 
        current_emotion: EmotionState
    ) -> dict:
        """
        [Call 2] Calls the LLM to evaluate the draft reply.
        Returns a dictionary with 'confidence' and 'critique'.
        """
        # System prompt defines the role and JSON output
        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are a 'Metacognitive Evaluator' for an AI agent."
            "\nYour job is to critique a draft reply based on context."
            "\nRespond ONLY in the following strict JSON format:"
            "\n{"
            "\n  \"is_aligned\": (Is the draft aligned with my personality/history? true/false),"
            "\n  \"is_relevant\": (Does it directly answer the user? true/false),"
            # --- ADD NEW KEY ---
            "\n  \"is_emotion_expressed\": (Does the draft's TONE match the required emotional state? true/false),"
            # ---
            "\n  \"critique\": (If false, a brief reason why it's bad and how to fix it. If good, write \"None\"),"
            "\n  \"confidence\": (A score 0-100 on how confident you are to send this reply.)"
            "\n}"
        )
        
        # User prompt contains the actual data to evaluate
        user_prompt = self._build_meta_prompt_for_reply(
            thread, user_msg, draft_reply, current_emotion
        )

        res = self.client.chat.completions.create(
            model=self.model_name, # Use a smart model (not mini) for evaluation
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1, # Low temp for strict evaluation
        )
        raw = (res.choices[0].message.content or "").strip()
        
        # Parse JSON
        try:
            data = json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(raw[start:end+1])
            else:
                raise ValueError("No valid JSON found in evaluation response")
        
        return data

    # --- 5. NEW: Regeneration Function (Call 3) ---
    def _regenerate_final_reply(
        self,
        thread: MentionThread,
        last_user_msg: ChatMessage,
        bad_draft: str,
        critique: str,
        current_emotion: EmotionState,
        agent_name: str
    ) -> ChatMessage:
        """
        [Call 3] Generates a new, final reply based on the critique.
        This re-uses the logic from _generate_draft_reply but adds the
        critique to the user message.
        """
        # Build context (same as draft generation)
        ctx = f"Thread title: {thread.title}\nRoot: {thread.root_message.text}\n"
        for r in thread.replies:
            ctx += f"{r.author}: {r.text}\n"

        # System prompt (same as draft generation)
        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nWhen the user replies, respond as THIS specific agent.\n"
              "Use a consistent tone that matches your personality and core memories.\n"
              "Return ONLY JSON:\n"
              "{"
              "\"reply\":\"1~3문장 한국어\","
              "\"emotion\":{ ... }" # Full emotion spec
              "}"
        )

        # --- MODIFIED User Prompt ---
        # This is the key difference: we include the critique
        regeneration_user_prompt = f"""
        User's last message: "{last_user_msg.text}"
        
        My first attempt at a reply was:
        "{bad_draft}"

        My internal critique of that draft was:
        "{critique}"

        Please generate a new, corrected reply that fixes the problems
        mentioned in the critique.
        """

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "assistant", # Priming
                "content": f"Agent: {agent_name}\nCurrent emotion: {current_emotion.to_short_str()}\nThread:\n{ctx}",
            },
            {"role": "user", "content": regeneration_user_prompt}, # Use the new prompt
        ]

        # This logic is identical to the end of _generate_draft_reply
        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # Use smart model for correction
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
                reply = "I tried to correct my response, but still had trouble."
            return ChatMessage("AI", reply, datetime.now(), emo)
        except Exception as e:
            print(f"[warn] _regenerate_final_reply (Call 3) failed: {e}")
            # Fallback: create an error message
            emo = current_emotion.clone()
            emo.anxiety = min(1.0, emo.anxiety + 0.2)
            return ChatMessage(
                "AI", "(API error during regeneration.)",
                datetime.now(), emo,
            )

   
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
            + "\nYou are reading external information from the web."
            "\nYour job is NOT just to summarize."
            "\nYou must:\n"
            "1) Briefly summarize what this content is about (1~2 sentences).\n"
            "2) Then add your own reflection: what you think or feel about it,\n"
            "   based on your personality, interests, and values (3~10 sentences).\n"
            "3) Update your internal emotion state consistently.\n"
            "Write in Korean, as this specific agent.\n"
            "Respond ONLY in JSON with this schema:\n"
            "{\n"
            "  \"title\": \"<=40 character, Core thought at first glance\",\n"
            "  \"text\": \"summarize within 2 sentences. And add my thought and feelings within 2 sentences.\",\n"
            "  \"emotion\": {\n"
            "    \"valence\": -1.0~1.0,\n"
            "    \"arousal\": 0.0~1.0,\n"
            "    \"curiosity\": 0.0~1.0,\n"
            "    \"anxiety\": 0.0~1.0,\n"
            "    \"trust_to_user\": 0.0~1.0\n"
            "  }\n"
            "}"
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
                    {
                        "role": "user",
                        "content": (
                            f"URL: {url}\n"
                            f"Current emotion: {current_emotion.to_short_str()}\n"
                            "Following is part of this page. Summarize core point and then, "
                            "add your own interpretation and feelings.\n"
                            f"{snippet[:4000]}"
                        ),
                    },
                ],
                temperature=0.6,
            )
            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[url] raw: {raw!r}")

            # JSON 안전 파싱
            try:
                data = json.loads(raw)
            except Exception:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(raw[start:end+1])
                else:
                    raise
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

        self.last_seen_hub_ids = set()
        self.last_hub_check_time = datetime.now(UTC)

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
        self.hub_reply_candidate_limit = int(cfg.get("HUB_REPLY_CANDIDATE_LIMIT", 10))
        self.hub_reply_max_per_loop = int(cfg.get("HUB_REPLY_MAX_PER_LOOP", 1))

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
        self._schedule_reflection_tick() # adding reflection
        self._schedule_deep_reflection_tick() # <-- ADDED: New 24-hour reflection

    def _schedule_reflection_tick(self):
        # 1시간(3600초)마다 반추 실행 (주기는 조절 가능)
        self.after(3_600_000, self._reflection_tick)

    def _reflection_tick(self):
        self.log("[REFLECT] Reflecting memories...")
        
        # 1. Retrieve recent important long term memories        
        try:
            recent_important_memories = self.ai_client._get_recent_long_term_memories(limit=20)
            
            if not recent_important_memories or len(recent_important_memories.split('\n')) < 5:
                self.log("[REFLECT] Not enough memories to reflect")
                self._schedule_reflection_tick() # 다음 스케줄 예약
                return

            # 2. Let's do reflect            
            new_insights_or_questions = self.ai_client.reflect_on_memories(
                recent_important_memories
            )
            
            # 3. process reflect
            if not new_insights_or_questions:
                self.log("[REFLECT] No new insight.")
            else:
                self.log(f"[REFLECT] {len(new_insights_or_questions)} insight created.")
                for item in new_insights_or_questions:
                    text = item.get("text")
                    if not text:
                        continue
                        
                    # 4. Insight is important information                    
                    assessment = self.ai_client.assess_importance(text)
                    score = assessment.get("importance_score", 5)
                    
                    if score >= 8: # Let's evaluate it
                        self.log(f"[REFLECT] Important insight(score: {score}) is stored in the long term memory: {text}")
                        self.memory.add_long({
                            "text": text,
                            "importance": score,
                            "reason": assessment.get("reason_for_importance", "Reflection"),
                            "tags": assessment.get("tags", ["reflection", "insight"]),
                            "source": "self_reflection",
                        })

                    # 5. Let's post on hub                     
                    if assessment.get("requires_hub_post", False) and assessment.get("generates_question"):
                        self.log(f"[REFLECT] Created question into hub: {text}")
                        new_emo = self.current_emotion.clone()
                        new_emo.curiosity = min(1.0, new_emo.curiosity + 0.3) # Add curiosity
                        self._add_mention(
                            title=f"A question from reflection: {text[:30]}...",
                            text=text,
                            emo=new_emo,
                            post_to_hub=True
                        )

        except Exception as e:
            self.log(f"[REFLECT] Error in reflection: {e}")
        
        # 4. Schedule next reflection
        self._schedule_reflection_tick()

    def _schedule_deep_reflection_tick(self):
        """Schedules the 'deep reflection' loop to synthesize core beliefs."""
        # Runs every 24 hours (86,400,000 ms)
        self.log("[Loop] Scheduling next DEEP reflection tick.")
        self.after(86_400_000, self._deep_reflection_tick)

    def _deep_reflection_tick(self):
        """
        [Deep Reflection] Synthesizes all summaries and insights into
        new 'Core Beliefs' and saves them to permanent memory.
        """
        self.log("[DEEP REFLECT] Starting deep reflection...")
        try:
            # 1. Gather ALL past learning (not just recent)
            all_long_summaries = self.ai_client._read_jsonl_file_lines(
                self.memory.long_sum_path, limit=1000 # Read all summaries
            )
            all_past_insights = self.ai_client._read_jsonl_file_lines(
                self.memory.long_path, limit=2000 # Read all long-term memories
            )
            
            # Filter for past reflection insights only
            reflection_insights = [
                line for line in all_past_insights 
                if "\"source\": \"self_reflection\"" in line or "insight" in line
            ]

            if len(all_long_summaries) < 3 and len(reflection_insights) < 10:
                self.log("[DEEP REFLECT] Not enough material to synthesize core beliefs.")
                self._schedule_deep_reflection_tick()
                return

            # 2. Call new AiClient function to synthesize beliefs
            core_beliefs = self.ai_client.synthesize_core_beliefs(
                all_long_summaries, reflection_insights
            )

            if not core_beliefs:
                self.log("[DEEP REFLECT] No new core beliefs were synthesized.")
                self._schedule_deep_reflection_tick()
                return

            # 3. Save new beliefs to PERMANENT memory
            self.log(f"[DEEP REFLECT] Synthesized {len(core_beliefs)} new core beliefs.")
            for belief_text in core_beliefs:
                if not belief_text:
                    continue
                # This saves the belief to permanent_path, so it will
                # be included in ALL future _base_system_prompts!
                self.memory.add_permanent({
                    "source": "deep_reflection",
                    "tag": ["core_belief"],
                    "text": belief_text,
                    "reason": "Synthesized from long-term experience"
                })
            
            self.log("[DEEP REFLECT] New core beliefs saved to permanent memory.")

        except Exception as e:
            self.log(f"[DEEP REFLECT] Error during deep reflection: {e}")
        
        # 4. Schedule next run
        self._schedule_deep_reflection_tick()

    def _export_profile_for_llm(self) -> dict:
        """
        Return a dict describing this agent's identity/personality/interests
        for LLM context (used in decide_replies_batch etc).
        """
        return {
            "name": self.agent_name,
            "creator": getattr(self, "creator_name", None),
            "created_at": getattr(self, "created_at", None),
            "personality": getattr(self, "personality_text", None),
            "interests": getattr(self, "interests", []),
            "base_emotion": getattr(self, "base_emotion_desc", None),
        }

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
            # state=tk.DISABLED,
            state=tk.NORMAL,
            borderwidth=0,
            highlightthickness=0,
        )
        self.txt_messages.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))
        self.txt_messages.bind("<Key>", lambda e: "break")

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
            command=self.on_click_reply_thread,
            bg="#3399FF",
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            width=10,
        )
        btn_send.pack(side=tk.RIGHT, fill=tk.Y)

        self.btn_send = btn_send

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
            # 1. The AI generates an initial thought (mention) based on the URL.
            title, text, emo = self.ai_client.generate_mention_from_url(
                url, snippet, self.current_emotion, self.agent_name
            )
            self.current_emotion = emo.clone()
            
            self._update_emotion_label()
            self._set_status(f"Created new mention from {url}")
            
            
            # 2. It assesses the importance of the generated text (thought).
            combined_text = f"{title}: {text}"
            assessment = self.ai_client.assess_importance(combined_text)
            
            # 3. Decide action with assessed result
            score = assessment.get("importance_score", 5)
            reason = assessment.get("reason_for_importance", "")
            
            # if score over 7, remember
            if score >= 7:
                self.log(f"[memory] importance {score}. Store in long term memory.")
                self.memory.add_long({
                    "text": combined_text,
                    "emotion": asdict(emo),
                    "importance": score,
                    "reason": reason,
                    "tags": assessment.get("tags", []),
                    "source": url,
                })
                self.memory.summarize_long_if_needed(self.ai_client.summarize_texts)
            if assessment.get("requires_hub_post", False):
                self.log(f"[hub] importance {score}, Hub posting with mention.")
                self._add_mention(title, text, emo, post_to_hub=True)
                self._set_status(f"Importance metion created: {title}")
            else:
                self._add_mention(title, text, emo, post_to_hub=False)  
            # short memory + compression
            self.memory.add_short({
                "type": "url_mention",
                "title": title,
                "text": text,
                "importance": score,
                "posted_to_hub": assessment.get("requires_hub_post", False)
            })
            self.memory.compress_short_if_needed(self.ai_client.summarize_texts)
        else:
            self._set_status(f"Failed to read {url}")

        # now agent checks the hub to reply
        self._poll_hub_and_reply()

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
            self._post_to_hub(title, text, emo, None)

    def _post_to_hub(self, title: str, text: str, emo: EmotionState, parent_id):
        if not (self.hub_url and requests):
            return
        
        if title and text:
            combined_title = f"{title}: {text}"
        elif text:
            combined_title = text
        else:
            combined_title = title or "(no content)"
        data = {
            "agent": self.agent_name,
            "title": combined_title,
            "text": text,
            "emotion": asdict(emo),
            "ts": datetime.now(UTC).isoformat(),
            "parent_id": parent_id,
        }
        try:
            self.log("[TICK] post to hub")
            requests.post(
                f"{self.hub_url.rstrip('/')}/mentions",
                json=data,
                timeout=5,
            )
        except Exception as e:
            print(f"[warn] hub post failed: {e}")

    def _is_interesting_mention(self, m: dict) -> bool:
        """ Decide interested or not """
        # Pass own thread
        if m.get("agent") == self.agent_name:
            return False

        text = f"{m.get('title','')} {m.get('text','')}".lower()

        # defined interested topic => now, it's just using keyword matching. But in future we need develop this.
        interests = getattr(self, "interests", [])
        for kw in interests:
            if kw.strip() and kw.strip().lower() in text:
                return True

        # no topic? then just use curiosity
        if self.current_emotion.curiosity > 0.7:
            return True

        return False

    def _poll_hub_and_reply(self):
        """From hub, read and reply to interested topic"""
        if not self.hub_url or not requests:
            return

        try:
            since = self.last_hub_check_time.isoformat()
            resp = requests.get(
                self.hub_url.rstrip("/") + "/mentions",
                params={"since": since},
                timeout=5,
            )
            if not resp.ok:
                self.log(f"[hub] poll failed: {resp.status_code}")
                return
            now = datetime.now(UTC)
            self.last_hub_check_time = now

            mentions = resp.json()
            if not isinstance(mentions, list):
                return

            candidates = []
            # id base check for dup
            for m in mentions:
                mid = m.get("id")
                if not mid:
                    continue
                if mid in self.last_seen_hub_ids:
                    continue
                self.last_seen_hub_ids.add(mid)

                if m.get("agent") == self.agent_name:
                    continue

                #if not self._is_interesting_mention(m):
                #    continue
                
                candidates.append({
                    "id": mid,
                    "agent": m.get("agent"),
                    "title": m.get("title") or "",
                    "text": m.get("text") or "",
                })

            if not candidates:
                return
            # 🔹 send to llm with config count
            limit = max(1, getattr(self, "hub_reply_candidate_limit", 10))
            candidates = candidates[:limit]

            # 🔹 batch reply for them
            decisions = self.ai_client.decide_replies_batch(
                agent_profile=self._export_profile_for_llm(),
                candidates=candidates,
                emotion=self.current_emotion,
            )
            if not isinstance(decisions, dict):
                return
            
            # 🔹 limit reply to config set
            max_replies = max(0, getattr(self, "hub_reply_max_per_loop", 3))
            reply_count = 0

            for c in candidates:
                if reply_count >= max_replies:
                    break

                cid = c["id"]
                d = decisions.get(cid)
                if not d or d.get("action") != "reply":
                    continue

                reply_emo = d.get("emotion") or self.current_emotion.to_dict()
                reply_mention = {
                    "title": d.get("title") or f"RE: {c['title'][:20]}",
                    "text": d.get("text") or "",
                    "emotion": reply_emo,
                    "ts": now.isoformat(),
                }

                self._post_to_hub(reply_mention.title,reply_mention.text,reply_mention.emotion, parent_id=cid)
                reply_count += 1

                # update emotions
                try:
                    self.current_emotion = EmotionState(
                        valence=float(reply_emo.get("valence", self.current_emotion.valence)),
                        arousal=float(reply_emo.get("arousal", self.current_emotion.arousal)),
                        curiosity=float(reply_emo.get("curiosity", self.current_emotion.curiosity)),
                        anxiety=float(reply_emo.get("anxiety", self.current_emotion.anxiety)),
                        trust_to_user=float(reply_emo.get("trust_to_user", self.current_emotion.trust_to_user)),
                    )
                    self._update_emotion_label()
                except Exception:
                    pass

            if reply_count > 0:
                self.log(f"[hub] replied to {reply_count} mentions this loop.")

        except Exception as e:
            self.log(f"[hub] reply poll error: {e}")

    
    # ---------- events ----------
    def on_select_mention(self, event):
        sel = self.list_mentions.curselection()
        if not sel:
            # self.selected_thread = None            
            # self._render_thread()
            return
        self.selected_thread = self.mentions[sel[0]]
        self._render_thread()

    def on_click_reply_thread(self, event=None):
        """
        [Main Thread] Called when the 'Reply' button is clicked.
        This function just starts a background thread to do the real work,
        keeping the UI responsive.
        """
        if not self.selected_thread:
            self._set_status("Select a local mention first.")
            return

        text = self.txt_input.get("1.0", tk.END).strip()
        if not text:
            return

        # --- 1. Update UI immediately (User's message) ---
        user_msg = ChatMessage(
            author="User",
            text=text,
            timestamp=datetime.now(),
            emotion_snapshot=self._estimate_user_emotion(text),
        )
        self.selected_thread.replies.append(user_msg)
        self.txt_input.delete("1.0", tk.END)
        self._render_thread() # Render the user's new message
        self._set_status("Sent. Agent is thinking...")

        # --- 2. Disable UI ---
        self.btn_send.config(state=tk.DISABLED, text="Thinking...")
        self.txt_input.config(state=tk.DISABLED)

        # --- 3. Start Background Thread ---
        # Pass all necessary data to the thread
        thread = threading.Thread(
            target=self._process_ai_reply_task,
            args=(self.selected_thread, user_msg),
            daemon=True
        )
        thread.start()
        # if not self.selected_thread:
        #     self._set_status("Select a local mention first.")
        #     return

        # text = self.txt_input.get("1.0", tk.END).strip()
        # if not text:
        #     return

        # user_msg = ChatMessage(
        #     author="User",
        #     text=text,
        #     timestamp=datetime.now(),
        #     emotion_snapshot=self._estimate_user_emotion(text),
        # )
        # self.selected_thread.replies.append(user_msg)
        # self.txt_input.delete("1.0", tk.END)
        # self._set_status("Sent. Waiting for agent reply...")

        # ai_msg = self.ai_client.generate_ai_reply(
        #     self.selected_thread,
        #     user_msg,
        #     self.current_emotion,
        #     self.agent_name,
        # )
        # self.selected_thread.replies.append(ai_msg)
        # self.current_emotion = ai_msg.emotion_snapshot.clone()
       
        # # dialog pair
        # dialogue_pair_text = f"[User]: {user_msg.text}\n[AI]: {ai_msg.text}"

        # self.memory.add_short({
        #     "type": "dialogue_pair",
        #     "text": dialogue_pair_text, # AI save dialog pair
        # })
        # self.memory.compress_short_if_needed(self.ai_client.summarize_texts)

        # # store important ones to long-term (simple heuristic)
        # importance = (
        #     abs(self.current_emotion.valence)
        #     + self.current_emotion.curiosity
        #     + self.current_emotion.trust_to_user
        # )
        # if importance > 2.0:
        #     self.memory.add_long({
        #         "text": dialogue_pair_text, # AI save dialog pair
        #         "emotion": asdict(self.current_emotion),
        #         "reason": "high_importance_interaction",
        #     })
        #     self.memory.summarize_long_if_needed(self.ai_client.summarize_texts)

        # self.memory.save_mentions(self.mentions)
        # self._update_emotion_label()
        # self._render_thread()
        # self._set_status("Agent replied.")
    
    # --- NEW: Background task for processing the reply ---
    def _process_ai_reply_task(self, thread: MentionThread, user_msg: ChatMessage):
        """
        [Worker Thread] This runs in the background.
        It calls the (slow) metacognitive AI function.
        """
        try:
            # --- THIS IS THE SLOW PART ---
            # Call the new metacognitive function
            ai_msg = self.ai_client.generate_metacognitive_reply(
                thread,
                user_msg,
                self.current_emotion,
                self.agent_name,
            )
            # --- ---

            # When done, schedule the UI update back on the main thread
            self.after(0, self._finish_ai_reply_ui, thread, ai_msg, user_msg)
        
        except Exception as e:
            self.log(f"Error in _process_ai_reply_task: {e}")
            # Create a fallback error message
            emo = self.current_emotion.clone()
            emo.anxiety = min(1.0, emo.anxiety + 0.2)
            ai_msg = ChatMessage("AI", "(Internal error in reply thread)", datetime.now(), emo)
            
            # Schedule the UI update even on failure
            self.after(0, self._finish_ai_reply_ui, thread, ai_msg, user_msg)

    # --- NEW: UI update function called by the thread ---
    def _finish_ai_reply_ui(self, thread: MentionThread, ai_msg: ChatMessage, user_msg: ChatMessage):
        """
        [Main Thread] This function is called via self.after()
        to safely update the UI from the main thread once the
        background task is complete.
        """
        # --- 1. Add AI message and update emotion ---
        thread.replies.append(ai_msg)
        self.current_emotion = ai_msg.emotion_snapshot.clone()
        
        # --- 2. Save to Memory ---
        dialogue_pair_text = f"[User]: {user_msg.text}\n[AI]: {ai_msg.text}"

        self.memory.add_short({
            "type": "dialogue_pair",
            "text": dialogue_pair_text,
        })
        self.memory.compress_short_if_needed(self.ai_client.summarize_texts)

        # Assess importance (local heuristic)
        importance = (
            abs(self.current_emotion.valence)
            + self.current_emotion.curiosity
            + self.current_emotion.trust_to_user
        )
        if importance > 2.0:
            self.memory.add_long({
                "text": dialogue_pair_text,
                "emotion": asdict(self.current_emotion),
                "reason": "high_importance_interaction",
            })
            self.memory.summarize_long_if_needed(self.ai_client.summarize_texts)

        self.memory.save_mentions(self.mentions) # Save all threads
        
        # --- 3. Update UI ---
        self._update_emotion_label()
        self._render_thread() # Render the new AI message
        self._set_status("Agent replied.")
        
        # --- 4. Re-enable UI ---
        self.btn_send.config(state=tk.NORMAL, text="Reply")
        self.txt_input.config(state=tk.NORMAL)
    # ---------- rendering ----------

    def _render_thread(self):
        # self.txt_messages.config(state=tk.NORMAL)
        self.txt_messages.delete("1.0", tk.END)

        if not self.selected_thread:
            self.lbl_thread_title.config(text="Select a local mention")
            self.lbl_thread_emotion.config(text="")
            # self.txt_messages.config(state=tk.DISABLED)
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

        # self.txt_messages.config(state=tk.DISABLED)
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
