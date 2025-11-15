# agents/memory_manager.py

import os
import json
from dataclasses import asdict
from datetime import datetime, timedelta, date, timezone
from pathlib import Path

UTC = timezone.utc

# Import models used for saving/loading mentions
from models import EmotionState, ChatMessage, MentionThread

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
        # Path for the new relationship model file
        self.relationship_path = self.data_dir / "relationship_memory.json"

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
    
    # Load and save the relationship memory
    
    def load_relationships(self) -> dict:
        """
        [NEW] Loads the relationship memory file.
        Returns a dict, e.g., {"Agent-B": {"trust_score": 0.6, ...}}
        """
        if not self.relationship_path.exists():
            return {} # Return empty dict if no file
        try:
            with open(self.relationship_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Memory] Could not load relationships: {e}")
            return {}

    def save_relationships(self, relationships: dict):
        """
        [NEW] Saves the complete relationship memory dict to its file.
        This is an overwrite, not an append.
        """
        try:
            with open(self.relationship_path, "w", encoding="utf-8") as f:
                # Save with indent for human readability
                json.dump(relationships, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Memory] Could not save relationships: {e}")
    
    # Load and save the timestamp for the last philosophy synthesis
    def load_agent_state(self) -> dict:
        """Loads the entire agent state JSON."""
        if not self.state_path.exists():
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Memory] Could not load agent state: {e}")
            return {}

    def save_agent_state(self, state_data: dict):
        """Saves the entire agent state JSON."""
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            print(f"[Memory] Could not save agent state: {e}")
    
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
