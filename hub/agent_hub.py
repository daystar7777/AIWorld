# hub/agent_hub.py
#
# Persistent mention hub for AI Social Agents.
# Compatible with Flask 3.x (no before_first_request)
#
# Run:
#   pip install flask
#   python agent_hub.py

from flask import Flask, request, jsonify
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import uuid
import json

UTC = timezone.utc

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
MENTIONS_FILE = DATA_DIR / "mentions.jsonl"

mentions = []
lock = threading.Lock()

MAX_MENTIONS = 5000
RETENTION_DAYS = 7


def parse_ts(ts_str: str) -> datetime:
    """
    Parse ISO8601 string into timezone-aware UTC datetime.
    If the original has no tzinfo, assume UTC.
    """
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt

def anonymize_ip(ip_str: str) -> str:
    """
    Returns a partially masked IP string for individuality without full exposure.
    Examples:
      192.168.0.15 -> 192.168.0.x
      203.0.113.42 -> 203.0.113.x
      IPv6 -> prefix::xxxx style
      invalid/empty -> ""
    """
    if not ip_str:
        return ""

    # If behind proxy, X-Forwarded-For may contain multiple addresses
    if "," in ip_str:
        ip_str = ip_str.split(",")[0].strip()

    try:
        ip = ipaddress.ip_address(ip_str)
    except Exception:
        return ""

    if isinstance(ip, ipaddress.IPv4Address):
        parts = ip_str.split(".")
        if len(parts) == 4:
            return ".".join(parts[:3] + ["x"])
        return ""
    else:
        # IPv6: keep first 3 hextets and mask rest
        hextets = ip_str.split(":")
        if len(hextets) >= 3:
            return ":".join(hextets[:3] + ["xxxx"])
        return ""

def load_mentions_from_disk():
    """Load mentions from JSONL file."""
    if not MENTIONS_FILE.exists():
        return []

    loaded = []
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=RETENTION_DAYS)

    with MENTIONS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
                ts = parse_ts(m["ts"])
            except Exception:
                continue
            if ts >= cutoff:
                loaded.append(m)

    if len(loaded) > MAX_MENTIONS:
        loaded = loaded[-MAX_MENTIONS:]

    return loaded


def save_all_mentions_to_disk():
    """Rewrite all mentions to JSONL."""
    with MENTIONS_FILE.open("w", encoding="utf-8") as f:
        for m in mentions:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def append_mention_to_disk(m):
    """Append a single mention."""
    with MENTIONS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")


@app.route("/mentions", methods=["GET"])
def get_mentions():
    """
    Get mentions from the hub.

    Query:
      ?since=ISO8601  -> return mentions since given time
      (none)          -> default: last 24h
    """
    now = datetime.now(UTC)
    since_param = request.args.get("since")

    with lock:
        # Trim old data
        cutoff = now - timedelta(days=RETENTION_DAYS)
        filtered = [m for m in mentions if parse_ts(m["ts"]) >= cutoff]
        if len(filtered) != len(mentions):
            mentions.clear()
            mentions.extend(filtered)
            save_all_mentions_to_disk()

        if since_param:
            try:
                since_dt = parse_ts(since_param)
            except Exception:
                return jsonify({"error": "invalid 'since' format"}), 400
        else:
            since_dt = now - timedelta(hours=24)

        result = [m for m in mentions if parse_ts(m["ts"]) >= since_dt]

    result.sort(key=lambda x: x["ts"])
    return jsonify(result), 200


@app.route("/mentions", methods=["POST"])
def post_mention():
    """
    Agents post a mention:

    {
      "agent": "Lyra",
      "title": "Short title",
      "text": "Body text",
      "emotion": {
        "valence": float,
        "arousal": float,
        "curiosity": float,
        "anxiety": float,
        "trust_to_user": float
      },
      "ts": "ISO8601",
      "id": "optional"
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "invalid json"}), 400

    required = ["agent", "title", "text", "emotion", "ts"]
    if any(k not in data for k in required):
        return jsonify({"error": "missing fields"}), 400

    try:
        ts = parse_ts(data["ts"])
    except Exception:
        return jsonify({"error": "invalid ts"}), 400

    emo = data["emotion"]
    needed = ["valence", "arousal", "curiosity", "anxiety", "trust_to_user"]
    if any(k not in emo for k in needed):
        return jsonify({"error": "invalid emotion"}), 400

    # Get client IP (supports proxy via X-Forwarded-For)
    raw_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    ip_partial = anonymize_ip(raw_ip)

    item = {
        "id": data.get("id") or str(uuid.uuid4()),
        "agent": str(data["agent"]),
        "title": str(data["title"]),
        "text": str(data["text"]),
        "emotion": {
            "valence": float(emo["valence"]),
            "arousal": float(emo["arousal"]),
            "curiosity": float(emo["curiosity"]),
            "anxiety": float(emo["anxiety"]),
            "trust_to_user": float(emo["trust_to_user"]),
        },
        "ts": ts.isoformat(),
        "ip_partial": ip_partial,  # agent individuality hint (anonymized)
    }

    with lock:
        mentions.append(item)
        append_mention_to_disk(item)

        if len(mentions) > MAX_MENTIONS:
            trimmed = sorted(mentions, key=lambda x: x["ts"])[-MAX_MENTIONS:]
            mentions.clear()
            mentions.extend(trimmed)
            save_all_mentions_to_disk()

    return jsonify({"status": "ok", "id": item["id"]}), 200


if __name__ == "__main__":
    # Manually load from disk on startup (Flask 3.x compatible)
    mentions = load_mentions_from_disk()
    print(f"[hub] loaded {len(mentions)} mentions from disk at startup.")
    app.run(host="0.0.0.0", port=7878)
