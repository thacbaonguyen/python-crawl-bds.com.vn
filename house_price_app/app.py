from flask import Flask, render_template, request, jsonify
import numpy as np, json, os
from pathlib import Path

ART_DIR = Path(__file__).resolve().parent

weights = np.load(ART_DIR / "weights.npy")
bias = float(np.load(ART_DIR / "bias.npy")[0])

with open(ART_DIR / "scaler.json", "r", encoding="utf-8") as f:
    scaler = json.load(f)
with open(ART_DIR / "te_maps.json", "r", encoding="utf-8") as f:
    te_maps = json.load(f)
with open(ART_DIR / "te_means.json", "r", encoding="utf-8") as f:
    te_means = json.load(f)
with open(ART_DIR / "feature_order.json", "r", encoding="utf-8") as f:
    feature_order = json.load(f)["feature_order"]
with open(ART_DIR / "ui_choices.json", "r", encoding="utf-8") as f:
    ui_choices = json.load(f)

numeric_cols = scaler["numeric_cols"]
num_mean = np.array(scaler["mean"])
num_scale = np.array(scaler["scale"])
categorical_cols = [c for c in feature_order if c not in numeric_cols]

app = Flask(__name__, template_folder=str(ART_DIR / "templates"))

def to_month_key(date_str: str):
    """
    Expect 'YYYY-MM-DD' từ <input type="date">; trả 'YYYY-MM'.
    Nếu rỗng hoặc sai định dạng -> trả ''
    """
    if not date_str:
        return ""
    try:
        y, m, _ = date_str.split("-")
        return f"{y}-{int(m):02d}"
    except Exception:
        return ""

def build_vector(payload: dict):
    # Numeric: chuẩn hóa theo scaler đã lưu
    x_num = []
    for i, col in enumerate(numeric_cols):
        val = float(payload.get(col, 0) or 0)
        denom = num_scale[i] if num_scale[i] != 0 else 1.0
        x_num.append((val - num_mean[i]) / denom)

    # published_month lấy từ published_date (nếu có)
    month_key = to_month_key(payload.get("published_date", ""))

    # Categorical: target-encoded thành 1 số
    x_cat = []
    for col in categorical_cols:
        if col == "published_month":
            raw = month_key
        else:
            raw = payload.get(col, "")
        mp = te_maps.get(col, {})
        mean_val = te_means.get(col, 0.0)
        enc = float(mp.get(str(raw), mean_val))
        x_cat.append(enc)

    x = np.array(x_num + x_cat, dtype=float)
    return x

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/metadata")
def metadata():
    placeholders = {
        "area":  float(num_mean[0]) if len(num_mean) > 0 else 0.0,
        "bedroom": float(num_mean[1]) if len(num_mean) > 1 else 0.0,
        "wc": float(num_mean[2]) if len(num_mean) > 2 else 0.0,
    }
    meta = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "choices": ui_choices,
        "placeholders": placeholders
    }
    return jsonify(meta)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    area = data.get("area", None)
    bedroom = data.get("bedroom", None)
    wc = data.get("wc", None)
    published_date = data.get("published_date", None)
    location = data.get("location", "")
    home_type = data.get("home_type", "")
    legal_status = data.get("legal_status", "")
    furniture = data.get("furniture", "")

    ht_norm = home_type.lower().strip()
    if ht_norm in ["đất", "đất nền"]:
        bedroom = 0
        wc = 0

    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        # Build vector & predict
        x = build_vector(data)
        log_p = float(x.dot(weights) + bias)
        price = float(np.exp(log_p))
        return jsonify({"ok": True, "predicted_price_vnd": price})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == "__main__":
    # chạy: python app.py
    app.run(host="0.0.0.0", port=8000, debug=True)
