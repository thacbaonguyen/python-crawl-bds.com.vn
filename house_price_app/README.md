
# House Price Linear Regression App

This folder contains:
- Trained linear model (OLS on log(price)) using:
  - Numeric features (standardized): area, bedroom, wc
  - Target-encoded categorical features: location, home_type, legal_status, furniture, published_month
- A Flask app to input original attributes and output predicted price (VND).

## How to run locally

```bash
cd house_price_app
source .venv/bin/activate  # (on Windows: .venv\Scripts\activate)
pip3 install flask numpy pandas scikit-learn
python3 app.py

cd /Users/dotung/python-crawl-bds.com.vn/house_price_app
pip3 install --user flask numpy pandas scikit-learn
python3 app.py
# open http://localhost:8000
```

The app converts your input into the same feature space:
- Standardizes numeric features with the saved means/stds
- Target-encodes categorical fields with mappings learned from the training set
- Predicts log(price) then returns price = exp(log_price)
