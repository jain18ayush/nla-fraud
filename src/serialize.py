"""
Phase 3a: Deterministic feature-vector → English text rendering.

Verbalizability table per dataset (only features the MLP actually consumes):

IEEE-CIS
  ProductCD       → product code category (W=web, H=hotel, C=card, S=service, R=retail)
  card4           → card network (visa/mastercard/discover/amex)
  card6           → card type (debit/credit/debit or credit/charge card)
  P_emaildomain   → purchaser email domain
  R_emaildomain   → recipient email domain
  p_email_group   → purchaser email group (free/corporate/anonymous/missing)
  r_email_group   → recipient email group
  DeviceType      → device type (desktop/mobile/missing)
  DeviceInfo      → device info string
  TransactionAmt  → transaction amount ($)
  addr1           → billing zip prefix
  addr2           → billing country code
  dist1           → distance (purchaser address to card address, miles)
  hour_of_day     → hour of transaction (0-23)
  day_of_week     → day of week (0=Mon, 6=Sun)
  email_match     → whether purchaser/recipient email domains match
  amt_zscore      → amount z-score vs card history
  velocity_1h     → number of transactions on this card in past 1h
  velocity_24h    → number of transactions on this card in past 24h
  card_tenure_days → days since first transaction seen on this card

Sparkov / synthetic
  merchant        → merchant name
  category        → merchant category
  amt             → transaction amount ($)
  hour_of_day     → hour (0-23)
  day_of_week     → day of week
  geo_distance_km → geographic distance between cardholder home and merchant (km)
  amt_zscore      → amount z-score vs card history
  velocity_1h     → transactions on card past 1h
  velocity_24h    → transactions on card past 24h
  card_tenure_days → days since first transaction on card
  age_at_txn      → cardholder age at transaction time (years)
"""

from __future__ import annotations

import math
from typing import Any

# ── Lookup tables ─────────────────────────────────────────────────────────────

_PRODUCT_CODE = {
    "W": "web purchase",
    "H": "hotel/lodging",
    "C": "card-present",
    "S": "service/subscription",
    "R": "retail purchase",
}

_DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_ANON_DOMAINS = {
    "protonmail.com", "guerrillamail.com", "yopmail.com", "dispostable.com",
    "mailnull.com", "sharklasers.com", "trashmail.com", "tempr.email", "cuvox.de",
    "anonaddy.com", "tutanota.com", "cock.li",
}
_FREE_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "live.com", "icloud.com", "me.com", "msn.com", "comcast.net",
}


def _missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    s = str(v).lower().strip()
    return s in {"nan", "none", "missing", "__missing__", ""}


def _hour_label(h: int | float) -> str:
    h = int(round(h)) % 24
    if 0 <= h < 6:
        return f"{h:02d}:xx (overnight)"
    if 6 <= h < 12:
        return f"{h:02d}:xx (morning)"
    if 12 <= h < 18:
        return f"{h:02d}:xx (afternoon)"
    return f"{h:02d}:xx (evening)"


def _zscore_label(z: float) -> str:
    if z > 4:
        return f"z={z:+.1f} (extremely high)"
    if z > 2:
        return f"z={z:+.1f} (high)"
    if z > 0.5:
        return f"z={z:+.1f} (above average)"
    if z < -2:
        return f"z={z:+.1f} (very low)"
    if z < -0.5:
        return f"z={z:+.1f} (below average)"
    return f"z={z:+.1f} (typical)"


# ── IEEE-CIS serializer ───────────────────────────────────────────────────────

def _serialize_ieee(row: dict) -> str:
    parts: list[str] = []

    # Product / transaction type
    prod = row.get("ProductCD")
    if not _missing(prod):
        desc = _PRODUCT_CODE.get(str(prod).upper(), f"product code {prod}")
        parts.append(f"Transaction type: {desc}")

    # Amount
    amt = row.get("TransactionAmt")
    if not _missing(amt):
        parts.append(f"Amount: ${float(amt):.2f}")

    # Time
    hour = row.get("hour_of_day")
    dow = row.get("day_of_week")
    time_parts = []
    if not _missing(hour):
        time_parts.append(_hour_label(float(hour)))
    if not _missing(dow):
        time_parts.append(_DOW[int(round(float(dow))) % 7])
    if time_parts:
        parts.append("Time: " + ", ".join(time_parts))

    # Card
    card4 = row.get("card4")
    card6 = row.get("card6")
    card_parts = []
    if not _missing(card4):
        card_parts.append(str(card4))
    if not _missing(card6):
        card_parts.append(str(card6))
    if card_parts:
        parts.append("Card: " + " ".join(card_parts))

    # Email domains
    p_email = row.get("P_emaildomain")
    r_email = row.get("R_emaildomain")
    p_group = row.get("p_email_group")
    r_group = row.get("r_email_group")
    if not _missing(p_email):
        anon_flag = " (privacy/anonymous)" if str(p_email).lower() in _ANON_DOMAINS else ""
        group_tag = f" [{p_group}]" if not _missing(p_group) else ""
        parts.append(f"Purchaser email: {p_email}{group_tag}{anon_flag}")
    elif not _missing(p_group):
        parts.append(f"Purchaser email: missing (group={p_group})")

    if not _missing(r_email) and r_email != p_email:
        parts.append(f"Recipient email: {r_email}")

    email_match = row.get("email_match")
    if not _missing(email_match):
        v = float(email_match)
        if v == 1.0:
            parts.append("Email match: purchaser and recipient domains match")
        elif v == 0.0:
            parts.append("Email match: purchaser and recipient domains differ")

    # Device
    dev_type = row.get("DeviceType")
    dev_info = row.get("DeviceInfo")
    dev_parts = []
    if not _missing(dev_type):
        dev_parts.append(str(dev_type))
    if not _missing(dev_info) and str(dev_info) not in {"missing", "__missing__"}:
        dev_parts.append(f"({dev_info})")
    if dev_parts:
        parts.append("Device: " + " ".join(dev_parts))

    # Geography
    addr1 = row.get("addr1")
    addr2 = row.get("addr2")
    dist1 = row.get("dist1")
    geo_parts = []
    if not _missing(addr2):
        geo_parts.append(f"country code {int(float(addr2))}" if str(addr2).replace(".", "").isdigit() else str(addr2))
    if not _missing(addr1):
        geo_parts.append(f"billing zip prefix {int(float(addr1))}")
    if not _missing(dist1) and float(dist1) > 0:
        geo_parts.append(f"address–card distance {float(dist1):.0f} mi")
    if geo_parts:
        parts.append("Geography: " + ", ".join(geo_parts))

    # Behavioral / velocity
    v1h = row.get("velocity_1h")
    v24h = row.get("velocity_24h")
    if not _missing(v1h):
        parts.append(f"Card velocity: {int(round(float(v1h)))} txn in past 1h")
    if not _missing(v24h):
        parts.append(f"Card velocity: {int(round(float(v24h)))} txn in past 24h")

    amt_z = row.get("amt_zscore")
    if not _missing(amt_z):
        parts.append(f"Amount vs card history: {_zscore_label(float(amt_z))}")

    tenure = row.get("card_tenure_days")
    if not _missing(tenure):
        t = float(tenure)
        if t < 1:
            parts.append("Card tenure: new card (first seen today)")
        elif t < 7:
            parts.append(f"Card tenure: {t:.0f} day(s)")
        elif t < 30:
            parts.append(f"Card tenure: {t:.0f} days ({t/7:.1f} weeks)")
        else:
            parts.append(f"Card tenure: {t:.0f} days ({t/30:.1f} months)")

    # Model score
    score = row.get("fraud_score")
    if not _missing(score):
        parts.append(f"Model fraud score: {float(score):.3f}")

    return "; ".join(parts) + "."


# ── Sparkov / synthetic serializer ────────────────────────────────────────────

_MERCHANT_CATEGORY_MAP = {
    "gas_transport": "gas station / transport",
    "grocery_pos": "grocery store (in-person)",
    "grocery_net": "grocery store (online)",
    "home": "home goods",
    "entertainment": "entertainment",
    "food_dining": "food & dining",
    "personal_care": "personal care",
    "health_fitness": "health & fitness",
    "shopping_pos": "retail shopping (in-person)",
    "shopping_net": "retail shopping (online)",
    "misc_pos": "misc retail (in-person)",
    "misc_net": "misc retail (online)",
    "travel": "travel",
    "kids_pets": "kids & pets",
}


def _serialize_sparkov(row: dict) -> str:
    parts: list[str] = []

    merchant = row.get("merchant")
    category = row.get("category")
    if not _missing(merchant):
        cat_desc = ""
        if not _missing(category):
            cat_desc = " (" + _MERCHANT_CATEGORY_MAP.get(str(category), str(category)) + ")"
        parts.append(f"Merchant: {merchant}{cat_desc}")
    elif not _missing(category):
        parts.append(f"Category: " + _MERCHANT_CATEGORY_MAP.get(str(category), str(category)))

    amt = row.get("amt")
    if not _missing(amt):
        parts.append(f"Amount: ${float(amt):.2f}")

    hour = row.get("hour_of_day")
    dow = row.get("day_of_week")
    time_parts = []
    if not _missing(hour):
        time_parts.append(_hour_label(float(hour)))
    if not _missing(dow):
        time_parts.append(_DOW[int(round(float(dow))) % 7])
    if time_parts:
        parts.append("Time: " + ", ".join(time_parts))

    age = row.get("age_at_txn")
    if not _missing(age):
        parts.append(f"Cardholder age: {float(age):.0f} years")

    geo = row.get("geo_distance_km")
    if not _missing(geo):
        g = float(geo)
        if g < 5:
            parts.append(f"Distance home→merchant: {g:.1f} km (local)")
        elif g < 50:
            parts.append(f"Distance home→merchant: {g:.1f} km (nearby)")
        elif g < 500:
            parts.append(f"Distance home→merchant: {g:.1f} km (regional)")
        else:
            parts.append(f"Distance home→merchant: {g:.1f} km (far/unusual)")

    v1h = row.get("velocity_1h")
    v24h = row.get("velocity_24h")
    if not _missing(v1h):
        parts.append(f"Card velocity: {int(round(float(v1h)))} txn in past 1h")
    if not _missing(v24h):
        parts.append(f"Card velocity: {int(round(float(v24h)))} txn in past 24h")

    amt_z = row.get("amt_zscore")
    if not _missing(amt_z):
        parts.append(f"Amount vs card history: {_zscore_label(float(amt_z))}")

    tenure = row.get("card_tenure_days")
    if not _missing(tenure):
        t = float(tenure)
        if t < 1:
            parts.append("Card tenure: new card")
        elif t < 30:
            parts.append(f"Card tenure: {t:.0f} days")
        else:
            parts.append(f"Card tenure: {t:.0f} days ({t/30:.1f} months)")

    score = row.get("fraud_score")
    if not _missing(score):
        parts.append(f"Model fraud score: {float(score):.3f}")

    return "; ".join(parts) + "."


# ── Public API ────────────────────────────────────────────────────────────────

_SERIALIZERS = {
    "ieee-fraud-detection": _serialize_ieee,
    "sparkov": _serialize_sparkov,
    "synthetic": _serialize_sparkov,
    "ealtman2019/credit-card-transactions": _serialize_sparkov,
}


def serialize_row(row: dict | Any, dataset_source: str) -> str:
    """Serialize one transaction row (dict or pandas Series) to English text."""
    if not isinstance(row, dict):
        row = row.to_dict()
    fn = _SERIALIZERS.get(dataset_source, _serialize_ieee)
    return fn(row)


def verbalizability_table(dataset_source: str) -> list[dict]:
    """Return the per-feature verbalizability table for the given dataset."""
    ieee_table = [
        {"feature": "ProductCD",        "verbalized_as": "transaction type (web/hotel/card-present/etc.)"},
        {"feature": "TransactionAmt",   "verbalized_as": "amount in USD"},
        {"feature": "hour_of_day",      "verbalized_as": "hour of day with time-of-day label"},
        {"feature": "day_of_week",      "verbalized_as": "day name (Monday–Sunday)"},
        {"feature": "card4",            "verbalized_as": "card network (Visa/Mastercard/etc.)"},
        {"feature": "card6",            "verbalized_as": "card type (debit/credit)"},
        {"feature": "P_emaildomain",    "verbalized_as": "purchaser email domain + anonymity flag"},
        {"feature": "R_emaildomain",    "verbalized_as": "recipient email domain"},
        {"feature": "p_email_group",    "verbalized_as": "purchaser email group (free/corporate/anon/missing)"},
        {"feature": "r_email_group",    "verbalized_as": "recipient email group"},
        {"feature": "email_match",      "verbalized_as": "whether purchaser/recipient email domains match"},
        {"feature": "DeviceType",       "verbalized_as": "device type (desktop/mobile/missing)"},
        {"feature": "DeviceInfo",       "verbalized_as": "raw device info string"},
        {"feature": "addr1",            "verbalized_as": "billing zip prefix"},
        {"feature": "addr2",            "verbalized_as": "billing country code"},
        {"feature": "dist1",            "verbalized_as": "distance (purchaser addr → card addr, miles)"},
        {"feature": "velocity_1h",      "verbalized_as": "transaction count on card in past 1h"},
        {"feature": "velocity_24h",     "verbalized_as": "transaction count on card in past 24h"},
        {"feature": "amt_zscore",       "verbalized_as": "amount z-score vs card's historical average"},
        {"feature": "card_tenure_days", "verbalized_as": "days since card first seen in data"},
    ]
    sparkov_table = [
        {"feature": "merchant",          "verbalized_as": "merchant name"},
        {"feature": "category",          "verbalized_as": "merchant category (mapped to readable name)"},
        {"feature": "amt",               "verbalized_as": "amount in USD"},
        {"feature": "hour_of_day",       "verbalized_as": "hour of day with label"},
        {"feature": "day_of_week",       "verbalized_as": "day name"},
        {"feature": "geo_distance_km",   "verbalized_as": "home-to-merchant distance with proximity label"},
        {"feature": "age_at_txn",        "verbalized_as": "cardholder age in years"},
        {"feature": "velocity_1h",       "verbalized_as": "transaction count in past 1h"},
        {"feature": "velocity_24h",      "verbalized_as": "transaction count in past 24h"},
        {"feature": "amt_zscore",        "verbalized_as": "amount z-score vs card history"},
        {"feature": "card_tenure_days",  "verbalized_as": "days since first card transaction"},
    ]
    tables = {
        "ieee-fraud-detection": ieee_table,
        "sparkov": sparkov_table,
        "synthetic": sparkov_table,
        "ealtman2019/credit-card-transactions": sparkov_table,
    }
    return tables.get(dataset_source, ieee_table)


if __name__ == "__main__":
    # Quick sanity check
    ieee_row = {
        "ProductCD": "W", "TransactionAmt": 102.50, "hour_of_day": 3,
        "day_of_week": 4, "card4": "visa", "card6": "debit",
        "P_emaildomain": "protonmail.com", "R_emaildomain": "gmail.com",
        "p_email_group": "anonymous", "r_email_group": "free",
        "email_match": 0.0, "DeviceType": "mobile", "DeviceInfo": "iOS",
        "addr1": 234.0, "addr2": 87.0, "dist1": 0.0,
        "velocity_1h": 47, "velocity_24h": 51, "amt_zscore": 8.2,
        "card_tenure_days": 12, "fraud_score": 0.93,
    }
    print("IEEE-CIS example:")
    print(serialize_row(ieee_row, "ieee-fraud-detection"))
    print()

    sparkov_row = {
        "merchant": "fraud_Rippin, Kub and Mann", "category": "misc_net",
        "amt": 4.97, "hour_of_day": 22, "day_of_week": 5,
        "geo_distance_km": 512.3, "age_at_txn": 31,
        "velocity_1h": 3, "velocity_24h": 7, "amt_zscore": -0.4,
        "card_tenure_days": 180, "fraud_score": 0.12,
    }
    print("Sparkov example:")
    print(serialize_row(sparkov_row, "sparkov"))
