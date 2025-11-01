import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount"""
    if currency == "USD":
        return f"${amount:,.2f}"
    elif currency == "EUR":
        return f"€{amount:,.2f}"
    elif currency == "GBP":
        return f"£{amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change"""
    if old_value == 0:
        return 0
    return ((new_value - old_value) / old_value) * 100

def safe_divide(numerator: float, denominator: float) -> float:
    """Safe division with zero handling"""
    if denominator == 0:
        return 0.0
    return numerator / denominator

def export_to_json(data: Dict[str, Any], filepath: str) -> None:
    """Export data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)