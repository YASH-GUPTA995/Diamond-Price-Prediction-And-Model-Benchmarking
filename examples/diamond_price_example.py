"""
Example script demonstrating how to use the price_predictor module
to predict diamond prices using the trained XGBoost model.
"""

import pandas as pd
from price_predictor import train_price_predictor, predict_price, predict_batch

# Step 1: Load the data
print("=" * 70)
print("DIAMOND PRICE PREDICTION SYSTEM")
print("=" * 70)

df = pd.read_csv('diamonds.csv')
print(f"\nLoaded diamonds.csv with shape: {df.shape}")

# Step 2: Train the model
print("\nTraining the price prediction model...")
model = train_price_predictor(df)

# Step 3: Predict prices for single diamonds
print("\n" + "=" * 70)
print("SINGLE DIAMOND PRICE PREDICTIONS")
print("=" * 70)

# Example 1: Budget Diamond
budget_diamond = {
    'carat': 0.3,
    'cut': 'Good',
    'color': 'J',
    'clarity': 'I1',
    'depth': 63.0,
    'table': 55.0,
    'x': 4.31,
    'y': 4.26,
    'z': 2.71
}

price1 = predict_price(budget_diamond)
print(f"\nBudget Diamond:")
print(f"  Specs: {budget_diamond['carat']}ct, {budget_diamond['cut']}, {budget_diamond['color']}, {budget_diamond['clarity']}")
print(f"  Predicted Price: ${price1:,.2f}")

# Example 2: Mid-Range Diamond
mid_range_diamond = {
    'carat': 0.75,
    'cut': 'Ideal',
    'color': 'E',
    'clarity': 'VVS1',
    'depth': 62.5,
    'table': 56.5,
    'x': 5.73,
    'y': 5.77,
    'z': 3.60
}

price2 = predict_price(mid_range_diamond)
print(f"\nMid-Range Diamond:")
print(f"  Specs: {mid_range_diamond['carat']}ct, {mid_range_diamond['cut']}, {mid_range_diamond['color']}, {mid_range_diamond['clarity']}")
print(f"  Predicted Price: ${price2:,.2f}")

# Example 3: Premium Diamond
premium_diamond = {
    'carat': 1.5,
    'cut': 'Premium',
    'color': 'D',
    'clarity': 'IF',
    'depth': 61.0,
    'table': 57.0,
    'x': 7.42,
    'y': 7.47,
    'z': 4.56
}

price3 = predict_price(premium_diamond)
print(f"\nPremium Diamond:")
print(f"  Specs: {premium_diamond['carat']}ct, {premium_diamond['cut']}, {premium_diamond['color']}, {premium_diamond['clarity']}")
print(f"  Predicted Price: ${price3:,.2f}")

# Step 4: Predict prices for multiple diamonds at once
print("\n" + "=" * 70)
print("BATCH PRICE PREDICTIONS")
print("=" * 70)

diamonds = [
    {'carat': 0.5, 'cut': 'Very Good', 'color': 'H', 'clarity': 'VS1', 'depth': 62.0, 'table': 56.0, 'x': 5.12, 'y': 5.15, 'z': 3.17},
    {'carat': 0.8, 'cut': 'Excellent', 'color': 'G', 'clarity': 'SI1', 'depth': 61.5, 'table': 55.5, 'x': 6.03, 'y': 6.07, 'z': 3.72},
    {'carat': 1.2, 'cut': 'Very Good', 'color': 'F', 'clarity': 'VS2', 'depth': 63.0, 'table': 57.0, 'x': 6.92, 'y': 6.96, 'z': 4.38},
]

batch_prices = predict_batch(diamonds)

print(f"\nPredicting prices for {len(diamonds)} diamonds:")
for i, (diamond, price) in enumerate(zip(diamonds, batch_prices), 1):
    print(f"  Diamond {i}: {diamond['carat']}ct ({diamond['cut']}) → ${price:,.2f}")

print("\n" + "=" * 70)
print("DIAMOND PRICE PREDICTION COMPLETE!")
print("=" * 70)
