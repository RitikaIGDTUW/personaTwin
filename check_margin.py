import json

with open("data/processed/sensitivity/studentlife_sensitivity_aggregates.json") as f:
    agg = json.load(f)

print("=== PERSONALIZED ===")
for direction, stats in agg.items():
    print(f"{direction:10s} margin_std={stats['margin_std']:.6f}  margin_mean={stats['margin_mean']:.6f}  n_participants={stats['participant_count']}")

with open("data/processed/sensitivity/studentlife_POPULATION_sensitivity_aggregates.json") as f:
    pop_agg = json.load(f)

print()
print("=== POPULATION (for comparison) ===")
for direction, stats in pop_agg.items():
    print(f"{direction:10s} margin_std={stats['margin_std']:.6f}  margin_mean={stats['margin_mean']:.6f}  n_participants={stats['participant_count']}")
