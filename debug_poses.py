import json, numpy as np

with open("data/aws_house_graph/poses.json") as f:
    poses = json.load(f)

print("First 5 poses (from /odom):")
for p in poses[:5]:
    print(f"  Node {p['id']}: ({p['x']:.2f}, {p['y']:.2f}, theta={p['theta']:.2f})")

with open("data/aws_house_graph/chamber_nodes.json") as f:
    chambers = json.load(f)

print("\nFirst 5 chamber_nodes (map frame, from distance transform):")
for c in chambers[:5]:
    print(f"  {c['label']}: ({c['x']:.2f}, {c['y']:.2f}) clearance={c['clearance_m']}m")

print("\nCOORDINATE MISMATCH CHECK:")
print(f"  chamber_nodes[0] (map frame intended): ({chambers[0]['x']:.2f}, {chambers[0]['y']:.2f})")
print(f"  poses[0]         (captured by odom):  ({poses[0]['x']:.2f}, {poses[0]['y']:.2f})")
print(f"  Offset (odom drift): dx={poses[0]['x']-chambers[0]['x']:.2f}, dy={poses[0]['y']-chambers[0]['y']:.2f}")

print("\nAll poses X range: [{:.2f}, {:.2f}]".format(
    min(p['x'] for p in poses), max(p['x'] for p in poses)))
print("All poses Y range: [{:.2f}, {:.2f}]".format(
    min(p['y'] for p in poses), max(p['y'] for p in poses)))
print("All chamber X range: [{:.2f}, {:.2f}]".format(
    min(c['x'] for c in chambers), max(c['x'] for c in chambers)))
print("All chamber Y range: [{:.2f}, {:.2f}]".format(
    min(c['y'] for c in chambers), max(c['y'] for c in chambers)))

# Check if poses keys include depth info
print(f"\nPose keys: {list(poses[0].keys())}")
