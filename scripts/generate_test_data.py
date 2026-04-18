#!/usr/bin/env python3
"""
Generate Synthetic Exploration Data for Testing
================================================
Creates fake exploration data (colored placeholder images + poses)
so the full pipeline can be tested offline without Gazebo.

The poses match real object positions from the AWS Small House World.
Images are colored rectangles with room/object labels.

Usage:
  python3 scripts/generate_test_data.py
"""

import json
import math
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ─── Waypoints with room context (same as explore_house.py) ───
WAYPOINTS = [
    # Living Room
    {"x": 1.5, "y": -1.7, "theta": 0.0, "label": "living_room_center", "room": "Living Room", "color": (180, 120, 90)},
    {"x": 0.3, "y": -2.5, "theta": -1.57, "label": "near_sofa", "room": "Living Room", "color": (160, 100, 80)},
    {"x": 1.5, "y": -3.5, "theta": 3.14, "label": "facing_tv_livingroom", "room": "Living Room", "color": (140, 90, 70)},
    {"x": 2.3, "y": -0.8, "theta": 0.0, "label": "near_trash_can", "room": "Living Room", "color": (170, 110, 85)},
    {"x": 0.8, "y": -4.5, "theta": -1.57, "label": "living_room_south", "room": "Living Room", "color": (150, 95, 75)},

    # Kitchen
    {"x": 7.5, "y": -1.0, "theta": 1.57, "label": "near_refrigerator", "room": "Kitchen", "color": (200, 200, 180)},
    {"x": 8.0, "y": -3.0, "theta": 3.14, "label": "kitchen_bench_area", "room": "Kitchen", "color": (210, 210, 190)},
    {"x": 7.0, "y": -4.5, "theta": 0.0, "label": "kitchen_south", "room": "Kitchen", "color": (190, 190, 170)},
    {"x": 6.5, "y": 0.9, "theta": 0.0, "label": "near_kitchen_table", "room": "Kitchen", "color": (195, 195, 175)},
    {"x": 8.5, "y": -2.0, "theta": 1.57, "label": "kitchen_cabinet_area", "room": "Kitchen", "color": (205, 205, 185)},

    # Dining Area
    {"x": 6.0, "y": 1.5, "theta": 1.57, "label": "dining_chairs_south", "room": "Dining", "color": (180, 160, 140)},
    {"x": 7.0, "y": 1.7, "theta": 0.0, "label": "dining_chairs_east", "room": "Dining", "color": (175, 155, 135)},
    {"x": 5.5, "y": 0.0, "theta": -0.785, "label": "dining_hallway", "room": "Dining", "color": (185, 165, 145)},

    # Bedroom
    {"x": -6.0, "y": 1.5, "theta": 1.57, "label": "near_bed", "room": "Bedroom", "color": (100, 130, 170)},
    {"x": -8.0, "y": 2.0, "theta": 3.14, "label": "near_reading_desk", "room": "Bedroom", "color": (90, 120, 160)},
    {"x": -4.5, "y": 2.5, "theta": 1.57, "label": "near_nightstand", "room": "Bedroom", "color": (95, 125, 165)},
    {"x": -3.0, "y": 2.0, "theta": 0.0, "label": "near_wardrobe", "room": "Bedroom", "color": (105, 135, 175)},
    {"x": -6.2, "y": -1.0, "theta": -1.57, "label": "bedroom_tv_area", "room": "Bedroom", "color": (85, 115, 155)},

    # Exercise Area
    {"x": 3.0, "y": 3.0, "theta": 0.0, "label": "near_fitness_equipment", "room": "Exercise", "color": (150, 200, 150)},
    {"x": 2.5, "y": 2.7, "theta": -0.785, "label": "near_dumbbells", "room": "Exercise", "color": (140, 190, 140)},
    {"x": 3.5, "y": 4.0, "theta": 1.57, "label": "exercise_area_north", "room": "Exercise", "color": (145, 195, 145)},

    # Balcony / Patio
    {"x": -0.5, "y": 4.0, "theta": 1.57, "label": "near_balcony_table", "room": "Balcony", "color": (200, 220, 240)},
    {"x": 1.0, "y": 4.0, "theta": 0.0, "label": "balcony_east", "room": "Balcony", "color": (190, 210, 230)},

    # Entrance / Hallway
    {"x": 4.3, "y": -4.5, "theta": -1.57, "label": "near_shoe_rack", "room": "Entrance", "color": (160, 140, 120)},
    {"x": 5.5, "y": -5.0, "theta": 0.0, "label": "entrance_door", "room": "Entrance", "color": (155, 135, 115)},
    {"x": 3.0, "y": -1.0, "theta": 0.0, "label": "central_hallway", "room": "Hallway", "color": (170, 170, 170)},

    # Corridor / Transitions
    {"x": -2.0, "y": 0.0, "theta": 3.14, "label": "bedroom_hallway_junction", "room": "Hallway", "color": (165, 165, 165)},
    {"x": 0.0, "y": 1.0, "theta": 1.57, "label": "central_corridor", "room": "Hallway", "color": (175, 175, 175)},
    {"x": 4.5, "y": 2.0, "theta": 0.0, "label": "exercise_kitchen_link", "room": "Hallway", "color": (160, 160, 160)},
]


def generate_placeholder_image(
    width: int = 640,
    height: int = 480,
    label: str = "",
    room: str = "",
    bg_color: tuple = (128, 128, 128),
) -> Image.Image:
    """Create a colored placeholder image with room/label text."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Add some visual variation (fake furniture outlines)
    # Floor
    draw.rectangle([0, height * 2 // 3, width, height], fill=_darken(bg_color, 30))
    # Wall
    draw.rectangle([0, 0, width, height * 2 // 3], fill=_lighten(bg_color, 20))
    # Ceiling line
    draw.line([(0, height // 6), (width, height // 6)], fill=_darken(bg_color, 50), width=2)

    # Add some "furniture" shapes based on label
    if "sofa" in label:
        draw.rectangle([width // 4, height // 2, 3 * width // 4, 2 * height // 3], fill=(100, 70, 50))
    elif "refrigerator" in label:
        draw.rectangle([width // 3, height // 6, 2 * width // 3, 2 * height // 3], fill=(220, 220, 230))
    elif "bed" in label:
        draw.rectangle([width // 6, height // 3, 5 * width // 6, 2 * height // 3], fill=(200, 180, 160))
    elif "tv" in label:
        draw.rectangle([width // 3, height // 4, 2 * width // 3, height // 2], fill=(20, 20, 30))
    elif "table" in label or "desk" in label:
        draw.rectangle([width // 4, height // 2 - 20, 3 * width // 4, height // 2 + 10], fill=(140, 100, 60))
    elif "fitness" in label or "dumbbell" in label:
        draw.ellipse([width // 3, height // 2 - 30, width // 3 + 60, height // 2 + 30], fill=(80, 80, 80))
    elif "shoe" in label:
        draw.rectangle([width // 3, height // 2, 2 * width // 3, 2 * height // 3], fill=(120, 80, 40))
    elif "wardrobe" in label:
        draw.rectangle([width // 4, height // 5, 3 * width // 4, 2 * height // 3], fill=(130, 90, 50))
    elif "trash" in label:
        draw.rectangle([2 * width // 5, height // 2, 3 * width // 5, 2 * height // 3], fill=(60, 60, 60))
    elif "cabinet" in label:
        draw.rectangle([width // 6, height // 4, 5 * width // 6, 2 * height // 3], fill=(180, 160, 130))

    # Room label at top
    draw.text((10, 10), f"[{room}]", fill="white")
    # Object label at bottom
    clean_label = label.replace("_", " ").title()
    draw.text((10, height - 30), clean_label, fill="white")

    return img


def _darken(color, amount):
    return tuple(max(0, c - amount) for c in color)

def _lighten(color, amount):
    return tuple(min(255, c + amount) for c in color)


def main():
    output_dir = Path(__file__).resolve().parent.parent / "data" / "aws_house_graph"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🏗️  Generating synthetic exploration data...")
    print(f"   Output: {output_dir}")
    print(f"   Waypoints: {len(WAYPOINTS)}")

    poses_data = []

    for idx, wp in enumerate(WAYPOINTS):
        # Generate placeholder image
        img = generate_placeholder_image(
            label=wp["label"],
            room=wp["room"],
            bg_color=wp["color"],
        )
        img_path = output_dir / f"node_{idx:03d}.png"
        img.save(str(img_path))

        # Record pose
        poses_data.append({
            "id": idx,
            "x": wp["x"],
            "y": wp["y"],
            "theta": wp["theta"],
            "label": wp["label"],
            "target_x": wp["x"],
            "target_y": wp["y"],
        })

        print(f"   📸 node_{idx:03d}.png — {wp['room']:10s} | {wp['label']}")

    # Save poses
    poses_file = output_dir / "poses.json"
    with open(poses_file, "w") as f:
        json.dump(poses_data, f, indent=2)

    print(f"\n✅ Generated {len(WAYPOINTS)} synthetic nodes")
    print(f"   Images: {output_dir}/node_*.png")
    print(f"   Poses:  {output_dir}/poses.json")
    print(f"\n💡 Now run the pipeline:")
    print(f'   python3 scripts/run_pipeline.py -i "Go to the kitchen and find the refrigerator"')
    print(f"\n⚠️  Note: These are placeholder images. CLIP scores won't be meaningful.")
    print(f"   For real results, run explore_house.py with Gazebo.")


if __name__ == "__main__":
    main()
