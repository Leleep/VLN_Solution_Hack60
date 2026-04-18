# LM-Nav: Architecture & Pipeline Explanation

This document provides a detailed breakdown of the Vision-Language Navigation (VLN) pipeline used in this repository. The pipeline allows a simulated TurtleBot3 to navigate through a house based on natural language instructions (like *"Go left past the couch and stop near the refrigerator"*).

The system is completely decoupled into **four distinct phases**. Instead of the AI making split-second driving decisions live in pure 3D space, it first maps the house into a discrete "topological graph" (a scatter patch of physical points tied to images), and then runs an offline pipeline to score those points and map a path.

---

## High-Level Diagram

```text
Phase 1: Setup → Phase 2: Explore → Phase 3: AI Pipeline → Phase 4: Execution
(Gazebo/ROS2)    (Map the house)    (Process Language)   (Drive the path)
```

---

## Phase 1: Environment Setup

Before the pipeline can do anything, the physics simulation and robot ROS modules must be running.

### \`launch_sim.sh\`
This is the master boot script.
- It configures heavy-duty Linux environment variables (ensuring Gazebo renders on the NVIDIA GPU instead of integrated graphics via PRIME offloading).
- It points ROS2 to the `aws-robomaker-small-house-world` files.
- It launches Gazebo, RViz, and the Nav2 (Navigation2) software stack.
- It spawns our custom `waffle_stable.model` (the 20kg anti-topple robot). 
- Crucially, it dynamically tunes the **Nav2 Costmap** layer (shrinking the `inflation_radius` to 0.15m and increasing the `cost_scaling_factor` to 15.0), effectively shrinking the robot's collision-avoidance boundary so it can squeeze through standard home doorways.

---

## Phase 2: Graph Building (Exploration)

The AI cannot navigate an environment it doesn't know. Before accepting language commands, it must map the physical space to visual sights.

### \`scripts/explore_house.py\`
This is the "house explorer."
- It contains a hardcoded list of 55 predetermined physical `[X, Y, Theta]` coordinates spanning the entire house.
- It commands the Nav2 Action Server to drive the robot to each coordinate sequentially.
- At each stop, it queries the robot's odometry for its exact physical location in the map.
- It snaps a picture from the robot's first-person RGB depth camera.
- It saves these pictures as `node_000.png` through `node_054.png` in the `data/aws_house_graph/` directory.
- Finally, it writes `poses.json` linking the image IDs to their actual physical coordinates.

**Result:** This collection of images tied to physical coordinate nodes forms our complete Topological Graph.

---

## Phase 3: The AI Pipeline (Decision Making)

Once the graph exists, you can give the robot language commands. This phase is purely computational and **offline**—the robot sits completely still in the simulation while this logic happens. 

This phase is orchestrated by running `python scripts/run_pipeline.py -i "<your natural language prompt>"`.

### \`scripts/run_pipeline.py\`
This orchestrator executes four sequential sub-modules located in the `lmnav/` folder:

1. **\`lmnav/llm_extractor.py\` (Processing Language)**
   * **Action:** It parses your full instruction and extracts the core entity/landmark (e.g., extracting `"refrigerator"` from the prompt `"go find the refrigerator"`).
   * **Backend:** Depending on your `pipeline_config.yaml`, it uses either spaCy, a local Ollama LLM execution, or the OpenAI API to do this.

2. **\`lmnav/clip_scorer.py\` (Vision-Language Matching)**
   * **Action:** It loads an OpenCLIP model (e.g., `ViT-L-14`). It encodes the extracted text (`"refrigerator"`) into a mathematical vector. Then, it loads all 55 `node_X.png` images we captured previously, converts them into vectors, and calculates the **cosine similarity** between the text vector and the image vectors.
   * **Result:** It ranks the nodes to find the image that statistically "looks the most like a refrigerator." 

3. **\`lmnav/graph_search.py\` (Path Planning)**
   * **Action:** Now that CLIP has identified *which* node represents the target, this script opens `poses.json` to find the physical `[X, Y, Theta]` coordinates of that node. 
   * **Result:** It calculates the shortest topological distance (using Dijkstra's algorithm logic) from the robot's current starting position to the target node.

4. **\`lmnav/visualizer.py\` (Output & Reporting)**
   * **Action:** It saves the calculated route as an ordered list of coordinate steps to sequence to `output/planned_walk.json`. It also draws a top-down diagram (`walk_visualization.png`) mapping the nodes so you can visually verify the AI's intended route.

---

## Phase 4: Execution

Now that the AI has generated a planned route, it is time to command the physical robot to move through the simulation.

### \`scripts/execute_walk.py\`
This is the driver script.
- It reads the `output/planned_walk.json` trajectory path generated in Phase 3.
- It connects to the Nav2 Action Server running in your Gazebo terminal.
- It feeds the coordinates to the robot sequentially. It acts as a supervisor, telling the robot to drive to coordinate A, waiting until the robot succeeds or fails via Nav2 Status reporting, and then commanding it to drive to coordinate B, progressing node-by-node until it reaches the final target.

### \`scripts/ego_view.py\` *(Optional Diagnostic)*
This is a standalone diagnostic terminal tool. You can run it while the robot is driving via `execute_walk.py`. It subscribes directly to the robot's ROS `/intel_realsense_r200_depth/image_raw` topic and prints the incoming ROS Image frames to an OpenCV popup window. This provides a live "dashcam" feed of the action as the robot executes the instruction.
