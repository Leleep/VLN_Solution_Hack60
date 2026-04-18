"""
LM-Nav Gazebo Pipeline
======================
Zero-shot Vision-Language Navigation using:
  - LLM (Ollama llama3) for landmark extraction
  - CLIP (ViT-L/14) for visual grounding
  - Dijkstra DP for optimal walk planning
  - Nav2 for execution in Gazebo

Adapted from the LM-Nav paper (Shah et al., 2022) for indoor use
with the AWS RoboMaker Small House World.
"""

__version__ = "0.1.0"
