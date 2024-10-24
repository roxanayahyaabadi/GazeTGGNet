# TGGNet (Transformer Graph Gaze Neural Network)
This repository introduces a graph-based model for gaze estimation using a Graph Neural Network (GNN) to process facial landmarks and relational edges, ensuring precise eye gaze direction estimation. Key innovations include leveraging GNN's efficiency for real-time applications and integrating Transformer architecture to enhance feature updates.

![](Media/FacialGraph.gif) 


# Landmak Extraction
## GazeCapture dataset
Please use `LandmarkExtraction_GazeCapture.py` file to extract all 478 landmarks from valid images.


# TGGNet
## GazeCapture
Please run `TGGNET_GazeCapture.py` code for training and testing the proposed TGGNet on Gazecapture. The test scale is normal Euclidean. For converting it back to real scale (centimeter), run 
