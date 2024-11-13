# TGGNet (Transformer Graph Gaze Neural Network)
This repository introduces a graph-based model for gaze estimation using a Graph Neural Network (GNN) to process facial landmarks and relational edges, ensuring precise eye gaze direction estimation. Key innovations include leveraging GNN's efficiency for real-time applications and integrating Transformer architecture to enhance feature updates.

![](Media/FacialGraph.gif) 


# Landmak Extraction
## GazeCapture dataset
Please use `LandmarkExtraction_GazeCapture.py` file to extract all 478 landmarks from valid images.


# TGGNet
## GazeCapture
Please run `TGGNET_GazeCapture.py` code for training and testing the proposed TGGNet on Gazecapture. The test scale is normal Euclidean. For converting it back to real scale (centimeter), run `RealScale_GazeCapture.py`.

# Demo
Change the `input_video_path` to match the input video's path.

Change the `model_path` to match the pre-trained model's path (`trained_model_No_Or.pt`). 

Change the `output_video_path` to match the output video's path. 

Then run `API.py` code to have the output video visualizing gaze vectors, POG coordinates, FPS, and execution time.

