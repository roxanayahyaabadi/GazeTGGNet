# TGGNet (Transformer Graph Gaze Neural Network)
This repository introduces a graph-based model for gaze estimation using a Graph Neural Network (GNN) to process facial landmarks and relational edges, ensuring precise eye gaze direction estimation. Key innovations include leveraging GNN's efficiency for real-time applications and integrating Transformer architecture to enhance feature updates.

![](Media/FacialGraph.gif) 


# Landmak Extraction
## GazeCapture dataset
Please use `LandmarkExtraction_GazeCapture.py` file to extract all 478 landmarks from valid images.


# TGGNet
## GazeCapture
Please run `TGGNET_GazeCapture.py` code for training and testing the proposed TGGNet on Gazecapture. The test scale is normal Euclidean. To convert it back to real scale (centimeter), run `RealScale_GazeCapture.py`.

# Demo
## API web-based demo:

Please go to `Demo/api/api.py` and change the path of the input video.

Then run it. You will see the web page open and be prompted to upload your test input video. Since the model is trained on the GazeCapture dataset, recording your video using an iPhone in portrait mode is recommended for better performance.

## API only code-based demo:

Change the `input_video_path` to match the input video's path.

Change the `model_path` to match the pre-trained model's path (`trained_model_No_Or.pt`). 

Change the `output_video_path` to match the output video's path. 

Then run `API.py` code to have the output video visualizing gaze vectors, POG coordinates, FPS, and execution time.



