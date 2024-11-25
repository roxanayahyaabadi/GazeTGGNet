import os

from flask import Flask, request, jsonify, render_template, send_from_directory, send_file

import requests

import cv2

import torch

import numpy as np

import networkx as nx

import mediapipe as mp

from torch_geometric.data import Data

from moviepy.editor import ImageSequenceClip

from torch_geometric.nn import TransformerConv, GlobalAttention

import torch.nn.functional as F

import time

# Define Constants

UPLOAD_PATH = 'uploads'

OUTPUT_PATH = 'outputs'

MAX_FILE_SIZE = 16 * 1024 * 1024  # Limit upload size to 16MB

MODEL_PATH = 'models/trained_model_No_Or.pt'

# Initialize Flask app

app = Flask(__name__, static_folder='static', template_folder='templates')

app.config['UPLOAD_FOLDER'] = UPLOAD_PATH

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure directories exist

if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


# Transformer model class

class TransformerNet(torch.nn.Module):

    def __init__(self, num_node_features):
        super(TransformerNet, self).__init__()

        head_dim1 = 64

        head_dim2 = 32

        head_dim3 = 16

        head_dim4 = 8

        self.conv1 = TransformerConv(num_node_features, head_dim1 * 8)

        self.conv2 = TransformerConv(head_dim1 * 8, head_dim2 * 8)

        self.conv3 = TransformerConv(head_dim2 * 8, head_dim3 * 4)

        self.conv4 = TransformerConv(head_dim3 * 4, head_dim4 * 4)

        self.att_pool = GlobalAttention(gate_nn=torch.nn.Linear(head_dim4 * 4, 1))

        self.fc = torch.nn.Linear(head_dim4 * 4, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.elu(self.conv1(x, edge_index))

        x = F.elu(self.conv2(x, edge_index))

        x = F.elu(self.conv3(x, edge_index))

        x = F.elu(self.conv4(x, edge_index))

        x = self.att_pool(x, data.batch)

        x = self.fc(x)

        return x


# Setup model and mediapipe tools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerNet(num_node_features=3).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


# Function to process images and extract landmarks using MediaPipe

def process_image(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    return [(lm.x, lm.y, lm.z) for lm in landmarks]


# Function to build graph from landmarks

def build_graph_from_landmarks(landmarks, edges):
    G = nx.Graph()

    for i, pos in enumerate(landmarks):
        G.add_node(i, pos=pos)

    G.add_edges_from(edges)

    x = torch.tensor([G.nodes[i]['pos'] for i in G.nodes()], dtype=torch.float)

    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = Data(x=x, edge_index=edge_index)

    return data


# Define edges for the facial landmarks graph

edges = [(468, node) for node in range(478) if node != 468]

edges += [(473, node) for node in range(478) if node != 473]

edges.extend([(471, 159), (159, 469), (469, 145), (145, 471),

              (476, 475), (475, 474), (474, 477), (477, 476),

              (1, 33), (1, 173), (1, 162), (1, 263), (1, 398), (1, 368),

              (33, 246), (146, 161), (161, 160), (160, 150), (150, 158), (158, 157),

              (157, 173), (173, 155), (155, 154), (154, 153), (153, 145), (145, 144),

              (144, 163), (163, 7), (7, 33),

              (398, 384), (384, 385), (385, 386), (386, 387), (387, 388),

              (388, 263), (263, 249), (249, 390), (390, 373), (373, 374),

              (374, 380), (380, 381), (381, 382), (382, 398)])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:

        filepath = os.path.join(UPLOAD_PATH, file.filename)

        file.save(filepath)

        # Process the video using the API functionality

        cap = cv2.VideoCapture(filepath)

        if not cap.isOpened():
            return jsonify({'error': 'Error: Could not open video.'}), 500

        # Get video properties

        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []

        landmarks_list = []

        gaze_vectors = []

        execution_times = []

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)

            start_time = time.time()

            landmarks = process_image(frame)

            if landmarks is None:
                continue

            landmarks_list.append(landmarks)

            data = build_graph_from_landmarks(landmarks, edges).to(device)

            with torch.no_grad():

                output = model(data)

                gaze_vectors.append(output.cpu().numpy())

            end_time = time.time()

            execution_times.append(end_time - start_time)

        cap.release()

        # Calculate average execution time

        average_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0

        # Visualize the gaze vectors on the frames

        output_frames = []

        for i, frame in enumerate(frames):

            if i >= len(gaze_vectors):
                break

            landmarks = landmarks_list[i]

            gaze_vector = gaze_vectors[i]

            gaze_vector_3d = np.array([-gaze_vector[0][0], gaze_vector[0][1], 1.0])

            start_point_468 = (int(landmarks[468][0] * frame.shape[1]), int(landmarks[468][1] * frame.shape[0]))

            end_point_468 = (start_point_468[0] + int(gaze_vector_3d[0] * 100),

                             start_point_468[1] - int(gaze_vector_3d[1] * 100))

            start_point_473 = (int(landmarks[473][0] * frame.shape[1]), int(landmarks[473][1] * frame.shape[0]))

            end_point_473 = (start_point_473[0] + int(gaze_vector_3d[0] * 100),

                             start_point_473[1] - int(gaze_vector_3d[1] * 100))

            cv2.arrowedLine(frame, start_point_468, end_point_468, (0, 0, 255), 4, tipLength=0.2)

            cv2.arrowedLine(frame, start_point_473, end_point_473, (0, 0, 255), 4, tipLength=0.2)

            output_frames.append(frame)

        # Save the output video

        output_video_path = os.path.join(OUTPUT_PATH, file.filename)

        clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in output_frames], fps=fps)

        clip.write_videofile(output_video_path, codec='libx264')

        # Clean up the files

        os.remove(filepath)

        return jsonify({

            'video_url': f'/outputs/{file.filename}',

            'average_execution_time': average_execution_time

        })


@app.route('/outputs/<filename>')
def get_output_file(filename):
    return send_from_directory(OUTPUT_PATH, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)