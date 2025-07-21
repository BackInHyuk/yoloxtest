#!/usr/bin/env python3
"""
KV260 YOLOX 웹캠 실시간 검출 with 웹 스트리밍
호스트 PC에서 http://board_ip:5000 으로 접속하여 스트림 확인 가능
"""

import cv2
import numpy as np
import time
import threading
import argparse
from flask import Flask, Response, render_template_string
from typing import List, Tuple, Optional
import json

# Vitis AI DPU 런타임 임포트
try:
    import vart
    import xir
except ImportError:
    print("Vitis AI 런타임이 설치되지 않았습니다.")
    exit(1)

class YOLOXDetector:
    def __init__(self, model_path: str, classes_file: str):
        """YOLOX 검출기 초기화"""
        self.model_path = model_path
        self.input_width = 416
        self.input_height = 416
        self.conf_threshold = 0.3
        self.nms_threshold = 0.45
        
        # 클래스 이름 로드
        self.class_names = self._load_classes(classes_file)
        
        # DPU 모델 로드
        self._load_model()
        
        # 색상 팔레트 생성
        self.colors = self._generate_colors(len(self.class_names))
        
        # 통계 정보
        self.fps = 0
        self.inference_time = 0
        self.detection_count = 0
    
    def _load_classes(self, classes_file: str) -> List[str]:
        """클래스 이름 파일 로드"""
        try:
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            print(f"로드된 클래스 수: {len(classes)}")
            return classes
        except FileNotFoundError:
            print(f"클래스 파일을 찾을 수 없습니다: {classes_file}")
            return [f"class_{i}" for i in range(80)]  # COCO 기본 80클래스
    
    def _load_model(self):
        """DPU 모델 로드"""
        try:
            # xmodel 로드
            graph = xir.Graph.deserialize(self.model_path)
            subgraphs = graph.get_root_subgraph().children_topological_sort()
            
            # DPU 서브그래프 찾기
            dpu_subgraph = None
            for subgraph in subgraphs:
                if subgraph.has_attr("device") and subgraph.get_attr("device").upper() == "DPU":
                    dpu_subgraph = subgraph
                    break
            
            if dpu_subgraph is None:
                raise RuntimeError("DPU 서브그래프를 찾을 수 없습니다.")
            
            # DPU 런너 생성
            self.dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
            
            # 입력/출력 텐서 정보 가져오기
            self.input_tensors = self.dpu_runner.get_input_tensors()
            self.output_tensors = self.dpu_runner.get_output_tensors()
            
            print("DPU 모델 로드 완료")
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            exit(1)
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """클래스별 색상 생성"""
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(color[0]), int(color[1]), int(color[2])))
        return colors
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """이미지 전처리"""
        img_h, img_w = image.shape[:2]
        scale = min(self.input_width / img_w, self.input_height / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        # 리사이즈
        resized = cv2.resize(image, (new_w, new_h))
        
        # 패딩 추가
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_x = (self.input_width - new_w) // 2
        pad_y = (self.input_height - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # 정규화 및 차원 변경
        input_data = padded.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))  # HWC -> CHW
        input_data = np.expand_dims(input_data, axis=0)   # 배치 차원 추가
        
        return input_data, scale, pad_x, pad_y
    
    def postprocess(self, outputs: List[np.ndarray], orig_shape: Tuple[int, int], 
                   scale: float, pad_x: int, pad_y: int) -> List[dict]:
        """후처리 및 NMS"""
        predictions = outputs[0][0]
        
        boxes = []
        scores = []
        class_ids = []
        
        for detection in predictions:
            if len(detection) < 5:
                continue
                
            x, y, w, h, objectness = detection[:5]
            class_scores = detection[5:]
            
            if objectness < self.conf_threshold:
                continue
            
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            final_score = objectness * class_score
            
            if final_score < self.conf_threshold:
                continue
            
            # 좌표 변환
            x1 = (x - w/2 - pad_x) / scale
            y1 = (y - h/2 - pad_y) / scale
            x2 = (x + w/2 - pad_x) / scale
            y2 = (y + h/2 - pad_y) / scale
            
            # 경계 확인
            x1 = max(0, min(x1, orig_shape[1]))
            y1 = max(0, min(y1, orig_shape[0]))
            x2 = max(0, min(x2, orig_shape[1]))
            y2 = max(0, min(y2, orig_shape[0]))
            
            boxes.append([x1, y1, x2, y2])
            scores.append(final_score)
            class_ids.append(class_id)
        
        # NMS 적용
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                return [
                    {
                        'bbox': boxes[i],
                        'score': scores[i],
                        'class_id': class_ids[i],
                        'class_name': self.class_names[class_ids[i]] if class_ids[i] < len(self.class_names) else f"class_{class_ids[i]}"
                    }
                    for i in indices
                ]
        
        return []
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """객체 검출 수행"""
        start_time = time.time()
        
        orig_shape = image.shape[:2]
        
        # 전처리
        input_data, scale, pad_x, pad_y = self.preprocess(image)
        
        # DPU 추론
        job_id = self.dpu_runner.execute_async([input_data], [])
        outputs = self.dpu_runner.wait(job_id)
        
        # 후처리
        detections = self.postprocess(outputs, orig_shape, scale, pad_x, pad_y)
        
        # 통계 업데이트
        self.inference_time = time.time() - start_time
        self.detection_count = len(detections)
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """검출 결과를 이미지에 그리기"""
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            score = detection['score']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            x1, y1, x2, y2 = map(int, bbox)
            color = self.colors[class_id % len(self.colors)]
            
            # 바운딩 박스
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # 라벨
            label = f"{class_name}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image

class WebStreamer:
    def __init__(self, detector: YOLOXDetector, camera_id: int = 0):
        self.detector = detector
        self.camera_id = camera_id
        self.cap = None
        self.frame = None
        self.result_frame = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Flask 앱 초기화
        self.app = Flask(__name__)
        self.setup_routes()
        
        # FPS 계산용
        self.frame_count = 0
        self.start_time = time.time()
    
    def setup_routes(self):
        """Flask 라우트 설정"""
        
        @self.app.route('/')
        def index():
            """메인 페이지"""
            return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>KV260 YOLOX Real-time Detection</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f0f0f0;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .video-container { 
            text-align: center; 
            margin: 20px 0;
        }
        .stats { 
            display: flex; 
            justify-content: space-around; 
            margin: 20px 0;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }
        .stat-item { 
            text-align: center;
        }
        .stat-value { 
            font-size: 24px; 
            font-weight: bold; 
            color: #007bff;
        }
        .stat-label { 
            font-size: 14px; 
            color: #666;
        }
        img { 
            max-width: 100%; 
            border: 2px solid #ddd;
            border-radius: 8px;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 style="text-align: center; color: #333;">🚀 KV260 YOLOX Real-time Detection</h1>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="fps">--</div>
                <div class="stat-label">FPS</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="inference-time">--</div>
                <div class="stat-label">Inference (ms)</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="detections">--</div>
                <div class="stat-label">Objects</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="status">🟢</div>
                <div class="stat-label">Status</div>
            </div>
        </div>
        
        <div class="video-container">
            <img src="/video_feed" alt="YOLOX Detection Stream" id="video-stream">
        </div>
        
        <div class="controls">
            <button onclick="toggleStream()">Toggle Stream</button>
            <button onclick="downloadSnapshot()">📸 Snapshot</button>
            <button onclick="refreshStats()">🔄 Refresh</button>
        </div>
    </div>

    <script>
        // 통계 업데이트
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('inference-time').textContent = (data.inference_time * 1000).toFixed(1);
                    document.getElementById('detections').textContent = data.detections;
                    document.getElementById('status').textContent = data.running ? '🟢' : '🔴';
                })
                .catch(error => console.error('Error:', error));
        }
        
        function toggleStream() {
            fetch('/toggle', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    console.log('Stream toggled:', data.running);
                    updateStats();
                });
        }
        
        function downloadSnapshot() {
            window.open('/snapshot', '_blank');
        }
        
        function refreshStats() {
            updateStats();
        }
        
        // 주기적으로 통계 업데이트
        setInterval(updateStats, 1000);
        updateStats(); // 초기 로드
    </script>
</body>
</html>
            ''')
        
        @self.app.route('/video_feed')
        def video_feed():
            """비디오 스트림 제공"""
            return Response(self.generate_frames(),
                           mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/stats')
        def stats():
            """통계 정보 API"""
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            return {
                'fps': fps,
                'inference_time': self.detector.inference_time,
                'detections': self.detector.detection_count,
                'running': self.is_running,
                'frame_count': self.frame_count
            }
        
        @self.app.route('/toggle', methods=['POST'])
        def toggle():
            """스트림 토글"""
            if self.is_running:
                self.stop_camera()
            else:
                self.start_camera()
            return {'running': self.is_running}
        
        @self.app.route('/snapshot')
        def snapshot():
            """현재 프레임 스냅샷"""
            with self.lock:
                if self.result_frame is not None:
                    _, buffer = cv2.imencode('.jpg', self.result_frame)
                    return Response(buffer.tobytes(), mimetype='image/jpeg')
            return "No frame available", 404
    
    def start_camera(self):
        """카메라 시작"""
        if self.is_running:
            return
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"카메라 {self.camera_id}를 열 수 없습니다.")
            return
        
        # 카메라 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        self.frame_count = 0
        self.start_time = time.time()
        
        print("카메라 시작됨")
    
    def stop_camera(self):
        """카메라 중지"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        print("카메라 중지됨")
    
    def generate_frames(self):
        """프레임 생성기"""
        while True:
            with self.lock:
                if self.result_frame is not None:
                    _, buffer = cv2.imencode('.jpg', self.result_frame, 
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # 기본 이미지 생성
                    default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(default_frame, "Camera Not Started", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.jpg', default_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    def process_frames(self):
        """프레임 처리 스레드"""
        while True:
            if self.is_running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # 검출 수행
                    detections = self.detector.detect(frame)
                    
                    # 결과 그리기
                    result_frame = self.detector.draw_detections(frame, detections)
                    
                    # 정보 오버레이
                    current_time = time.time()
                    elapsed_time = current_time - self.start_time
                    fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    cv2.putText(result_frame, f"FPS: {fps:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Inference: {self.detector.inference_time*1000:.1f}ms", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Objects: {len(detections)}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    with self.lock:
                        self.frame = frame
                        self.result_frame = result_frame
                    
                    self.frame_count += 1
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
    
    def run(self, host='0.0.0.0', port=5000):
        """웹 서버 실행"""
        # 프레임 처리 스레드 시작
        processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        processing_thread.start()
        
        # 카메라 자동 시작
        self.start_camera()
        
        print(f"웹 서버 시작: http://{host}:{port}")
        print("Ctrl+C로 종료")
        
        try:
            self.app.run(host=host, port=port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\n서버 종료 중...")
        finally:
            self.stop_camera()

def main():
    parser = argparse.ArgumentParser(description='KV260 YOLOX 웹 스트리밍 검출기')
    parser.add_argument('--model', default='yolox_nano_pt.xmodel', help='YOLOX 모델 파일')
    parser.add_argument('--classes', default='coco2017_classes.txt', help='클래스 파일')
    parser.add_argument('--camera', type=int, default=0, help='카메라 장치 번호')
    parser.add_argument('--host', default='0.0.0.0', help='웹 서버 호스트')
    parser.add_argument('--port', type=int, default=5000, help='웹 서버 포트')
    
    args = parser.parse_args()
    
    # YOLOX 검출기 초기화
    print("YOLOX 검출기 초기화 중...")
    detector = YOLOXDetector(args.model, args.classes)
    
    # 웹 스트리머 초기화
    streamer = WebStreamer(detector, args.camera)
    
    # 웹 서버 실행
    streamer.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
