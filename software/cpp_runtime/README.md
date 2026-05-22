# C++ Runtime

Real-time embedded inference runtime for the Smart Car Autonomous Platform running on NVIDIA Jetson hardware.

This runtime is responsible for executing the deployed autonomous lane-following pipeline, including camera acquisition, TensorRT-accelerated neural inference, lane boundary extraction, and target point generation for downstream vehicle control.

The runtime was developed as a performance-focused C++ deployment path to reduce latency and improve real-time execution compared to earlier Python implementations.

---

## Features

- Real-time camera and video input support
- TensorRT-accelerated UNet inference
- CUDA-enabled GPU execution on NVIDIA Jetson
- Lane mask post-processing and filtering
- Left/right lane boundary extraction
- Centerline estimation and target point generation
- Runtime visualization overlays
- CAN-compatible control integration support
- Low-latency deployment-oriented architecture

---

## Runtime Pipeline

```text
Camera Input
      ↓
Frame Preprocessing
      ↓
TensorRT UNet Inference
      ↓
Lane Mask Generation
      ↓
Boundary Extraction
      ↓
Centerline / Target Point Calculation
      ↓
Vehicle Control Interface
```

---

## Project Structure

```text
cpp_runtime/
│
├── CMakeLists.txt
└── src/
    ├── main.cpp
    ├── lane_control_main.cpp
    ├── lane_detector.cpp
    ├── lane_detector.hpp
    ├── tensorrt_runner.cpp
    └── tensorrt_runner.hpp
```

---

## Build on Jetson

Install dependencies:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev can-utils
```

Build the runtime:

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

JetPack should already provide:
- CUDA
- TensorRT
- cuDNN

---

## Run

### Camera Runtime

```bash
./lane_runtime --engine ../models/lane_unet_small_256x144_fp16.engine --camera 0
```

### Video Runtime

```bash
./lane_runtime --engine ../models/lane_unet_small_256x144_fp16.engine --video ../data/realtest5.mp4 --save ../cpp_overlay.mp4
```

---

## Notes

This runtime repository contains only the deployed inference pipeline and runtime control path.

Training, dataset generation, and model export tooling are maintained separately from the embedded deployment runtime to keep the Jetson execution environment lightweight and deployment-focused.