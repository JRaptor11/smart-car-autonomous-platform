# C++ Embedded Runtime

Real-time embedded inference runtime for autonomous lane following deployed on NVIDIA Jetson hardware.

This runtime is responsible for:
- image acquisition
- TensorRT inference execution
- lane mask generation
- boundary extraction
- centerline estimation
- downstream vehicle control interfacing

The runtime was developed as the deployed onboard autonomy pipeline for the autonomous vehicle platform.

---

## Runtime Responsibilities

The embedded runtime performs the full onboard perception pipeline in real time directly on Jetson embedded hardware.

### Core Functions
- Camera/video frame acquisition
- Image preprocessing
- TensorRT UNet inference execution
- Lane mask postprocessing
- Left/right lane boundary extraction
- Centerline generation
- Target point estimation
- Visualization overlay generation
- Vehicle control interfacing

---

## Runtime Architecture

```text
Camera Input
      ↓
Frame Capture
      ↓
Image Preprocessing
      ↓
TensorRT UNet Inference
      ↓
Lane Mask Generation
      ↓
Boundary Extraction
      ↓
Centerline Estimation
      ↓
Target Point Generation
      ↓
Vehicle Control Interface
```

---

## Why the Runtime Was Rewritten in C++

Early perception pipeline development and model validation were initially implemented in Python during rapid prototyping stages.

As system integration complexity increased, the runtime was rewritten into a lower-latency C++ deployment architecture to improve:
- inference throughput
- runtime stability
- processing consistency
- embedded deployment performance
- real-time control responsiveness

The final runtime leveraged:
- TensorRT acceleration
- CUDA GPU execution
- OpenCV image processing
- streamlined inference execution paths

to support real-time onboard autonomous operation.

---

## Runtime Components

### Frame Acquisition
Handles:
- camera input
- video input
- frame buffering
- image acquisition timing

### Inference Engine
Responsible for:
- TensorRT engine loading
- GPU memory management
- CUDA inference execution
- model input/output handling

### Postprocessing
Processes:
- segmentation masks
- lane boundary extraction
- centerline generation
- target point estimation

### Visualization
Generates:
- runtime overlays
- lane visualization
- centerline rendering
- debugging visualization output

### Control Interface
Provides:
- downstream target point output
- controller integration hooks
- vehicle control interfacing

---

## Build Requirements

### Platform
- NVIDIA Jetson
- JetPack SDK
- CUDA
- TensorRT
- cuDNN

### Dependencies
- OpenCV
- CMake
- build-essential

---

## Build Instructions

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev can-utils

mkdir -p build
cd build

cmake ../cpp_runtime
make -j$(nproc)
```

JetPack should already provide:
- CUDA
- TensorRT
- cuDNN

---

## Runtime Execution

### Camera Input

```bash
./lane_runtime \
  --engine ../lane_unet_small_256x144_fp16.engine \
  --camera 0
```

### Video Input

```bash
./lane_runtime \
  --engine ../lane_unet_small_256x144_fp16.engine \
  --video ../data/realtest5.mp4 \
  --save ../cpp_overlay.mp4
```

---

## Runtime Overlay Example

![Runtime Overlay](../../media/images/lane_detection_runtime_overlay.png)

Real-time runtime visualization overlay showing lane segmentation, boundary extraction, and target point generation.

---

## Embedded Deployment Focus

The runtime was specifically optimized for:
- low-latency embedded inference
- real-time autonomous operation
- constrained onboard compute environments
- GPU-accelerated execution
- embedded robotics deployment

The deployment architecture prioritized:
- stable runtime execution
- lightweight inference paths
- efficient GPU utilization
- deterministic processing behavior

---

## Related Repository Sections

### System-Level Documentation
- `../../README.md`

### Architecture Diagrams
- `../../docs/architecture/`

### Hardware Integration
- `../../hardware/`

### Demonstration Media
- `../../media/`