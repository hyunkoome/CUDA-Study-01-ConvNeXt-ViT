# ConvNeXt-ViT CUDA 실습 프로젝트

> ✅ 30일 완성! CUDA + ONNX + TensorRT + Plugin + WebGPU 포팅

## 📅 커리큘럼 요약
- Day 1~7: ConvNeXt-ViT 구조 + ONNX Export
- Day 8~14: Graph 최적화 + 분석
- Day 15~21: CUDA 커널 및 Plugin 개발
- Day 22~28: TensorRT 추론 및 삽입
- Day 29~30: WebGPU 포팅 및 보고서 정리

## 📁 주요 폴더
- `models/` – ConvNeXt-ViT PyTorch 모델
- `cuda_tasks/` – Daily CUDA Task (벡터 연산부터 attention까지)
- `trt_plugin/` – TensorRT Plugin 코드
- `webgpu_porting/` – 포팅된 WGSL 커널 및 실행 샘플

## ✅ 환경 구축
```bash
conda env create -f environment.yml
conda activate convnext-cuda
```

시작은 `cuda_tasks/day01_vec_add/` 부터!
