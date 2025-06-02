# YOLOv12 + FCAFormer 기반 CUDA 최적화

> ✅ 30일 완성! CUDA + ONNX + TensorRT + Plugin + WebGPU 포팅

# 📅 커리큘럼 요약: YOLOv12 + FCAFormer 기반 CUDA + ONNX + TensorRT + WebGPU 실습

| 기간         | 주제                           | 세부 내용 요약 |
|--------------|--------------------------------|----------------|
| **Day 1~7**   | ✅ ConvNeXt-ViT 구조 분석 및 ONNX Export 준비 | 모델 구조 이해, PyTorch → ONNX 변환, dynamic/static axis 확인 및 export 테스트 |
| **Day 8~14**  | 🔍 ONNX Graph 최적화 및 구조 분석 | `onnxsim`, `onnxruntime`, `onnx-graphsurgeon` 등으로 불필요 노드 제거 및 reshape, transpose 최적화 |
| **Day 15~21** | ⚙️ CUDA 커널 함수 및 TensorRT Plugin 개발 | 주요 연산 (Conv, LayerNorm 등) 직접 CUDA로 구현 및 Plugin 템플릿 작성, 테스트 Runner 연동 |
| **Day 22~28** | ⚡ TensorRT 추론 통합 및 최적화 | Engine 빌드, INT8/FP16 지원, Custom Plugin 삽입 및 Profile 분석, Performance Benchmark 측정 |
| **Day 29~30** | 🧪 WebGPU 포팅 및 보고서 작성 | `cuda2wgsl` 또는 수동 변환으로 WebGPU 대응, 결과 비교 보고서 정리 및 코드 리팩토링 |


## 📁 주요 폴더
- `models/` – PyTorch 모델
- `cuda_tasks/` – Daily CUDA Task (벡터 연산부터 attention까지)
- `trt_plugin/` – TensorRT Plugin 코드
- `webgpu_porting/` – 포팅된 WGSL 커널 및 실행 샘플

# 📅 YOLOv12 + FCAFormer + CUDA 최적화 및 WebGPU 대응 40일 커리큘럼

| Day | Main Task (YOLOv12 / FCAFormer) | Daily CUDA Task | Description | 보고서 |
| --- | --- | --- | --- | --- |
| 1 | YOLOv12 구조 개요 파악 (Detect 구조 포함) | vec_add_kernel.cu, dot_product_kernel.cu | 기본 벡터 연산 커널 이해 | [lab02_dot_product](https://www.notion.so/lab02_dot_product-206c605c59a0805da8e2fc7c2d1af8fa?pvs=21) |
| 2 | C2f 구조 분석 및 forward 시각화 | matrix_add_kernel.cu, relu_kernel.cu | 기본 연산 + 활성화 함수 구현 | |
| 3 | Detect 모듈: bias_init / reg_max 분석 | softmax_kernel.cu, argmax_kernel.cu | softmax 및 class 예측용 argmax | |
| 4 | make_anchors(), strides 계산 흐름 분석 | layernorm_meanvar.cu, broadcast_add_kernel.cu | 정규화 mean/var 추출 및 브로드캐스트 연산 | |
| 5 | forward() vs _inference() 경로 차이 분석 | elementwise_mul.cu, layernorm_forward.cu | 곱셈 및 LayerNorm 전방 구현 | |
| 6 | FCA Block 구조 정리 및 시각화 | matrix_transpose.cu, softmax_2d_kernel.cu | 행렬 전치 + attention용 softmax2d | |
| 7 | Fourier Positional Encoding 구조 확인 | warp_shuffle_max.cu, prefix_sum_kernel.cu | max 연산 및 prefix sum 커널 실습 | |
| 8 | FCA vs SE vs ECA 비교 실험 준비 | matmul_sharedmem.cu, global_avg_pool.cu | matmul 공유메모리 + GAP 구현 | |
| 9 | YOLOv12 Loss 구성 요소 정리 (box, cls, obj) | gelu_kernel.cu, einsum_qk_attn.cu | GELU 및 einsum 기반 QK 연산 | |
| 10 | FCA block 최소 YOLOv12에 삽입 실습 | topk_kernel.cu | top-k 연산 기반 score 선택 | |
| 11 | PE 구조 실험 (YOLOv12 backbone에 추가) | conv2d_basic.cu | 기본 Conv2D 구현 | |
| 12 | C2f + FCA 조합 성능 비교 실험 | conv2d_shared_tile.cu | 타일 기반 conv 최적화 | |
| 13 | YOLOv12 Neck 구조 분석 | batch_norm_forward.cu | 배치 정규화 구현 | |
| 14 | head 구조 재확인 및 One2Many vs One2One 비교 | reduction_sum_blocks.cu | 블록 단위 합 연산 구현 | |
| 15 | FCA block별 FLOPs 측정 및 profiling | instance_norm_forward.cu | InstanceNorm forward 연산 | |
| 16 | 실험 결과 기반 FCA 채널 수 튜닝 | group_norm_forward.cu | GroupNorm forward 연산 | |
| 17 | Loss weight 조정 및 성능 향상 실험 | layernorm_backward.cu | LayerNorm backward 연산 | |
| 18 | export=True, ONNX 변환 흐름 파악 | matmul_tile_opt.cu | 최적화된 matmul 타일 연산 | |
| 19 | sigmoid 후 후처리 postprocess 분석 | attention_softmax_fused.cu | fused attention softmax 구현 | |
| 20 | postprocess() 커스터마이징 실습 | topk_kernel.cu 재복습 | TopK 후 box/score 정렬 실습 | |
| 21 | FCA 삽입 구조 ONNX 추론 확인 | conv2d_shared_tile.cu 재실습 | ONNX-friendly 연산 구성 확인 | |
| 22 | YOLOv12의 class 수 변경 대응 실습 | gelu_kernel.cu 재실습 | 다양한 클래스 설정 대응 | |
| 23 | 학습 시 mAP, PR Curve 시각화 코드 작성 | global_avg_pool.cu 재실습 | metric 로깅 개선 | |
| 24 | WebGPU를 위한 변환 옵션 점검 | cuda2wgsl_notes.md | WebGPU 대응 전략 학습 | |
| 25 | WGSL 변환 대상 연산 선택 및 정리 | - | 주요 연산 리스트업 | |
| 26 | Fused Conv + FCA 연산 계획 수립 | - | 경량화 전략 세우기 | |
| 27 | 성능 비교 실험 (YOLOv12 vs YOLOv12-FCA) | - | 결과 수집 및 테이블 정리 | |
| 28 | WebGPU 기반 데모 추론 확인 | - | 데모 코드 점검 및 성능 측정 | |
| 29 | 결과 리포트 요약 및 발표 준비 | final_review_day.cu | 전체 복습 및 보고서 작성 | |
| 30 | 기술 발표 + 리더십 공유 문서 정리 | - | 전체 요약 + 실험 코드 정리 및 문서화 | |


# 📅 연장 커리큘럼 Day 31~40
**주제: YOLOv12 + 2bit 양자화 (-1/0/+1) + WebGPU 추론 최적화**

| Day | Main Task (YOLOv12 / FCAFormer) | Daily Task | Description | 보고서 |
| --- | --- | --- | --- | --- |
| 31 | WebGPU 대상 커널 2bit 양자화 구조 설계 | 2bit_quant_kernel.cu | +1/0/-1 양자화 CUDA 커널 설계 | |
| 32 | Quant + FCA 통합 구조 실험 | quant_fca_fusion.cu | CUDA에서 양자화된 attention 연산 실험 | |
| 33 | YOLOv12에서 int4 vs 2bit 실험 비교 | int4_compare.cu | 실수/양자화 모델의 성능 비교 | |
| 34 | WGSL로 Quantized Attention 구조 설계 | cuda2wgsl_notes.md | WGSL로 attention 연산 포팅 시도 | |
| 35 | WebGPU용 최소 YOLOv12 구현 정리 | - | core 추론 블록을 WGSL friendly하게 정리 | |
| 36 | YOLOv12 WebGPU 데모 테스트 및 튜닝 | - | FPS 측정 및 병목 최적화 | |
| 37 | FCAFormer 구조 양자화 실험 | quant_fcaformer.cu | FCAFormer에 양자화 도입 실험 | |
| 38 | WebGPU + YOLOv12 + FCA 최종 구조 확정 | - | 완성된 구조 리포트 정리 | |
| 39 | 최종 기술 리포트 요약 (성능, 구조 등) | - | 성능 비교 요약, FLOPs 정리 | |
| 40 | 아카이빙: 코드+보고서 정리 및 공유 | - | 깃허브, 논문 초안 또는 블로그 포스팅 | |


| Day | Main Task | 실습 목표 |
|-----|-----------|-----------|
| 31 | `TernaryConv`, `TernaryLinear` 구현 | PyTorch에서 -1/0/+1 weight 기반 연산 구현 |
| 32 | 양자화 함수 및 weight ternarization | 학습된 float 모델 → ternary 모델 변환 |
| 33 | ONNX Export | Ternary weight 모델을 ONNX로 저장 |
| 34 | WebGPU 환경 정비 | WebGPU 템플릿 정리, WASM 추론 파이프라인 구조화 |
| 35 | WGSL: TernaryConv 커널 설계 | +1/0/-1 weight 기반 Conv 연산을 WGSL로 구현 |
| 36 | WGSL: Post-process 모듈 구현 | NMS, resize, sigmoid 등 후처리 커널 개발 |
| 37 | YOLOv12-lite WebGPU 추론 파이프라인 완성 | 이미지 → WebGPU 추론 → 결과 시각화 |
| 38 | 속도 비교 실험 | Float32 vs 2bit 성능, 속도 비교 (모바일 포함) |
| 39 | Fallback 구현 | GPU 미탑재 환경에서 CPU 추론 fallback 확인 |
| 40 | 최종 보고서 작성 및 시연 영상 정리 | 기술문서 + 성능 비교 리포트 작성 |

---

### 🧩 필요 파일 구조 예시
```bash
├── yolo_v12_quant/
│   ├── train_ternary.py            # (Day31) ternary weight 생성
│   ├── convert_to_onnx.py          # (Day33)
│   ├── wasm_infer/
│   │   ├── index.html              # WebDemo
│   │   ├── model.onnx              # Ternary 모델
│   │   ├── yolo_wgsl.js            # 추론 로직
│   │   ├── ternary_conv.wgsl       # (Day35)
│   │   └── postprocess.wgsl        # (Day36)
```

## ✅ 환경 구축
```bash
conda env create -f environment.yml
conda activate convnext-cuda
```

시작은 `cuda_tasks/day01_vec_add/` 부터!
