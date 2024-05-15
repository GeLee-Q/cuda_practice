
# LLM 大模型 CUDA算子优化专题

## Docker 构造环境

```
docker run -it \
            -m 81920M \
            --memory-swap=81920M \
            --shm-size=128G \
            --privileged \
            --net=host \
            --name=cuda_llm \
            --gpus all \
            -v $PWD:/workspace \
            -w /workspace \
            -v /etc/localtime:/etc/localtime\
            nvcr.io/nvidia/pytorch:22.01-py3 bash

docker exec -it cuda_llm bash
```

## 运行测试脚本
- 首先会编译产出
- 然后使用 ctest 自动进行测试

```
bash run.sh 
```

## GEMM 测试结果

```
1: Test command: /workspace/cuda_practice/build/test_gemm
1: Test timeout computed to be: 10000000
1: my_mul: 0.055548s
1: cublas: 0.000831s
1: Accuracy test passed!
1/1 Test #1: test_gemm ........................   Passed    1.01 sec

100% tests passed, 0 tests failed out of 1
```