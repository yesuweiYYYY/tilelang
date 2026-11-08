import tilelang.testing
import torch
import example_warp_specialize_gemm_barrierpipe_stage2
import example_warp_specialize_gemm_copy_0_gemm_1
import example_warp_specialize_gemm_copy_1_gemm_0
import example_warp_specialize_gemm_softpipe_stage2


def _requires_sm90():
    if torch.cuda.get_device_capability()[0] < 9:
        print("  (skipped: TMA warp specialization requires SM90+)")
        return True
    return False


def regression_example_warp_specialize_gemm_barrierpipe_stage2():
    if _requires_sm90():
        return
    tilelang.testing.process_func(example_warp_specialize_gemm_barrierpipe_stage2.run_regression_perf, M=1024, N=1024, K=1024)


def regression_example_warp_specialize_gemm_copy_0_gemm_1():
    if _requires_sm90():
        return
    tilelang.testing.process_func(example_warp_specialize_gemm_copy_0_gemm_1.run_regression_perf, M=1024, N=1024, K=1024)


def regression_example_warp_specialize_gemm_copy_1_gemm_0():
    if _requires_sm90():
        return
    tilelang.testing.process_func(example_warp_specialize_gemm_copy_1_gemm_0.run_regression_perf, M=1024, N=1024, K=1024)


def regression_example_warp_specialize_gemm_softpipe_stage2():
    if _requires_sm90():
        return
    tilelang.testing.process_func(example_warp_specialize_gemm_softpipe_stage2.run_regression_perf, M=1024, N=1024, K=1024)


if __name__ == "__main__":
    tilelang.testing.regression()
