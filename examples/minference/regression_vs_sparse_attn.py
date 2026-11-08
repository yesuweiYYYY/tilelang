import tilelang.testing
import torch
import example_vertical_slash_sparse_attn


def regression_example_vertical_slash_sparse_attn():
    if torch.cuda.get_device_capability()[0] < 9:
        print("  (skipped: TMA sparse-attention kernel requires SM90+)")
        return
    tilelang.testing.process_func(example_vertical_slash_sparse_attn.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
