import correlation_cuda
import torch
from torch.autograd import Function
from torch.nn.modules.module import Module


class CorrelationFunction(Function):
    """PyTorch autograd wrapper for the CUDA correlation op.

    NOTE:
      The upstream LCCNet correlation layer often ships with the *legacy* autograd
      Function API (instance-based forward/backward). Newer PyTorch versions
      (and some configurations) reject that legacy form.

      This wrapper uses the *new-style* API:
        - forward(ctx, ...)
        - backward(ctx, ...)
      while keeping the underlying correlation_cuda bindings unchanged.
    """

    # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)

    @staticmethod
    def forward(ctx,
                input1,
                input2,
                pad_size=3,
                kernel_size=3,
                max_displacement=20,
                stride1=1,
                stride2=2,
                corr_multiply=1):
        # Save tensors + hyperparams for backward.
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(
                input1,
                input2,
                rbot1,
                rbot2,
                output,
                pad_size,
                kernel_size,
                max_displacement,
                stride1,
                stride2,
                corr_multiply,
            )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        pad_size = ctx.pad_size
        kernel_size = ctx.kernel_size
        max_displacement = ctx.max_displacement
        stride1 = ctx.stride1
        stride2 = ctx.stride2
        corr_multiply = ctx.corr_multiply

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(
                input1,
                input2,
                rbot1,
                rbot2,
                grad_output,
                grad_input1,
                grad_input2,
                pad_size,
                kernel_size,
                max_displacement,
                stride1,
                stride2,
                corr_multiply,
            )

        # Return grads for:
        #   input1, input2, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply
        return grad_input1, grad_input2, None, None, None, None, None, None


class Correlation(Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        # New-style autograd Function invocation uses the .apply(...) classmethod.
        result = CorrelationFunction.apply(
            input1,
            input2,
            self.pad_size,
            self.kernel_size,
            self.max_displacement,
            self.stride1,
            self.stride2,
            self.corr_multiply,
        )
        return result