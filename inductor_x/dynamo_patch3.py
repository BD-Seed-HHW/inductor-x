# Issue: Error
# x.permute(0, 2, 1, 3).contiguous().view(xxx)
# Message: 
import torch
from torch.library import Library, impl
python_dispatcher_lib = Library("aten", "IMPL", "PythonDispatcher")
@impl(python_dispatcher_lib, "embedding_backward")
def embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse):
    if sparse:
        raise RuntimeError("the current NPU does not yet support sparse tensor, when sparse is set to True")
    return torch.ops.aten.embedding_dense_backward(grad, indices,  num_weights, padding_idx, scale_grad_by_freq)