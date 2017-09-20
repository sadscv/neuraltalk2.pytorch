from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        # input 是model的输出，维度为batchsize*
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        '''
        torch.view:Returns a new tensor with the same data but different size.

        The returned tensor shares the same data and must have the same number of elements,
        but may have a different size. A tensor must be contiguous() to be viewed.

        >>> x = torch.randn(4, 4)
        >>> x.size()
        torch.Size([4, 4])
        >>> y = x.view(16)
        >>> y.size()
        torch.Size([16])
        >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
        >>> z.size()
        torch.Size([2, 8])
        '''
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        # gather用法 https://discuss.pytorch.org/t/select-specific-columns-of-each-row-in-a-torch-tensor/497
        # mask 如果target中某些项为0(即代表单词UNK）则将gather后的结果末尾去掉相应数量的单元。（虽然我并不知道这有什么用）
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)