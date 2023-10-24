"""
Model utility functions
"""
# Modules
import torch
from enum import Enum

# Classes
class MeterType(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class Meter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt = ":f", meter_type = MeterType.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.meter_type = meter_type
        self.reset()

    def reset(self):
        self.count = 0
        self.val = 0.0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, val, n = 1):
        self.count += n
        self.val = val
        self.sum += val * n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""

        if self.meter_type is MeterType.NONE:
            fmtstr = ""

        elif self.meter_type is MeterType.AVERAGE:
            fmtstr = "{name} {avg:.3f}"

        elif self.meter_type is MeterType.SUM:
            fmtstr = "{name} {sum:.3f}"

        elif self.meter_type is MeterType.COUNT:
            fmtstr = "{name} {count:.3f}"

        else:
            raise ValueError("invalid meter type %r" % self.meter_type)

        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    """
    Displays a progress for a set of meters
    """
    def __init__(self, num_batches, meters, prefix = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch + 1)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

# Functions
def accuracy(output, labels, top_k = (1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = labels.size(0)

        indices, pred = output.topk(max_k, 1)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []

        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
            res.append(correct_k.mul_(100.0 / (batch_size)))

        return res

