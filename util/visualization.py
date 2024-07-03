from torch.utils.tensorboard import SummaryWriter


def writer(logs_dir):
    # max_queue，log数据最多等待队列5，flush_secs 30s自动更新一次
    return SummaryWriter(log_dir=logs_dir, max_queue=5, flush_secs=30)
