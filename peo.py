from torch.optim.optimizer import Optimizer,required
class PEO(Optimizer):
    '''
        Parzen Entropy Optimizer
        自定义优化器 用于Parzen窗信息熵同类熵减、异类熵增处理
        cache_grad()先缓存同类梯度,step()梯度做差乘以学习率调整参数
    '''

    # 用于缓存同类别梯度
    cache_param_group = 0

    def __init__(self,params, lr=required):
        defaults = dict(lr=lr)
        super(PEO, self).__init__(params, defaults)

    def step(self):
        for group,cache_group in zip(self.param_groups,self.cache_param_group):
            for p,c_p in zip(group['params'],cache_group['params']):
                if p.grad is None or c_p.grad is None:
                    continue
                p.data.add_(-group['lr'] * (c_p.grad.data-p.grad.data))

    # 缓存梯度信息用于稍后参数调整
    def cache_grad(self):
        self.cache_param_group = self.param_groups
