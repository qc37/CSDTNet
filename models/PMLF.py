class AvgTopKPooling(nn.Module):
    def __init__(self, ksize=3, kk=3):
        super(AvgTopKPooling, self).__init__()
        self.ksize = ksize
        self.kk = kk

    def forward(self, inputs):
        k_size = self.ksize
        channel = inputs.size(1)

        unfolded = F.unfold(inputs, kernel_size=k_size, stride=k_size)
        unfolded = unfolded.view(inputs.size(0), channel, k_size * k_size, -1)
        unfolded = unfolded.permute(0, 3, 1, 2).contiguous() 

 
        topk_values, _ = torch.topk(unfolded, self.kk, dim=-1)
        
        avg_topk_per_block = torch.mean(topk_values, dim=-1)         
        avg_topk_all_block = torch.mean(avg_topk_per_block, dim=1)      
      

        if torch.mean(avg_topk_all_block) >= L:
            output = avg_topk_all_block
        else:
            max_topk_all_block, _ = torch.max(avg_topk_per_block, dim=1)
            output = max_topk_all_block
       

        return output
