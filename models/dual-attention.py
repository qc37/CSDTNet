class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()

        # self.q_conv.weight = self.k_conv.weight
        # self.q_conv.bias = self.k_conv.bias
        # self.v_conv = nn.Conv1d(channels, channels, 1)

        # Position Attention Module (PAM) Convolutions
        self.q_conv_pam = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv_pam = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv_pam = nn.Conv1d(channels, channels, 1)

        # Channel Attention Module (CAM) Convolutions
        self.q_conv_cam = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv_cam = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv_cam = nn.Conv1d(channels, channels, 1)

        # Normalization and activation
        self.trans_conv_pam = nn.Conv1d(channels, channels, 1)
        self.trans_conv_cam = nn.Conv1d(channels, channels, 1)
        self.after_norm_pam = nn.BatchNorm1d(channels)
        self.after_norm_cam = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # Final convolution layer
        self.final_conv = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

    def forward(self, x, xyz):
        x = x + xyz

        # Position Attention Module (PAM)
        x_q_pam = self.q_conv_pam(x).permute(0, 2, 1)
        x_k_pam = self.k_conv_pam(x)
        x_v_pam = self.v_conv_pam(x)
        energy_pam = torch.bmm(x_q_pam, x_k_pam)
        attention_pam = self.softmax(energy_pam)
        attention_pam = attention_pam / (1e-9 + attention_pam.sum(dim=1, keepdim=True))
        x_r_pam = torch.bmm(x_v_pam, attention_pam)
        x_r_pam = self.act(self.after_norm_pam(self.trans_conv_pam(x - x_r_pam)))
        pam_output = x + x_r_pam

        # Channel Attention Module (CAM)
        x_q_cam = self.q_conv_cam(x)
        x_k_cam = self.k_conv_cam(x).permute(0, 2, 1)
        x_v_cam = self.v_conv_cam(x)
        energy_cam = torch.bmm(x_k_cam, x_q_cam)
        attention_cam = self.softmax(energy_cam)
        attention_cam = attention_cam / (1e-9 + attention_cam.sum(dim=1, keepdim=True))
        x_r_cam = torch.bmm(x_v_cam, attention_cam)
        x_r_cam = self.act(self.after_norm_cam(self.trans_conv_cam(x - x_r_cam)))
        cam_output = x + x_r_cam

        # Combine the outputs of PAM and CAM
        feat_sum =pam_output + cam_output

        # Final convolution
        output = self.final_conv(feat_sum)

        return output
