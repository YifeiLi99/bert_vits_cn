class SynthesizerEval(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
            self,
            n_vocab,
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            n_speakers=0,
            gin_channels=0,
            use_sdp=False,
            **kwargs
    ):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )
        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
        )
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def remove_weight_norm(self):
        print("Removing weight norm...")
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()

    def infer(self, x, x_lengths, bert=None, sid=None, noise_scale=1, length_scale=1, max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, bert)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w + 0.35)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def infer_pause(self, x, x_lengths, bert, pause_mask, pause_value, sid=None, noise_scale=1, length_scale=1,
                    max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, bert)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w + 0.35)
        w_ceil = w_ceil * pause_mask + pause_value
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def infer_stream(self, x, x_lengths, bert, sid=None, noise_scale=1, length_scale=1):
        print("-----------------------------------------------------------------------------")
        import datetime
        import numpy
        print(datetime.datetime.now())
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, bert)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w + 0.35)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        len_z = z_p.size()[2]
        print('frame size is: ', len_z)
        if (len_z < 100):
            print('no nead steam')
            z = self.flow(z_p, y_mask, g=g, reverse=True)
            one_time_wav = self.dec(z, g=g)[0, 0].data.cpu().float().numpy()
            return one_time_wav

        # can not change these parameters
        hop_length = 256  # bert_vits.json
        hop_frame = 9
        hop_sample = hop_frame * hop_length
        stream_chunk = 100
        stream_index = 0
        stream_out_wav = []

        while (stream_index + stream_chunk < len_z):
            if (stream_index == 0):  # start frame
                cut_s = stream_index
                cut_s_wav = 0
            else:
                cut_s = stream_index - hop_frame
                cut_s_wav = hop_sample

            if (stream_index + stream_chunk > len_z - hop_frame):  # end frame
                cut_e = stream_index + stream_chunk
                cut_e_wav = 0
            else:
                cut_e = stream_index + stream_chunk + hop_frame
                cut_e_wav = -1 * hop_sample

            z_chunk = z_p[:, :, cut_s:cut_e]
            m_chunk = y_mask[:, :, cut_s:cut_e]
            z_chunk = self.flow(z_chunk, m_chunk, g=g, reverse=True)
            o_chunk = self.dec(z_chunk, g=g)[0, 0].data.cpu().float().numpy()
            o_chunk = o_chunk[cut_s_wav:cut_e_wav]
            stream_out_wav.extend(o_chunk)
            stream_index = stream_index + stream_chunk
            print(datetime.datetime.now())

        if (stream_index < len_z):
            cut_s = stream_index - hop_frame
            cut_s_wav = hop_sample
            z_chunk = z_p[:, :, cut_s:]
            m_chunk = y_mask[:, :, cut_s:]
            z_chunk = self.flow(z_chunk, m_chunk, g=g, reverse=True)
            o_chunk = self.dec(z_chunk, g=g)[0, 0].data.cpu().float().numpy()
            o_chunk = o_chunk[cut_s_wav:]
            stream_out_wav.extend(o_chunk)

        stream_out_wav = numpy.asarray(stream_out_wav)
        return stream_out_wav
