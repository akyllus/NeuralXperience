class ConvNeXt_XL(ConvNeXt):
    def __init__(self, in_channels, out_channels, drop_path_rate=0.5, layer_scale_init_value=1e-6):
        super().__init__(in_channels, out_channels, drop_path_rate, layer_scale_init_value)
        self.layers = [
            ConvNeXtBlock(out_channels, drop_path_rate, layer_scale_init_value, use_conv_shortcut=False)
            for _ in range(36)]