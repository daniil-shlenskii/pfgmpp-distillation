class EMAHelper:
    @classmethod
    def decay_to_key(cls, decay: float):
        return f"ema_{decay:.4f}".replace('.', '_')

    @classmethod
    def key_to_decay(cls, key: str):
        return float(key.replace("ema_", "").replace("_", "."))

    @classmethod
    def update(cls, ema_model, online_model, decay):
        for ema_param, param in zip(ema_model.parameters(), online_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
