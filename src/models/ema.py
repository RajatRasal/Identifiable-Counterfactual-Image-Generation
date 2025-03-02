class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.store()
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = (
                self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
            )

    def store(self):
        self.state = {
            name: param.clone() for name, param in self.model.named_parameters()
        }

    def apply_shadow(self):
        self.store()
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            param.data.copy_(self.state[name])

    # Context manager interface
    def __enter__(self):
        self.apply_shadow()  # apply EMA weights on entering context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()  # restore the original weights on exiting context
