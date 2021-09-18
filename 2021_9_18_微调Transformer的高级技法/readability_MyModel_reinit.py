class MyModel(nn.Module):
            
    def __init__(self, reinit_n_layers=0):        
        super().__init__() 
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')       
        self.regressor = nn.Linear(768, 1)  
        self.reinit_n_layers = reinit_n_layers
        if reinit_n_layers > 0: self._do_reinit()            
            
    def _do_reinit(self):
        # Re-init pooler.
        self.roberta_model.pooler.dense.weight.data.normal_(mean=0.0, std=self.roberta_model.config.initializer_range)
        self.roberta_model.pooler.dense.bias.data.zero_()
        for param in self.roberta_model.pooler.parameters():
            param.requires_grad = True
        
        # Re-init last n layers.
        for n in range(self.reinit_n_layers):
            self.roberta_model.encoder.layer[-(n+1)].apply(self._init_weight_and_bias)
            
    def _init_weight_and_bias(self, module):                        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.roberta_model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)        
 
    def forward(self, input_ids, attention_mask):        
        raw_output = self.roberta_model(input_ids, attention_mask, return_dict=True)        
        pooler = raw_output["pooler_output"]    # Shape is [batch_size, 768]
        output = self.regressor(pooler)         # Shape is [batch_size, 1]
        return output 