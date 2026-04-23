  切回 YAMNet（如需对比）

  将 main() 中的 ASTFeatureExtractor 改回                    
  FeatureExtractor，backbone_unfreeze_layers 改回
  yamnet_unfreeze_layers，cache version 改回                 
  v17_flow_fixed_signal，删除 AST checkpoint 即可。