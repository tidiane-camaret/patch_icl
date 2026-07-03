
# 6_patchset_cnn_2_lvls
## Q : on omnisynth, is second level of patchset_cnn able to produce masks at full resolution ? 
- Q1 : level 0 accuracy (resolution 16, omnisynth : 2x2 grid, 1 ctx)
## R : 

# 5_patchset_cnn_2_instcopy_effect 
## Q : do tgt repetition helps generalization ? 
experiments/2d/train.py --config-name patchset_cnn_train synth=omniglot data.context_size=1 synth.scene.grid=2 arch.resolution=16
synth.scene.p_copy=0.9 : damymmfw  
synth.scene.p_copy=0 : 
## R : 

# 4_weaknesses_universeg (03_07_26)
## Q : when trained and eval on omnisynth, where does universeg fails ?
## R : 

# 3_omnisynth_to_medical (03_07_26)
## Q : Does training on omnisynth generalize on medical tasks ? 
- Q1 : when training universeg on omnisynth, what is the impact of starting from pretrained weights vs from scratch ? 
experiments/2d/train.py synth=omniglot data.context_size=1 synth.scene.p_copy=0 
train.pretrained=false : zj98g7yd
train.pretrained=true : rsbifrsl
- R1 : faster convergence when pretrained=false, 0.7 vs 0.6 after 10 epochs
- Q2 :  what is then the performance on biomedparse ? 
