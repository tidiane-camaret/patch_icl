# 12_small_occupancies
## Q : on omnisynth_medseg, how to retain small occupany objects ?

# 11_1rst_lvl_weaknesses
## Q : on omnisynth_medseg, what explains 1rst level failures ?
## R : mostly explained by tgt cell occupancy (objects < 32 px)

# 10_patchset_cnn_2_lvls
## Q : on omnisynth, behavior of 2nd lvl 32-> 128 ? 
experiments/2d/train.py --config-name 2_omnisynth_medseg_refine model=patchset_cnn arch.l=2 train.batch_size=32 eval.batch_size=32
- Q1 : arch.resolution = [32,64] , loss per level, no info exchange between level : does level 1 actually refines the pred ?
- Q2 : behavior when encode once + crop features instead of encoding crop
- Q2 : level 1 can attend to level 0 register : does it help the prediction ? R 
- Q3 : single loss on fused level logits : see behavior
- Q4 : attention all <-> all (no mask), sampling head for tgt and context, guided by final loss
## R : 


# 9_patchset_omnisynth_medseg_instcopy_effect 
## Q : on a harder task than glyphs, do instcopy help ICL ?
experiments/2d/train.py --config-name 1_omnisynth_medseg model=patchset_cnn arch.l=2 train.batch_size=32 eval.batch_size=32
synth.scene.p_copy=0 : 4gm0xbda
synth.scene.p_copy=0.4 : xij74it0
## R : 

## Q : do tgt repetition helps generalization ? 
experiments/2d/train.py --config-name patchset_cnn_train synth=omniglot data.context_size=1 synth.scene.grid=2 arch.resolution=16
synth.scene.p_copy=0.9 : yr9md63v
synth.scene.p_copy=0.4 : 0qsegdq8
synth.scene.p_copy=0 : um0h88cm

# 8_patchset_enc_nb_feature stages
## Q : base config : arch.enc_dims=[64, 64, 64, 64]. do we need all those ? 
experiments/2d/train.py --config-name patchset_cnn_train synth=omniglot data.context_size=1 arch.resolution=32 arch.enc_dims=[64] arch.l=2 train.topk_k=64 synth.scene.p_copy=0 synth.source=medseg synth.scene.placement=random synth.scene.max_nb_objects=8 synth.scene.background=image
## R : 


# 8_universeg_patchset_learning_rate
## Q : currently different lr for both. how does it impact learning curve
## R : 


# 7_omnisynth_medseg_comp
## Q : patchset and universeg perf comp on omnisynth_medseg 
experiments/2d/train.py --config-name 1_omnisynth_medseg 
model=patchset_cnn arch.enc_dims=[64] arch.l=2 : xkidx6b0
model=universeg : vc6qh8t2
## R : 


# 6_patchset_cnn_finer_res_effect
## Q : when sampling at higher res, the input has more details but more patches -> more long-range attention spatially-wise
- Q1 : dice at resolutions 16,32,64 compared to universeg 
experiments/2d/train.py --config-name patchset_cnn_train synth=omniglot data.context_size=1 synth.scene.grid=2
arch.resolution=64 train.batch_size=16 eval.batch_size=16 : obtytmi5
arch.resolution=128 train.batch_size=8 eval.batch_size=8 : lzo4mua5
## R : 

# 6_patchset_cnn_finer_res_effect
## Q : when sampling at higher res, the input has more details but more patches -> more long-range attention spatially-wise
- Q1 : dice at resolutions 16,32,64 compared to universeg 
experiments/2d/train.py --config-name patchset_cnn_train synth=omniglot data.context_size=1 synth.scene.grid=2
arch.resolution=64 train.batch_size=16 eval.batch_size=16 : obtytmi5
arch.resolution=128 train.batch_size=8 eval.batch_size=8 : lzo4mua5
## R : 

# 5_patchset_cnn_instcopy_effect 
## Q : do tgt repetition helps generalization ? 
experiments/2d/train.py --config-name patchset_cnn_train synth=omniglot data.context_size=1 synth.scene.grid=2 arch.resolution=16
synth.scene.p_copy=0.9 : yr9md63v
synth.scene.p_copy=0.4 : 0qsegdq8
synth.scene.p_copy=0 : um0h88cm

## R :  0.4 and 0 : 0.58 dice, 0.9 : 0.45 dice. in this simple setting, having more non-copy items helps. TODO : try with more realistic tasks (e.g. omnisynth medsegbench)

# 4_weaknesses (03_07_26)
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
