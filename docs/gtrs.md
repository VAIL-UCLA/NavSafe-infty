# GTRS-Dense evaluation

The inference implementation is bundled with NavSafe. The same adapter runs baseline and SimScale-augmented checkpoints; their learned weights differ, while all variants default to the same 16,384-entry evaluation trajectory vocabulary.

| Checkpoint | Backbone | Vocabulary |
| --- | --- | --- |
| gtrs_dense_vov.ckpt (previously evaluated baseline) | V2-99 | 16384 |
| gtrs_dense_resnet_sim_expert_navhard.ckpt | ResNet34 | 16384 |
| gtrs_dense_resnet_sim_reward_navhard.ckpt | ResNet34 | 16384 |
| gtrs_dense_vov_sim_expert_navhard.ckpt | V2-99 | 16384 |
| gtrs_dense_vov_sim_reward_navhard.ckpt | V2-99 | 16384 |

Pass the checkpoint path to the normal evaluation command with model type `gtrs_dense`. All listed checkpoints default to the bundled `16384.npy`; neither `NAVSAFE_GTRS_VOCAB_SIZE` nor `NAVSAFE_GTRS_VOCAB` needs to be set. Backbone selection is automatic. Checkpoint weights are not bundled.

The adapter uses the selected external vocabulary rather than the checkpoint's stored vocabulary tensor. Both official vocabulary files remain bundled for explicit reproduction of historical runs; their SHA256 values and upstream revision are recorded in [provenance](../navsafe/modelzoo/gtrs_dense/PROVENANCE.md).

Historical NavSafe SimScale evaluations explicitly used 8192 candidates, while the evaluated V2-99 baseline used 16384. Those historical results do not become 16384-candidate results when this default changes: rerun the SimScale evaluations for a comparison using the same candidate vocabulary. When reusing an old launch script, remove its explicit 8192 size/path overrides or set both to the 16384 vocabulary.

An explicit `NAVSAFE_GTRS_VOCAB` can override the bundled vocabulary; its size must agree with `NAVSAFE_GTRS_VOCAB_SIZE`. All policy tensors must match; only the external vocabulary and the unused legacy query embedding are exempted. Backbone initialization does not download ImageNet/DD3D weights.

Upstream checkpoint releases and instructions: [SimScale](https://github.com/OpenDriveLab/SimScale). This integration does not add training support or a dependency on either source checkout.
