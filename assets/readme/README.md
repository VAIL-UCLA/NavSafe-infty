# README visual assets

Logo and framework source: [NavSafe project website repository](https://github.com/navsafe-vail/navsafe-vail.github.io), commit `57bff2d83a4e17ad3f32f99fd02fda968f4b0d97`.

- `navsafe-logo.png`: copied unchanged from the website's `logo.png`.
- `teaser.png`: rasterized from the user-provided `fig1-teaser.pdf` at 2200 pixels on the longest edge (PDF SHA-256: `344dfdc676060ffe4c198fbb9844a36b95679b0a80ae01661e6bf846991912d8`).
- `framework.png`: rasterized from `assets/figures/fig2.pdf` at 1800 pixels on the longest edge.
- `website.svg`, `paper.svg`, and `dataset.svg`: Shields.io badges matching the [BridgeSim README](https://github.com/VAIL-UCLA/BridgeSim). The Hugging Face label is adapted from `Checkpoints` to `Dataset`; all badge links in the main README point to NavSafe resources.

Badge sources (saved locally to avoid a rendering dependency on Shields.io):

- Website: `https://img.shields.io/badge/Website-Explore%20Now-blueviolet?style=flat&logo=google-chrome`
- Paper: `https://img.shields.io/badge/arXiv-Paper-%3CCOLOR%3E.svg`
- Dataset: `https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface`

Figure export (requires Poppler):

```bash
pdftoppm -f 1 -singlefile -scale-to 2200 -png /path/to/fig1-teaser.pdf assets/readme/teaser
pdftoppm -f 1 -singlefile -scale-to 1800 -png /path/to/navsafe-vail.github.io/assets/figures/fig2.pdf assets/readme/framework
```

Figures and branding belong to the NavSafe project and their original rights holders.
