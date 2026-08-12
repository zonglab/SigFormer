# Third-party data and model-resource notice

The repository's **MIT License applies to SigFormer source code authored for this project**. It does not grant new rights to third-party datasets, reference catalogues, published cohort resources, or other material that is governed by separate terms.

The local validation handoff intentionally retains the current `SigFormer/resource/` directory and the generated `example_data/template_mock/` files so that the package can be tested before publication. These paths are ignored by `.gitignore` by default to reduce the chance of accidental public redistribution.

## Resources requiring an explicit licence check before public release

- COSMIC SBS reference tables, including the bundled COSMIC v3.4 files.
- Any PCAWG or normal-tissue resource copied from an external publication, consortium, or database.
- The generated template mock cohort, because it was generated from COSMIC reference signatures.
- The pretrained SigFormer model weights. Confirm that the data and terms under which the model was trained permit both model training and public redistribution of the resulting checkpoint. Current COSMIC licence terms explicitly restrict use of COSMIC data to train AI models without express written consent, so this point should be resolved with the data licensor before a public release if COSMIC data were used in model training.

Do not assume that placing an MIT `LICENSE` file at repository root changes the licence of these resources. Obtain permission or replace restricted files with a documented download step before publishing the repository.
