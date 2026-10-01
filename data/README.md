# Data

`processed.cleveland.data` is the unmodified Cleveland file from the UCI Machine Learning Repository
([Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), DOI
[10.24432/C52P4X](https://doi.org/10.24432/C52P4X)), extracted from the
[official ZIP](https://archive.ics.uci.edu/static/public/45/heart+disease.zip).

- 303 records, 13 predictors and the outcome `num` (0 = no disease, 1-4 = disease); no header row.
- Missing values are coded as `?` (4 in `ca`, 2 in `thal`).
- SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` (checked by `heartcal.data.load_raw`).

Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Citation: Janosi A, Steinbrunn W,
Pfisterer M, Detrano R. *Heart Disease* [Dataset]. UCI Machine Learning Repository; 1989.
