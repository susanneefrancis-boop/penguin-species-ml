# penguin-species-ml# AI/ML-Based Species Identification in Antarctic Penguins

A machine learning approach to classifying *Pygoscelis* penguin species (Adelie, Chinstrap, Gentoo) from morphometric data.

**MSc Zoology: AI/ML Mini Project**

## Live Demo
(https://f6b1cd380068c224fa.gradio.live/)

## Biological Problem
Species identification in the field usually depends on visual and morphological inspection. This project tests whether standard body measurements alone are enough for a machine learning model to separate three sympatric penguin species automatically.

## Dataset
- **Source:** Palmer Archipelago (Antarctica) penguin data, collected by Dr. Kristen Gorman and the Palmer Station LTER programme
- **Reference:** K. B. Gorman, T. D. Williams, and W. R. Fraser, "Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins (genus *Pygoscelis*)," *PLoS ONE*, vol. 9, no. 3, e90081, 2014.
- **Size:** 344 birds, 333 after removing rows with missing values
- **Features:** bill length, bill depth, flipper length, body mass, sex, island (Biscoe, Dream, Torgersen)
- **Classes:** Adelie, Chinstrap, Gentoo

## Methods
1. Data loading and exploratory analysis
2. Preprocessing: removed missing values, one-hot encoded island and sex, label-encoded species, stratified 80/20 train/test split, standard scaling fitted on the training set only
3. Models compared with 5-fold stratified cross-validation: Logistic Regression, Random Forest, SVM (RBF kernel)
4. Best model evaluated once on the held-out test set
5. Evaluation with confusion matrix, classification report and one-vs-rest ROC curves
6. Interpretation with Random Forest feature importance and PCA

## Results
| Model | CV Accuracy (mean) | Test Accuracy |
|---|---|---|
| Logistic Regression | 0.9963 | 0.9851 |
| Random Forest | 0.9925 | 1.0000 |
| SVM (RBF) | 0.9963 | 0.9851 |

- Best model by cross-validation: **Logistic Regression**
- Test set: 67 samples, 66 correctly classified (one Adelie predicted as Chinstrap)
- Morphometrics-only check (no island or sex): 0.9880 +/- 0.0112

## Figures
| File | Content |
|---|---|
| `fig2_class_distribution.png` | Class distribution |
| `fig3_pairplot.png` | Pairwise feature distributions |
| `fig4_pca.png` | PCA of morphometric features |
| `fig6_confusion_matrix.png` | Confusion matrix |
| `fig7_roc_curves.png` | ROC curves |
| `fig8_feature_importance.png` | Random Forest feature importance |

## How to Run
```bash
pip install -r requirements.txt
python app.py
```
Then open the local address shown in the terminal.

## Repository Files
- `Penguin_Species_ML.ipynb`: full analysis notebook
- `penguins.csv`: dataset
- `penguin_model.joblib`: trained model pipeline
- `app.py`: Gradio web app
- `requirements.txt`: Python dependencies
- `fig*.png`: figures used in the report

## Limitations
- Small, clean, curated dataset, so this is a relatively easy classification task
- Test set is small (67 samples), so a single error changes accuracy by about 1.5%
- Island is a strong location-based cue and may inflate performance; a morphometrics-only comparison addresses this
- Results apply to these three species in the Palmer Archipelago and may not generalise elsewhere

## License and Data Credit
Data courtesy of Dr. Kristen Gorman and the Palmer Station LTER programme. Please cite the reference above when reusing the data.
