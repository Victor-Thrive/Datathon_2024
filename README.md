
# Datathon_2024

# Model Documentation: Predicting Student Performance

## Project Overview

### Predicting Student Academic Performance in Nigeria

### Objective:

This project aims to predict whether students will pass or fail WAEC or JAMB exams based on their performance in senior secondary school maths, English and other demographic and other related factor Exams. Multiple classification models were used to compare their predictive accuracy.

## Dataset:

-   **Source**: `student_survey.csv`, containing student survey data, including maths and English scores.

-   **Preprocessing**: The students’ scores were converted into a binary outcome (Pass/Fail) based on predefined exam grade intervals. The dataset was then split into training and testing sets.

## Data Generation with Python

-   Fork the repository
-   Clone the repository `git clone repo-url`
-   Navigate into the cloned repository `cd datathon_2024`
-   Create a virtual environment `python -m venv venv`
-   Activate the virtual environment `python -m venv venv` or `source venv/bin/activate`
-   Install project dependencies `pip install -r requirements.txt`
-   Generate the data into csv file `python generate_data.py`

## Prerequisites

**R version**: `4.x.x`

**Required Libraries/Packages**:

```{r eval=false}}
library(reticulate)
library(tidyverse)
library(tidymodels)
library(parsnip)
library(discrim)
library(ranger)
library(kknn)
library(naivebayes)
library(yardstick)

```

Install these packages using:

```{r eval=false}}
install.packages(c("tidyverse", "tidymodels", "parsnip", "discrim", "ranger", "kknn", "naivebayes", "yardstick"))

```

## Data Pre-processing

### **Load and Clean the Data**

-   Load the data and create a new binary variable, `result`, based on `maths_score` and `english_score`. If the score falls within predefined grade intervals, the student passes; otherwise, they fail.

```{r eval=false}}
df <- read.csv("student_survey.csv", stringsAsFactors = TRUE) |> 
  mutate(result = case_when(
    maths_score %in%  exam_grade ~ "Pass",
    english_score %in% exam_grade ~ "Pass",
    TRUE ~ "Fail"
  )) |> 
  select(-c("maths_score", "english_score"))

```

### **Split the Data**

-   The dataset was split into 80% training and 20% testing sets, ensuring balanced classes with stratification.

```{r eval=false}}
set.seed(123)
df_split <- initial_split(df, prop = 0.8, strata = result)
train_data <- training(df_split)
test_data <- testing(df_split)

```

## **Model Development**

### **Preprocessing Recipe**

-   Create a recipe for preprocessing the data: dummy encoding for categorical variables and removing zero-variance predictors.

```{r eval=false}}
recipe <- recipe(result ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

```

### **Model Selection**

You used several models to predict student performance:

-   **Logistic Regression**

-   **Decision Tree**

-   **Random Forest**

-   **Support Vector Machine (SVM)**

-   **k-Nearest Neighbors (k-NN)**

-   **Naive Bayes**

-   **Neural Networks**

-   **Linear Discriminant Analysis (LDA)**

-   **Multinomial Logistic Regression**

```{r eval=false}}
log_reg_model <- logistic_reg() %>% set_engine("glm")
tree_model <- decision_tree() %>% set_engine("rpart") %>% set_mode("classification")
rf_model <- rand_forest() %>% set_engine("ranger") %>% set_mode("classification")
svm_model <- svm_rbf() %>% set_engine("kernlab") %>% set_mode("classification")
knn_model <- nearest_neighbor() %>% set_engine("kknn") %>% set_mode("classification")
nb_model <- naive_Bayes() %>% set_engine("naivebayes") %>% set_mode("classification")
nn_model <- mlp(hidden_units = 5) %>% set_engine("nnet") %>% set_mode("classification")
lda_model <- discrim_linear() %>% set_engine("MASS") %>% set_mode("classification")
multinom_model <- multinom_reg() %>% set_engine("nnet") %>% set_mode("classification")

```

## **Model Fitting**

Each model was fitted using a `workflow` that includes the preprocessing recipe.

```{r eval=false}}
log_reg_fit <- fit(log_reg_workflow, data = train_data)
tree_fit <- fit(tree_workflow, data = train_data)
rf_fit <- fit(rf_workflow, data = train_data)
# Continue fitting other models...

```

## **Model Evaluation**

You created a function to evaluate each model on the test data and calculate metrics such as accuracy.

```{r eval=false}}
evaluate_model <- function(model, test_data, response_col) {
  test_pred <- predict(model, test_data) %>% bind_cols(test_data)
  response_col_sym <- rlang::sym(response_col)
  metrics_results <- test_pred %>% metrics(truth = !!response_col_sym, estimate = .pred_class)
  return(metrics_results)
}

```

You evaluated all the models and combined their results for comparison.

```{r eval=false}}
log_reg_results <- evaluate_model(log_reg_fit, test_data, "result")
# Continue evaluating other models...

all_results <- bind_rows(
  log_reg_results %>% mutate(model = "Logistic Regression"),
  tree_results %>% mutate(model = "Decision Tree"),
  rf_results %>% mutate(model = "Random Forest"),
  # Continue binding other models...
)

best_model <- all_results %>%
  filter(.metric == "accuracy") %>%
  arrange(desc(.estimate))

```

## **Results**

The models were compared based on their accuracy:

```{r eval=false}}
best_model

```

## **Conclusion**

-   Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes, LDA, and Multinomial Logistic Regression performed the best in predicting student outcomes, all achieving an accuracy of 90.6%.

k-NN and Neural Networks had slightly lower performance, indicating that these methods might need further tuning or are less suited for this dataset.
