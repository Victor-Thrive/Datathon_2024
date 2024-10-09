library(reticulate)
library(tidyverse)
library(tidymodels)
library(parsnip)
library(discrim)
library(ranger)
library(kknn)
library(naivebayes)
library(yardstick) 

exam_grade <- c("50 - 54", "55 - 59", "60 - 64", 
                "65 - 69", "70 - 75", "80 - 100")

df<- read.csv("student_survey.csv",stringsAsFactors = TRUE) |>
  mutate(result = case_when(
    maths_score %in%  exam_grade ~ "Pass",
    english_score %in% exam_grade ~ "Pass",
    TRUE ~ "Fail"  # If no condition is met, result is set to NA
  )) |>
  select(-c("maths_score","english_score"))



# Split the data into training and testing sets
set.seed(123)
df_split <- initial_split(df, prop = 0.8, strata = result) # Stratify to maintain class balance
train_data <- training(df_split)
test_data <- testing(df_split)

# Ensure the response variable is a factor
train_data$result <- as.factor(train_data$result)
test_data$result <- as.factor(test_data$result)

# Pre-processing recipe
recipe <- recipe(result ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%  # Convert categorical to dummy variables
  step_zv(all_predictors())                   # Remove zero-variance predictors

# Logistic Regression
log_reg_model <- logistic_reg() %>%
  set_engine("glm")

log_reg_workflow <- workflow() %>%
  add_model(log_reg_model) %>%
  add_recipe(recipe)

# Decision Trees
tree_model <- decision_tree() %>%
  set_engine("rpart") %>%
  set_mode("classification")

tree_workflow <- workflow() %>%
  add_model(tree_model) %>%
  add_recipe(recipe)

# Random Forest
rf_model <- rand_forest() %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe)

# Support Vector Machine
svm_model <- svm_rbf() %>%
  set_engine("kernlab") %>%
  set_mode("classification");

svm_workflow <- workflow() %>%
  add_model(svm_model) %>%
  add_recipe(recipe)

# K-Nearest Neighbors
knn_model <- nearest_neighbor() %>%
  set_engine("kknn") %>%
  set_mode("classification");

knn_workflow <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(recipe);

# Naive Bayes
nb_model <- naive_Bayes() %>%
  set_engine("naivebayes") %>%
  set_mode("classification");

nb_workflow <- workflow() %>%
  add_model(nb_model) %>%
  add_recipe(recipe);

# Neural Networks
nn_model <- mlp(hidden_units = 5) %>%
  set_engine("nnet") %>%
  set_mode("classification");

nn_workflow <- workflow() %>%
  add_model(nn_model) %>%
  add_recipe(recipe);

# LDA
lda_model <- discrim_linear() %>%
  set_engine("MASS") %>%
  set_mode("classification");

lda_workflow <- workflow() %>%
  add_model(lda_model) %>%
  add_recipe(recipe);

# Multinomial Logistic Regression
multinom_model <- multinom_reg() %>%
  set_engine("nnet") %>%
  set_mode("classification");

multinom_workflow <- workflow() %>%
  add_model(multinom_model) %>%
  add_recipe(recipe);

# Fit each model
log_reg_fit <- fit(log_reg_workflow, data = train_data)
tree_fit <- fit(tree_workflow, data = train_data)
rf_fit <- fit(rf_workflow, data = train_data)
svm_fit <- fit(svm_workflow, data = train_data)
knn_fit <- fit(knn_workflow, data = train_data)
nb_fit <- fit(nb_workflow, data = train_data)
nn_fit <- fit(nn_workflow, data = train_data)
lda_fit <- fit(lda_workflow, data = train_data)
multinom_fit <- fit(multinom_workflow, data = train_data)


# Create a function to evaluate the models
evaluate_model <- function(model, test_data, response_col) {
  # Make predictions
  test_pred <- predict(model, test_data) %>%
    bind_cols(test_data)  # Combine predictions with test data
  
  # Ensure the response column is in the test data
  response_col_sym <- rlang::sym(response_col)
  
  # Calculate classification metrics
  metrics_results <- test_pred %>%
    metrics(truth = !!response_col_sym, estimate = .pred_class)  # Use the correct response column
  
  return(metrics_results)
}


# Evaluate each model
log_reg_results <- evaluate_model(log_reg_fit, test_data, response_col = "result")
tree_results <- evaluate_model(tree_fit, test_data, response_col = "result")
rf_results <- evaluate_model(rf_fit, test_data, response_col = "result")
svm_results <- evaluate_model(svm_fit, test_data, response_col = "result")
knn_results <- evaluate_model(knn_fit, test_data, response_col = "result")
nb_results <- evaluate_model(nb_fit, test_data, response_col = "result")
nn_results <- evaluate_model(nn_fit, test_data, response_col = "result")
lda_results <- evaluate_model(lda_fit, test_data, response_col = "result")
multinom_results <- evaluate_model(multinom_fit, test_data, response_col = "result")

# Compare all results
all_results <- bind_rows(
  log_reg_results %>% mutate(model = "Logistic Regression"),
  tree_results %>% mutate(model = "Decision Tree"),
  rf_results %>% mutate(model = "Random Forest"),
  svm_results %>% mutate(model = "SVM"),
  knn_results %>% mutate(model = "k-NN"),
  nb_results %>% mutate(model = "Naive Bayes"),
  nn_results %>% mutate(model = "Neural Network"),
  lda_results %>% mutate(model = "LDA"),
  multinom_results %>% mutate(model = "Multinomial Logistic Regression")
)

# Compare models based on accuracy
best_model <- all_results %>%
  filter(.metric == "accuracy") %>%
  arrange(desc(.estimate))

# Print the best model
best_model
