import json
import os
import sys

# Reconfigure stdout to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Define the question-answer pairs
eval_data = []

# 1. Factual Questions (9 questions)
factual_questions = [
    {
        "question": "What does the abbreviation LMS stand for in the context of the LMS update rule, and what is another name for this rule?",
        "gold_answer": "LMS stands for 'least mean squares'. The LMS update rule is also known as the Widrow-Hoff learning rule.",
        "gold_chunk_id": "chunk_001",
        "question_type": "factual",
        "difficulty": "easy"
    },
    {
        "question": "Why does batch gradient descent on the cost function J for linear regression always converge to the global minimum (assuming the learning rate is not too large)?",
        "gold_answer": "Batch gradient descent always converges to the global minimum because the optimization problem for linear regression has only one global optimum and no other local optima, since the cost function J is a convex quadratic function.",
        "gold_chunk_id": "chunk_002",
        "question_type": "factual",
        "difficulty": "medium"
    },
    {
        "question": "In the context of locally weighted linear regression, how is a non-parametric learning algorithm defined, and how does it differ from a parametric learning algorithm?",
        "gold_answer": "A parametric learning algorithm has a fixed, finite number of parameters that are fit to the data, allowing the training data to be discarded once parameters are learned. A non-parametric algorithm requires keeping the training data (or an amount of information that grows linearly with the size of the training set) to represent the hypothesis and make future predictions.",
        "gold_chunk_id": "chunk_008",
        "question_type": "factual",
        "difficulty": "medium"
    },
    {
        "question": "What is the 'positive class' and 'negative class' in a binary classification problem, and what symbols are sometimes used to denote them?",
        "gold_answer": "In binary classification, the positive class corresponds to the label 1 (sometimes denoted by the symbol '+'), and the negative class corresponds to the label 0 (sometimes denoted by the symbol '-').",
        "gold_chunk_id": "chunk_009",
        "question_type": "factual",
        "difficulty": "easy"
    },
    {
        "question": "What does the principal of maximum likelihood prescribe for choosing the parameters θ given a probabilistic model?",
        "gold_answer": "The principle of maximum likelihood prescribes choosing the parameters θ so as to make the observed data as high probability as possible, which means maximizing the likelihood function L(θ).",
        "gold_chunk_id": "chunk_006",
        "question_type": "factual",
        "difficulty": "easy"
    },
    {
        "question": "In the context of generative learning algorithms, what is the key conceptual difference between discriminative and generative learning algorithms in terms of what they model?",
        "gold_answer": "Discriminative learning algorithms model the conditional distribution p(y|x) of the labels given the features directly. Generative learning algorithms instead model the joint distribution by modeling p(x|y) (the distribution of features given the class labels) and p(y) (the class prior probabilities).",
        "gold_chunk_id": "chunk_021",
        "question_type": "factual",
        "difficulty": "medium"
    },
    {
        "question": "What is the canonical parameter (also called the natural parameter) of a distribution in the exponential family?",
        "gold_answer": "The canonical parameter (also called the natural parameter) of a distribution in the exponential family is denoted by η (eta).",
        "gold_chunk_id": "chunk_017",
        "question_type": "factual",
        "difficulty": "easy"
    },
    {
        "question": "In the context of modern neural networks, how does the 1-D convolution module (Conv1D-S) relate to a standard matrix multiplication?",
        "gold_answer": "The Conv1D-S (1-D convolution with shared parameters) module can be viewed as a matrix multiplication Conv1D-S(z) = Qz, where the matrix Q contains shared parameters (the convolution weights) arranged in a structured, banded Toeplitz-like matrix.",
        "gold_chunk_id": "chunk_045",
        "question_type": "factual",
        "difficulty": "medium"
    },
    {
        "question": "What are the two passes that make up the backpropagation algorithm for computing gradients in a neural network?",
        "gold_answer": "The backpropagation algorithm consists of two passes: the forward pass, which computes the activations of each layer, and the backward pass, which propagates the error gradients backward from the output layer to the input.",
        "gold_chunk_id": "chunk_051",
        "question_type": "factual",
        "difficulty": "easy"
    }
]

# 2. Formula Questions (9 questions)
formula_questions = [
    {
        "question": "Write down the closed-form equation to find the value of θ that minimizes J(θ) in linear regression, also known as the normal equations solution.",
        "gold_answer": "θ = (X^T * X)^-1 * X^T * y",
        "gold_chunk_id": "chunk_004",
        "question_type": "formula",
        "difficulty": "easy"
    },
    {
        "question": "What is the formula for the logistic function (also called the sigmoid function) g(z), and what is its derivative g'(z) in terms of g(z)?",
        "gold_answer": "g(z) = 1 / (1 + e^-z) and g'(z) = g(z)(1 - g(z))",
        "gold_chunk_id": "chunk_010",
        "question_type": "formula",
        "difficulty": "easy"
    },
    {
        "question": "What is the formula used to calculate the non-negative weights w(i) for a query point x in locally weighted linear regression?",
        "gold_answer": "w(i) = exp(-(x(i) - x)^2 / (2 * τ^2))",
        "gold_chunk_id": "chunk_008",
        "question_type": "formula",
        "difficulty": "easy"
    },
    {
        "question": "What is the formula for the logistic loss function ℓ_logistic(t, y) for a logit t and a label y in {0, 1}?",
        "gold_answer": "ℓ_logistic(t, y) = y * log(1 + exp(-t)) + (1 - y) * log(1 + exp(t))",
        "gold_chunk_id": "chunk_011",
        "question_type": "formula",
        "difficulty": "medium"
    },
    {
        "question": "What is the formula for the conditional probability P(y = i | x; θ) in multi-class classification using the softmax distribution?",
        "gold_answer": "P(y = i | x; θ) = exp(θ_i^T * x) / ∑_{j=1}^k exp(θ_j^T * x)",
        "gold_chunk_id": "chunk_014",
        "question_type": "formula",
        "difficulty": "easy"
    },
    {
        "question": "Write the general formula defining a class of distributions in the exponential family in terms of the natural parameter η, sufficient statistic T(y), and log partition function a(η).",
        "gold_answer": "p(y; η) = b(y) * exp(η^T * T(y) - a(η))",
        "gold_chunk_id": "chunk_017",
        "question_type": "formula",
        "difficulty": "easy"
    },
    {
        "question": "What is the formula for the Laplace smoothed parameter estimate φ_{j|y=1} in a Naive Bayes classifier (under the multivariate Bernoulli event model)?",
        "gold_answer": "φ_{j|y=1} = (1 + ∑_{i=1}^n 1{x_j(i) = 1 ∧ y(i) = 1}) / (2 + ∑_{i=1}^n 1{y(i) = 1})",
        "gold_chunk_id": "chunk_030",
        "question_type": "formula",
        "difficulty": "medium"
    },
    {
        "question": "What is the update formula for the multidimensional generalization of Newton's method (Newton-Raphson method) for maximizing a log-likelihood function ℓ(θ)?",
        "gold_answer": "θ := θ - H^-1 * ∇_θ ℓ(θ)",
        "gold_chunk_id": "chunk_015",
        "question_type": "formula",
        "difficulty": "medium"
    },
    {
        "question": "What is the mathematical formula for the GELU activation function in terms of the error function (erf)?",
        "gold_answer": "σ(z) = (z / 2) * [1 + erf(z / sqrt(2))]",
        "gold_chunk_id": "chunk_040",
        "question_type": "formula",
        "difficulty": "medium"
    }
]

# 3. Comparative Questions (9 questions)
comparative_questions = [
    {
        "question": "Compare Batch Gradient Descent and Stochastic Gradient Descent (SGD) in terms of their training data pass requirements per parameter step and their convergence behavior.",
        "gold_answer": "Batch gradient descent scans the entire training set to compute the gradient before taking a single parameter step, which is computationally expensive for large datasets. In contrast, stochastic gradient descent updates parameters using only a single training example at a time, allowing it to make progress immediately and run much faster on large training sets, though its parameters may oscillate around the minimum rather than converging exactly.",
        "gold_chunk_id": "chunk_003",
        "question_type": "comparative",
        "difficulty": "medium"
    },
    {
        "question": "Compare parametric and non-parametric learning algorithms in terms of how their model representations scale with the training set size.",
        "gold_answer": "A parametric learning algorithm (like standard linear regression) fits a fixed, finite number of parameters to the data, and the training data can be discarded after learning. A non-parametric algorithm (like locally weighted linear regression) requires keeping the entire training set around to make predictions, because the amount of data needed to represent the hypothesis grows linearly with the size of the training set.",
        "gold_chunk_id": "chunk_008",
        "question_type": "comparative",
        "difficulty": "medium"
    },
    {
        "question": "Compare Newton's method and Gradient Descent for optimization in terms of their rate of convergence and computation costs per iteration.",
        "gold_answer": "Newton's method typically converges in much fewer iterations than gradient descent because it enjoys faster convergence. However, each iteration of Newton's method is computationally more expensive than gradient descent for large feature dimensions because it requires calculating and inverting a d x d Hessian matrix.",
        "gold_chunk_id": "chunk_016",
        "question_type": "comparative",
        "difficulty": "medium"
    },
    {
        "question": "Compare Gaussian Discriminant Analysis (GDA) and Logistic Regression in terms of their modeling assumptions, data efficiency, and robustness.",
        "gold_answer": "Gaussian Discriminant Analysis (GDA) makes stronger modeling assumptions and is more data efficient (requiring less training data to learn well) when these assumptions are correct or approximately correct. Logistic regression makes weaker assumptions and is more robust to deviations from modeling assumptions. In the limit of large datasets, when the data is non-Gaussian, logistic regression will almost always perform better than GDA.",
        "gold_chunk_id": "chunk_025",
        "question_type": "comparative",
        "difficulty": "medium"
    },
    {
        "question": "Compare the Bernoulli event model and the Multinomial event model in the context of Naive Bayes text classification in terms of feature representation and word generation.",
        "gold_answer": "In the Bernoulli event model, the features are binary indicators representing the presence or absence of each dictionary word in the email, and words are generated independently according to Bernoulli distributions. In the Multinomial event model, the features are integers representing the identity of each word in the email from a vocabulary, and words are generated sequentially from a single multinomial distribution over the vocabulary, where the document length is the number of words rather than the size of the vocabulary.",
        "gold_chunk_id": "chunk_031",
        "question_type": "comparative",
        "difficulty": "hard"
    },
    {
        "question": "Compare Sigmoid/Tanh activation functions with modern activation functions like ReLU, leaky ReLU, and GELU regarding the gradient vanishing problem.",
        "gold_answer": "Sigmoid and tanh activation functions are bounded from both sides, causing their gradients to vanish as the input z approaches positive or negative infinity. In contrast, modern activation functions like ReLU, leaky ReLU, and GELU are unbounded as input goes to positive infinity, preventing gradient vanishing in that direction, and variants like leaky ReLU and GELU maintain a non-zero gradient even for negative inputs.",
        "gold_chunk_id": "chunk_040",
        "question_type": "comparative",
        "difficulty": "medium"
    },
    {
        "question": "Compare the 1-D convolution module with shared parameters (Conv1D-S) and generic matrix multiplication in terms of computation time complexity and parameter count.",
        "gold_answer": "A Conv1D-S convolution operation requires O(km) computation time and has only k parameters (where k is the kernel size and m is the input dimension). In contrast, a generic matrix multiplication mapping from Rm to Rm requires O(m^2) computation time and has m^2 parameters. Thus, convolution is significantly more computationally efficient and parameter-efficient due to parameter sharing.",
        "gold_chunk_id": "chunk_045",
        "question_type": "comparative",
        "difficulty": "hard"
    },
    {
        "question": "Compare Gaussian Discriminant Analysis (GDA) and Naive Bayes in terms of the types of input features they are designed to handle.",
        "gold_answer": "Gaussian Discriminant Analysis (GDA) is designed for continuous, real-valued input feature vectors, whereas the Naive Bayes classifier is designed for discrete-valued features (such as binary word-occurrence indicators or text vocabulary indices).",
        "gold_chunk_id": "chunk_026",
        "question_type": "comparative",
        "difficulty": "easy"
    },
    {
        "question": "Compare a two-layer neural network with linear activation functions and one with non-linear activation functions in terms of representational capacity.",
        "gold_answer": "If a neural network uses linear activation functions (σ(z) = z), the composition of multiple layers collapses mathematically into a single linear function of the input (Wx), reducing the model's representational power to standard linear regression. Non-linear activation functions prevent this collapse and enable the network to learn complex, non-linear relationships.",
        "gold_chunk_id": "chunk_040",
        "question_type": "comparative",
        "difficulty": "medium"
    }
]

# 4. Multi-hop Questions (9 questions)
multi_hop_questions = [
    {
        "question": "Explain how the mathematical definition of the gradient descent update rule leads to the positive sign in the LMS update rule θ_j := θ_j + α(y^{(i)} - h_θ(x^{(i)}))x_j^{(i)} starting from the negative derivative of a single training example's squared error.",
        "gold_answer": "In gradient descent, we subtract the partial derivative of the cost function: θ_j := θ_j - α * ∂J/∂θ_j. For a single training example, the partial derivative of the squared error J(θ) = 1/2 * (h_θ(x) - y)^2 is (h_θ(x) - y)x_j. Substituting this into the update rule gives θ_j := θ_j - α * (h_θ(x) - y)x_j. Distributing the negative sign into the error term changes (h_θ(x) - y) to (y - h_θ(x)), resulting in the positive sign in the Widrow-Hoff LMS rule: θ_j := θ_j + α * (y - h_θ(x))x_j.",
        "gold_chunk_id": "chunk_001",
        "question_type": "multi_hop",
        "difficulty": "medium"
    },
    {
        "question": "Under what specific training set conditions will the design matrix X fail to yield a unique closed-form solution via the standard normal equation θ = (X^T * X)^-1 * X^T * y, and why?",
        "gold_answer": "The closed-form solution fails when the matrix X^T * X is not invertible. According to the text (including footnote 3), this happens under two conditions: if the number of linearly independent training examples is fewer than the number of features, or if the features themselves are not linearly independent. In these cases, the matrix inverse cannot be directly computed.",
        "gold_chunk_id": "chunk_004",
        "question_type": "multi_hop",
        "difficulty": "medium"
    },
    {
        "question": "Explain step-by-step why maximizing the log-likelihood ℓ(θ) of the target values under Gaussian noise assumptions leads to the minimization of the least-squares cost function J(θ), and why the variance σ^2 does not affect the final choice of θ.",
        "gold_answer": "The log-likelihood function is ℓ(θ) = n * log(1 / (sqrt(2 * pi) * σ)) - (1 / σ^2) * 1/2 * ∑_{i=1}^n (y(i) - θ^T * x(i))^2. The first term is a constant with respect to θ. The second term is scaled by -1 / σ^2. To maximize ℓ(θ), we must minimize the subtracted summation 1/2 * ∑_{i=1}^n (y(i) - θ^T * x(i))^2, which is the least-squares cost function J(θ). Because the parameter σ^2 acts purely as a positive scaling factor on the term being minimized, the optimal θ minimizing the sum of squares remains the same regardless of the value of σ^2.",
        "gold_chunk_id": "chunk_006",
        "question_type": "multi_hop",
        "difficulty": "hard"
    },
    {
        "question": "Derive the derivative of the sigmoid function g(z) = 1 / (1 + e^-z) step-by-step, showing how the quotient or chain rule application simplifies to the product of g(z) and 1 - g(z).",
        "gold_answer": "Applying the derivative rules to g(z) = (1 + e^-z)^-1, we get g'(z) = -(1 + e^-z)^-2 * (-e^-z) = e^-z / (1 + e^-z)^2. This expression can be factored into [1 / (1 + e^-z)] * [e^-z / (1 + e^-z)]. Since e^-z / (1 + e^-z) = (1 + e^-z - 1) / (1 + e^-z) = 1 - 1 / (1 + e^-z), we substitute g(z) = 1 / (1 + e^-z) to get g'(z) = g(z)(1 - g(z)).",
        "gold_chunk_id": "chunk_010",
        "question_type": "multi_hop",
        "difficulty": "easy"
    },
    {
        "question": "Explain how expressing the Bernoulli distribution in the exponential family form mathematically reveals the connection between its natural parameter η and the logistic sigmoid function.",
        "gold_answer": "By writing the Bernoulli probability mass function p(y; φ) as an exponential, we get exp(y * log(φ / (1 - φ)) + log(1 - φ)). Matching this to the canonical exponential family form b(y) * exp(η^T * T(y) - a(η)) with T(y) = y shows that the natural parameter η = log(φ / (1 - φ)). Solving this relation for φ in terms of η yields φ = 1 / (1 + e^-η), which is exactly the logistic sigmoid function of the natural parameter η.",
        "gold_chunk_id": "chunk_017",
        "question_type": "multi_hop",
        "difficulty": "hard"
    },
    {
        "question": "Explain how the scale-invariance property of the simplified LayerNorm (LN-S) mathematically leads to the scale invariance of the full LayerNorm (LN) under a positive scaling factor α, and what implication this has for modern deep learning architectures.",
        "gold_answer": "The simplified LayerNorm is scale-invariant, i.e., LN-S(α * z) = LN-S(z) because both the mean and standard deviation scale by α, which cancels out in the normalization formula. As a result, the full LayerNorm satisfies LN(MM_{α * W, α * b}(z)) = β + γ * LN-S(α * MMW,b(z)) = β + γ * LN-S(MMW,b(z)) = LN(MM_{W, b}(z)). The implication for modern deep learning architectures is that scaling all weight matrices (except the last layer's weights, which are not followed by normalization layers) by a positive factor α > 0 does not change the network's final output.",
        "gold_chunk_id": "chunk_044",
        "question_type": "multi_hop",
        "difficulty": "hard"
    },
    {
        "question": "Using the definition of the cross-entropy loss ℓ_ce(t, y) = -log(exp(t_y) / ∑_j exp(t_j)) where t_j = θ_j^T * x, show how the partial derivative of this loss with respect to a single model parameter vector θ_i is derived using the chain rule.",
        "gold_answer": "By applying basic calculus, the partial derivative of the cross-entropy loss with respect to the logit t_i is first derived as ∂ℓ_ce(t,y)/∂t_i = φ_i - 1{y = i}, where φ_i is the softmax probability. Under the parameterization t_i = θ_i^T * x, the derivative of t_i with respect to θ_i is ∂t_i/∂θ_i = x. Applying the chain rule, the gradient of the loss with respect to θ_i is the product of these two partial derivatives, which yields ∂ℓ_ce/∂θ_i = ∂ℓ_ce/∂t_i * ∂t_i/∂θ_i = (φ_i - 1{y = i})x.",
        "gold_chunk_id": "chunk_014",
        "question_type": "multi_hop",
        "difficulty": "hard"
    },
    {
        "question": "Explain GDA makes predictions using a decision boundary, how this decision boundary relates to the class probability p(y=1|x) = 0.5, and how the shape and orientation of the contours for the two classes in GDA are determined by their parameters.",
        "gold_answer": "In GDA, class predictions are made using a straight decision boundary where p(y=1|x) = 0.5. On one side of this boundary, the model predicts y=1 as the most likely outcome, and on the other, it predicts y=0. The contours of the two Gaussian distributions representing the two classes have the same shape and orientation because they share a single covariance matrix Σ, but their centers are offset because they have different class-specific means μ_0 and μ_1.",
        "gold_chunk_id": "chunk_024",
        "question_type": "multi_hop",
        "difficulty": "medium"
    },
    {
        "question": "Explain why a naive implementation of the backward function for a vector-to-vector element-wise activation module M(z) = σ(z) seems to require O(m^2) computation time based on its Jacobian matrix representation, and show how the properties of the derivative reduce this time complexity to O(m).",
        "gold_answer": "Naively, the backward function B[σ, z](v) is represented by multiplying an m x m Jacobian matrix of partial derivatives by a vector v, which requires O(m^2) operations. However, because σ is applied element-wise, the partial derivative ∂σ(z_j)/∂z_i is zero for all j ≠ i. This simplifies the Jacobian to a diagonal matrix diag(σ'(z_1), ..., σ'(z_m)). Multiplying this diagonal matrix by v is mathematically equivalent to the element-wise product (Hadamard product) σ'(z) ⊙ v, which only requires O(m) operations, matching the computational efficiency of the forward pass.",
        "gold_chunk_id": "chunk_053",
        "question_type": "multi_hop",
        "difficulty": "hard"
    }
]

# 5. Unanswerable Questions (9 questions)
unanswerable_questions = [
    {
        "question": "How does the Adam optimizer's update rule adjust the learning rate using both first and second moments of the gradients, and how is bias correction applied in the initial steps?",
        "gold_answer": "This is not covered in the source material.",
        "gold_chunk_id": None,
        "question_type": "unanswerable",
        "difficulty": "medium"
    },
    {
        "question": "Explain how the dropout regularization technique modifies the forward and backward passes of a neural network during training, and how the weights are scaled at test time.",
        "gold_answer": "This is not covered in the source material.",
        "gold_chunk_id": None,
        "question_type": "unanswerable",
        "difficulty": "medium"
    },
    {
        "question": "In deep reinforcement learning, how does the Deep Q-Network (DQN) algorithm stabilize training using an experience replay buffer and a target network?",
        "gold_answer": "This is not covered in the source material.",
        "gold_chunk_id": None,
        "question_type": "unanswerable",
        "difficulty": "medium"
    },
    {
        "question": "What is the mathematical formulation of the attention mechanism in Transformer models, specifically the scaled dot-product attention equation?",
        "gold_answer": "This is not covered in the source material.",
        "gold_chunk_id": None,
        "question_type": "unanswerable",
        "difficulty": "medium"
    },
    {
        "question": "How does the K-Means clustering algorithm partition a dataset into K clusters, and what is its optimization objective function?",
        "gold_answer": "This is not covered in the source material.",
        "gold_chunk_id": None,
        "question_type": "unanswerable",
        "difficulty": "easy"
    },
    {
        "question": "Explain how the support vector machine (SVM) optimization problem is formulated using Lagrange multipliers to find the optimal separating hyperplane in the dual space.",
        "gold_answer": "This is not covered in the source material.",
        "gold_chunk_id": None,
        "question_type": "unanswerable",
        "difficulty": "hard"
    },
    {
        "question": "What are the mathematical equations for the forward and backward passes of a Long Short-Term Memory (LSTM) cell, and how do the gates prevent the vanishing gradient problem?",
        "gold_answer": "This is not covered in the source material.",
        "gold_chunk_id": None,
        "question_type": "unanswerable",
        "difficulty": "hard"
    },
    {
        "question": "How does the AdaBoost algorithm dynamically adjust training sample weights across successive iterations to combine weak classifiers into a strong ensemble?",
        "gold_answer": "This is not covered in the source material.",
        "gold_chunk_id": None,
        "question_type": "unanswerable",
        "difficulty": "medium"
    },
    {
        "question": "Explain how Principal Component Analysis (PCA) computes the eigenvectors of the data covariance matrix to project high-dimensional data into a lower-dimensional subspace.",
        "gold_answer": "This is not covered in the source material.",
        "gold_chunk_id": None,
        "question_type": "unanswerable",
        "difficulty": "medium"
    }
]

# Combine all lists
raw_questions = (
    factual_questions +
    formula_questions +
    comparative_questions +
    multi_hop_questions +
    unanswerable_questions
)

# Structure dictionaries in exact requested key order
all_questions = []
type_counts = {}
difficulties = ["easy", "medium", "hard"]
question_types = ["factual", "formula", "comparative", "multi_hop", "unanswerable"]

for idx, item in enumerate(raw_questions):
    qid = f"q{idx+1:03d}"
    
    # Check type and difficulty
    assert item["question_type"] in question_types, f"Invalid type: {item['question_type']}"
    assert item["difficulty"] in difficulties, f"Invalid difficulty: {item['difficulty']}"
    
    # Check gold values
    if item["question_type"] == "unanswerable":
        assert item["gold_chunk_id"] is None
        assert item["gold_answer"] == "This is not covered in the source material."
    else:
        assert item["gold_chunk_id"] is not None
        assert isinstance(item["gold_chunk_id"], str)
    
    # Construct exact ordered dictionary
    ordered_item = {
        "id": qid,
        "question": item["question"],
        "gold_answer": item["gold_answer"],
        "gold_chunk_id": item["gold_chunk_id"],
        "question_type": item["question_type"],
        "difficulty": item["difficulty"]
    }
    all_questions.append(ordered_item)
    type_counts[item["question_type"]] = type_counts.get(item["question_type"], 0) + 1

# Check final counts
assert len(all_questions) == 45
for qt in question_types:
    assert type_counts.get(qt, 0) == 9

# Save to file
os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "eval_set.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_questions, f, indent=2, ensure_ascii=False)

print("SUCCESS: eval_set.json generated and ordered successfully!")
print(f"Total question-answer pairs: {len(all_questions)}")
print("\nSummary Count by Question Type:")
for qt in question_types:
    print(f"  - {qt}: {type_counts.get(qt, 0)}")
