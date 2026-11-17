# Schedule Risk Predictor: interview talk track

## The 60 second pitch

I built a model that predicts schedule slip on capital projects from
signals available at about 30 percent complete. Two targets: will the
project finish more than 14 days late, and by how many days. Features are
earned value and project controls signals: SPI, CPI, change orders,
rework, dependency density, critical path ratio, float, procurement
delays. I compared logistic regression, random forest, and gradient
boosting on a held out test set. Gradient boosting won the delay
prediction at about 16 days mean absolute error. Permutation importance
put SPI, rework, and dependency density on top, which matches how project
controls teams actually read risk. It is served through a FastAPI
endpoint that returns a slip probability, a delay estimate, and a risk
band.

## Say this part before they ask

The dataset is synthetic, generated from a documented process based on
earned value relationships, because real portfolio data is proprietary.
I say that up front in the README. The point of the project is the
method: leakage free features, honest held out evaluation, and a serving
path. On real data I would expect lower numbers and I would retrain the
same pipeline on their historical exports.

## Likely questions, short answers

Why two models instead of one.
Classification answers the triage question, will it slip. Regression
answers the planning question, by how much. Teams act differently on
each, so I evaluate them separately.

How did you avoid leakage.
Every feature is a snapshot at 30 percent complete. Nothing computed from
the final outcome is allowed in. In production I would enforce point in
time correctness in the feature pipeline the same way.

Why did logistic regression beat the tree models on classification.
The generative process is mostly additive, so a linear boundary fits it
well. I kept the comparison in the report because on real data the
ranking likely flips, and picking models by test results instead of
habit is the discipline being shown.

How would you know the model is still good six months later.
Track the same metrics on completed projects as labels arrive, watch
drift in SPI and CPI distributions, and recalibrate per portfolio. Same
pattern as my production evaluation work: new failure cases become new
evaluation cases.

What would you build first at Foresight.
Point in time dataset from their schedule and progress history, a
baseline slip model, and an evaluation harness planners trust. Then
calibration and the feedback loop where planners confirm or override,
because those labels are the training data that compounds.

## Study checklist before any interview

Run generate_data.py and train.py yourself and read every line.
Be able to define SPI and CPI and say what a value of 0.8 means.
Be able to explain MAE versus RMSE and why MAE is the headline here.
Be able to explain permutation importance in one sentence.
Be able to say why the healthy sample project can predict a negative
delay (early finish) and what you would do about it in production
(clip ranges, report uncertainty bands).
Know the limitation cold: synthetic data, additive process, real data
will be noisier and needs point in time snapshots.
