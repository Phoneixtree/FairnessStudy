# On Calibration and Fairness Study

This project focuses on Fair Machine Learning,
Our main contributions are summarized as follows:
1) Multi-class Multicalibration: In Section II, we extend the concept of multicalibration, traditionally applied to binary
classification, to address multi-class classification problems. This adaptation leverages the multi-class
framework to enhance calibration by accounting for diverse class-specific characteristics.
2) Fairness Loss: With calculable fairness loss introduced , we quantify deviations from fairness constraints
during the calibration process. By continuously tracking both fairness loss and model accuracy, we analyze the trade-offs
inherent in achieving balanced outcomes across classes . Given a fixed fairness threshold, we aim to identify
methods that optimize model accuracy while maintaining fairness within acceptable bounds. This approach ensures that
the calibration process not only enhances predictive performance but also adheres to established fairness standards.
3) Multi-class Measurement: In Definition 5-6, we introduce a novel framework to evaluate the performance of multi-
class classification models by defining new class-specific measurements. These measurements provide a more granular
understanding of the relationships between distinct classes, capturing both their similarities and differences in a weighted
manner.
