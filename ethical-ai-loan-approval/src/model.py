from sklearn.linear_model import LogisticRegression


def build_model() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)
