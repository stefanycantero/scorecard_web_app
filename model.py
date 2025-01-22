import pickle

class ScorecardModel:
    def __init__(self, base_model, min_score=300, max_score=850):
        self.base_model = base_model
        self.min_score = min_score
        self.max_score = max_score

    def predict(self, X):
        # Generar probabilidades con el modelo base
        probabilities = self.base_model.predict(X)
        print("Probabilidades:",probabilities)
        # Escalar a la escala del scorecard
        scores = self.min_score + (self.max_score - self.min_score) * probabilities
        return scores

    def evaluate(self, X, y):
        return self.base_model.evaluate(X, y)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
