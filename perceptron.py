class PerceptronClassifier(BaseEstimator, ClassifierMixin):
    
    """
    Parameters
    ----------
    Attributes
    ----------
    Notes
    -----
    See also
    --------
    Examples
    --------
    """
    # Constructor for the classifier object
    def __init__(self, in_dim, out_dim, hidden_units, layers, 
                 learning_rate = 0.01, weight_decay = 0, epochs = -1):

        """Setup a Perceptron classifier .
        Parameters
        ----------
        Returns
        -------
        """     

        # Initialise ranomd state if set
        self.random_state = random_state
        
        # Initialise class variabels
        
    
    # The fit function to train a classifier
    def fit(self, X, y):
        # WRITE CODE HERE
        
        return
    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        # WRITE CODE HERE
        return
    # The predict_proba function to make a set of predictions for a set of query instances. This returns a set of class distributions.
    def predict_proba(self, X):
        # WRITE CODE HERE
        return