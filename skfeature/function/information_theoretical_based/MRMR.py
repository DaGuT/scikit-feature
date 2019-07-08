from skfeature.function.information_theoretical_based import LCSI
import numpy as np

def mrmr(X, y, **kwargs):
    """
    This function implements the MRMR feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy = LCSI.lcsi(X, y, gamma=0, function_name='MRMR', n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = LCSI.lcsi(X, y, gamma=0, function_name='MRMR')
    return F, J_CMI, MIfy

class MRMR:
    def __init__(self,n_selected_features=None):
        if n_selected_features!=None:
            self.n_selected_features = n_selected_features

    def fit(self,X,y):

        #just empty array of selected indeces
        support_ = []

        #we get vector-mask with selected features
        if hasattr(self, "n_selected_features"):
            support_,self.J_CMI, self.MIfy = mrmr(X,y,n_selected_features=self.n_selected_features)
        else:
            support_,self.J_CMI, self.MIfy = mrmr(X,y)

        #array of the same length as we have number of features, filled with FALSE
        self.support_=np.zeros(X.shape[-1], dtype=bool)
        #and we change false -> True on positions which features were selected
        self.support_[support_]=True

        print(support_)
        print(support_.shape)


        #we set num of features
        self.n_features_ = self.support_.sum()

        return self

    def transform(self,X):

        #we check if we have fitted it
        if not hasattr(self, "support_"):
            print("Fit your selector first! All features were returned.")
            return X

        #we extract seleted features
        return X[self.support_]

    def get_support(self):

        return self.support_
